import lmdb
import pickle
import torch
import numpy as np
from ase import Atoms
from tqdm import tqdm
from pymatgen.core import Lattice, Structure, Element
from pymatgen.io.cif import CifWriter
import os
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import lmdb, torch, pandas as pd, numpy as np
from io import StringIO
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
import json

import tempfile
from pathlib import Path
from pymatgen.io.cif import CifWriter

def structure_to_cif_string(structure):
    # 1) NamedTemporaryFile 로 임시 파일 이름 생성
    tmp = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    tmp_name = tmp.name
    tmp.close()

    # 2) CifWriter 로 이 경로에 기록
    CifWriter(structure).write_file(tmp_name)

    # 3) 파일을 읽어서 문자열로 반환
    cif_str = Path(tmp_name).read_text()

    # 4) 임시 파일 삭제
    Path(tmp_name).unlink()

    return cif_str


def row_from_oc_sample(atomic_numbers, positions, lattice, system_id, bulk_mp_id, relaxed_energy):
    """
    sample : dict   # OC20-Dense entry
    sid    : str    # 고유 system_id  (예: 'sid-000123')
    bulk_mp_id : str  # mp-... 식
    energy : float  # eV/atom 또는 eV? 사용자 정의
    """
    Z = atomic_numbers        # (N,)
    pos = positions           # (N,3) Å Cartesian
    cell = lattice          # (3,3)

    # 1) Structure 객체 (Cartesian 좌표)
    species = [Element.from_Z(int(z)).symbol for z in Z]
    struct = Structure(lattice=Lattice(matrix=cell, pbc=[True, True, True]),
                       species=species,
                       coords=pos,
                       coords_are_cartesian=True,
                       to_unit_cell=True)

    # 2) CIF 문자열
    cif_raw = structure_to_cif_string(struct)

    # 3) 공간군 정보
    try:
        # 더 관대한 대칭성 허용 오차로 분석
        sga = SpacegroupAnalyzer(struct, symprec=0.1)  # 0.01 → 0.1로 변경
        spg_num = sga.get_space_group_number()               # 원본
        conv_struct = sga.get_conventional_standard_structure()
        spg_num_conv = SpacegroupAnalyzer(conv_struct).get_space_group_number()
        cif_conv = CifWriter(conv_struct).write_string()
        
        # 디버깅을 위한 출력
        # print(f"System {system_id}: Space group = {spg_num}")
        
    except Exception as e:
        # slab·표면과 같이 대칭 없는 경우 → 전부 P1
        # print(f"System {system_id}: Exception in space group analysis: {e}")
        spg_num = spg_num_conv = 1
        cif_conv = np.nan

    # 4) 구성 열
    n_atoms = len(struct)
    elements = sorted({el.symbol for el in struct.species})

    return {
        # 인덱스 열 두 개는 pandas가 자동 생성하도록 둡니다
        "system_id": system_id,                 # ← 원하는 sid
        "bulk_id": bulk_mp_id,            # ← mp-... (없으면 np.nan)
        "energy": energy,
        "band_gap": np.nan,
        "pretty_formula": struct.composition.formula,
        "e_above_hull": np.nan,
        "elements": str(elements),        # CSV에 list 그대로 넣으려면 str()
        "num_atoms": n_atoms,
        "cif": cif_raw,
        "spacegroup.number": spg_num,
        "spacegroup.number.conv": spg_num_conv,
        "cif.conv": cif_conv,
    }


def lmdb_to_atoms(lmdb_path, mapping_path, bulk_path, adsorbate_path, structure_type='initial', max_samples=None):
  
  rows = []
  
  with open(mapping_path, 'rb') as f:
    mapping = pickle.load(f)
    
  with open(bulk_path, 'rb') as f:
    bulks_mapping = pickle.load(f)
  
  with open(adsorbate_path, 'rb') as f:
    adsorbates_mapping = pickle.load(f)
  
  refined_data_list = []
  env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
  with env.begin() as txn:
      cursor = txn.cursor()
      for i, (key, value) in enumerate(tqdm(cursor, desc="Converting LMDB to Atoms")):
          if max_samples is not None and i >= max_samples:
              break
          data = pickle.loads(value)
          atomic_numbers = data.atomic_numbers.cpu().numpy() if torch.is_tensor(data.atomic_numbers) else data.atomic_numbers
          if structure_type == 'initial':
              positions = data.pos.cpu().numpy() if torch.is_tensor(data.pos) else data.pos
          elif structure_type == 'relaxed':
              positions = data.pos_relaxed.cpu().numpy() if torch.is_tensor(data.pos_relaxed) else data.pos_relaxed
          else:
              raise ValueError("structure_type must be 'initial' or 'relaxed'")
          tags = data.tags.cpu().numpy() if torch.is_tensor(data.tags) else data.tags
          if hasattr(data, 'cell') and data.cell is not None:
              cell = data.cell.cpu().numpy() if torch.is_tensor(data.cell) else data.cell
              if len(cell.shape) == 3 and cell.shape[0] == 1:
                  cell = cell[0]
              cell = cell.astype(float)
              pbc = [True, True, False]
          else:
              cell = None
              pbc = [True, True, False]
          
          # System
          system = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
          system.arrays['tags'] = tags
          if hasattr(data, 'sid'):
              system.info['sid'] = data.sid
          if hasattr(data, 'config'):
              system.info['config'] = data.config
          if structure_type == 'initial' and hasattr(data, 'y'):
              system.info['energy'] = data.y
          elif structure_type == 'relaxed' and hasattr(data, 'y_relaxed'):
              system.info['energy'] = data.y_relaxed
          if hasattr(data, 'natoms'):
              system.info['natoms'] = data.natoms
          
          # Decomposition according to sid mapping
          meta_data = mapping[data.sid]
          
          system_id = meta_data.get('system_id')
          adsorbate_id = int(system_id.split('_')[0])
          bulk_id = int(system_id.split('_')[1])
          mpid = meta_data.get('mpid')
          miller_idx = meta_data.get('miller_idx')
          shift = meta_data.get('shift')
          top = meta_data.get('top')
          adsorbate_smiles = meta_data.get('adsorbate')
          adsorption_site = meta_data.get('adsorption_site')
          
          system_species = [Element.from_Z(int(z)).symbol for z in atomic_numbers]
          system_lattice = Lattice(matrix=cell, pbc=[True, True, True])
          system_pos = positions
          system_struct  = Structure(lattice=system_lattice, species=system_species, coords=system_pos, coords_are_cartesian=True, to_unit_cell=True)
          
          # Bulk part
          bulk_atoms = bulks_mapping[bulk_id]['atoms']
          
          bulk_species = [Element.from_Z(int(z)).symbol for z in bulk_atoms.get_atomic_numbers()]
          bulk_lattice = Lattice(matrix=cell, pbc=[True, True, True])
          bulk_pos = bulk_atoms.get_positions()
          bulk_struct  = Structure(lattice=bulk_lattice, species=bulk_species, coords=bulk_pos, coords_are_cartesian=True, to_unit_cell=True)
          
          # Slab(bulk + surface) part - 이것만 학습 데이터로 사용
          slab_mask = (system.arrays['tags'] == 0) | (system.arrays['tags'] == 1)
          slab_atoms = system[slab_mask]
          
          slab_species = [Element.from_Z(int(z)).symbol for z in slab_atoms.get_atomic_numbers()]
          slab_lattice = Lattice(matrix=cell, pbc=[True, True, True])
          slab_pos = slab_atoms.get_positions()
          slab_struct  = Structure(lattice=slab_lattice, species=slab_species, coords=slab_pos, coords_are_cartesian=True, to_unit_cell=True)
          
          
          # Adsorbate part - 조건으로만 사용
          adsorbate_atoms = adsorbates_mapping[adsorbate_id][0]
          
          # Adsorbate 정보 추출
          adsorbate_types = adsorbate_atoms.get_atomic_numbers()
          adsorbate_positions = adsorbate_atoms.get_positions()
          
          # 중심 원자(첫 번째 원자)의 원자번호
          if len(adsorbate_types) > 0:
              anchor_atomic_number = int(adsorbate_types[0])
          else:
              anchor_atomic_number = -1
          
          # adsorbate_info 생성 (원자 타입과 위치를 결합)
          adsorbate_info = [adsorbate_types.tolist(), adsorbate_positions.tolist()]
          adsorbate_info_str = json.dumps(adsorbate_info)
          
          # SLAB만을 학습 데이터로 사용 (adsorbate 제외)
          row_data = row_from_oc_sample(atomic_numbers=slab_atoms.get_atomic_numbers(), positions=slab_atoms.get_positions(), lattice=cell, system_id=system_id, bulk_mp_id=mpid, relaxed_energy=data.y_relaxed)
          
          # Adsorbate 정보는 조건으로만 추가 (학습 데이터에는 포함되지 않음)
          row_data['adsorbate_types'] = str(adsorbate_types.tolist())
          row_data['adsorbate_positions'] = str(adsorbate_positions.tolist())
          row_data['anchor_atomic_number'] = anchor_atomic_number
          row_data['adsorbate_info'] = adsorbate_info_str
          
          rows.append(row_data)
  
  df = pd.DataFrame(rows)
  df.to_csv("train.csv", index=True)
  env.close()


lmdb_path = "data/oc20_dense/lmdbs/train/train_data.lmdb"
mapping_path = "data/oc20_dense/mappings/mapping.pkl"
bulk_path = "data/oc20_dense/mappings/bulks.pkl"
adsorbate_path = "data/oc20_dense/mappings/adsorbates.pkl"
# You can set max_samples=N for testing, or None for all
lmdb_to_atoms(lmdb_path, mapping_path, bulk_path, adsorbate_path, structure_type='relaxed', max_samples=None)
# with open(output_pickle, 'wb') as f:
#     pickle.dump(refined_data_list, f)
# print(f"Saved {len(refined_data_list)} samples to {output_pickle}")