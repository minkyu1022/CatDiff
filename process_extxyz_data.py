import pandas as pd
import numpy as np
from ase import Atoms
from tqdm import tqdm
from pymatgen.core import Lattice, Structure, Element
from pymatgen.io.cif import CifWriter
import os
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import tempfile
from pathlib import Path
from pymatgen.io.cif import CifWriter
from ase.io import iread
import json

from ase.io import write

# Constants (should match cspnet.py)
MAX_COMP_NUM = 3
MAX_ADSO_NUM = 11

def center_adsorbate_xy_only(atoms):
    """
    Canonicalize atoms by centering adsorbate in x,y directions (PBC-aware)
    
    Args:
        atoms: ASE Atoms object with 'tags' array
        
    Returns:
        new_atoms: Canonicalized Atoms object (x,y centered, z unchanged)
    """
    
    adsorbate_mask = atoms.get_array("tags")==2
    
    if not np.any(adsorbate_mask):
        return atoms  # Return unchanged if no adsorbate
    
    # Get adsorbate fractional coordinates
    ads_frac = atoms.get_scaled_positions()[adsorbate_mask]
    
    # Calculate adsorbate center (x, y only)
    center_xy = np.mean(ads_frac[:, :2], axis=0)
    
    # Calculate shift to center at (0.5, 0.5)
    shift = np.array([0.5, 0.5]) - center_xy
    
    # Apply shift to all atoms (x, y only), z unchanged
    new_atoms = atoms.copy()
    new_frac = new_atoms.get_scaled_positions()
    new_frac[:, 0] = (new_frac[:, 0] + shift[0]) % 1.0  # x wrap with PBC
    new_frac[:, 1] = (new_frac[:, 1] + shift[1]) % 1.0  # y wrap with PBC
    # z coordinate remains unchanged
    new_atoms.set_scaled_positions(new_frac)
    
    return new_atoms

def pad_slab_composition(slab_composition):
    """
    Pad slab composition to fixed length
    
    Args:
        slab_composition: [[elements], [ratios]] with variable length
        
    Returns:
        padded_elements: [MAX_COMP_NUM] with 0 padding
        padded_ratios: [MAX_COMP_NUM] with 0.0 padding
        valid_mask: [MAX_COMP_NUM] with 1 for valid, 0 for padding
    """
    elements, ratios = slab_composition
    num_elements = len(elements)
    
    # Pad elements with 0
    padded_elements = elements + [0] * (MAX_COMP_NUM - num_elements)
    padded_elements = padded_elements[:MAX_COMP_NUM]  # Ensure exact length
    
    # Pad ratios with 0.0
    padded_ratios = ratios + [0.0] * (MAX_COMP_NUM - num_elements)
    padded_ratios = padded_ratios[:MAX_COMP_NUM]  # Ensure exact length
    
    # Create valid mask
    valid_mask = [1] * num_elements + [0] * (MAX_COMP_NUM - num_elements)
    valid_mask = valid_mask[:MAX_COMP_NUM]  # Ensure exact length
    
    return padded_elements, padded_ratios, valid_mask

def pad_adsorbate_types(adsorbate_types):
    """
    Pad adsorbate atom types to fixed length
    
    Args:
        adsorbate_types: list of atomic numbers with variable length
        
    Returns:
        padded_adsorbate_types: [MAX_ADSO_NUM] with 0 padding
        adsorbate_valid_mask: [MAX_ADSO_NUM] with 1 for valid, 0 for padding
    """
    num_adsorbate = len(adsorbate_types)
    
    # Pad with 0
    padded_adsorbate_types = list(adsorbate_types) + [0] * (MAX_ADSO_NUM - num_adsorbate)
    padded_adsorbate_types = padded_adsorbate_types[:MAX_ADSO_NUM]  # Ensure exact length
    
    # Create valid mask
    adsorbate_valid_mask = [1] * num_adsorbate + [0] * (MAX_ADSO_NUM - num_adsorbate)
    adsorbate_valid_mask = adsorbate_valid_mask[:MAX_ADSO_NUM]  # Ensure exact length
    
    return padded_adsorbate_types, adsorbate_valid_mask

def row_from_extxyz_sample(atoms, system_id, energy):
    """
    extxyz 데이터로부터 CSV 행 데이터 생성 (padding 포함)
    """
    
    # Slab과 adsorbate 분리
    slab_atoms, adsorbate_atoms = separate_slab_adsorbate(atoms)
    
    # Slab composition과 multiplicity 계산
    slab_atomic_numbers = slab_atoms.get_atomic_numbers()
    
    # 원소별 개수 계산
    element_counts = {}
    for atomic_num in slab_atomic_numbers:
        element_counts[atomic_num] = element_counts.get(atomic_num, 0) + 1
    
    # 최소 공약수로 나누어 최소 비율 구하기
    counts = list(element_counts.values())
    if counts:
        from math import gcd
        from functools import reduce
        gcd_value = reduce(gcd, counts)
        
        # 최소 비율 계산
        min_ratios = {k: v // gcd_value for k, v in element_counts.items()}
        
        # Composition: [[원소번호들], [비율들]] 형태
        elements = list(min_ratios.keys())
        ratios = list(min_ratios.values())
        total_ratio = sum(ratios)
        normalized_ratios = [r / total_ratio for r in ratios]
        slab_composition = [elements, normalized_ratios]
        
        # Multiplicity: 배수
        slab_multiplicity = gcd_value
    else:
        slab_composition = [[], []]
        slab_multiplicity = 1
    
    # Pad slab composition
    padded_elements, padded_ratios, slab_valid_mask = pad_slab_composition(slab_composition)
    
    # 기존 정보
    atom_types = atoms.get_atomic_numbers()
    cell_params = atoms.cell.cellpar()
    lengths = cell_params[:3]
    angles = cell_params[3:]
    frac_coords = atoms.get_scaled_positions()
    n_atoms = len(atom_types)

    return {
        "system_id": system_id,
        "atom_types": atom_types.tolist(),  # numpy array → Python list
        "lengths": lengths.tolist(),        # numpy array → Python list
        "angles": angles.tolist(),          # numpy array → Python list
        "frac_coords": frac_coords.tolist(), # numpy array → Python list
        "energy": energy,
        "num_atoms": n_atoms,
        "spacegroup.number": 1,

        # Original slab composition (for reference)
        "slab_composition": slab_composition,  # 이미 list 형태
        
        # Padded slab composition
        "slab_composition_elements": padded_elements,      # [MAX_COMP_NUM] - 이미 list
        "slab_composition_ratios": padded_ratios,          # [MAX_COMP_NUM] - 이미 list
        "slab_composition_mask": slab_valid_mask,          # [MAX_COMP_NUM] - 이미 list
        
        "slab_num_atoms": len(slab_atomic_numbers),
        "slab_multiplicity": slab_multiplicity,
        "slab_atom_types": slab_atomic_numbers.tolist(),   # numpy array → Python list
        "slab_frac_coords": slab_atoms.get_scaled_positions().tolist(), # numpy array → Python list
    }

def separate_slab_adsorbate(atoms):
    """atoms 객체에서 slab과 adsorbate를 분리"""
    if 'tags' not in atoms.arrays:
        raise ValueError("tags 배열이 없습니다!")
    
    tags = atoms.arrays['tags']
    
    # Slab = bulk (tag 0) + surface (tag 1)
    slab_mask = (tags == 0) | (tags == 1)
    slab_atoms = atoms[slab_mask]
    
    # Adsorbate = tag 2
    adsorbate_mask = (tags == 2)
    adsorbate_atoms = atoms[adsorbate_mask]
    
    return slab_atoms, adsorbate_atoms

def read_all_sid_energy_from_txt(txt_file_path):
    """txt 파일의 모든 줄을 읽어서 (system_id, energy) 리스트 반환"""
    sid_energy_list = []
    try:
        with open(txt_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        sid = parts[0]
                        energy = float(parts[2])
                    else:
                        sid = "unknown"
                        energy = np.nan
                    sid_energy_list.append((sid, energy))
    except Exception as e:
        print(f"에너지 읽기 에러 ({txt_file_path}): {e}")
    return sid_energy_list

def process_extxyz_to_csv(extxyz_dir, txt_dir, output_csv, max_samples=None):
    """
    extxyz 파일들을 읽어서 CSV 파일로 변환
    
    Args:
        extxyz_dir: extxyz 파일들이 있는 디렉토리
        txt_dir: txt 파일들이 있는 디렉토리  
        output_csv: 출력할 CSV 파일 경로
        max_samples: 처리할 최대 샘플 수 (None이면 모두)
    """
    rows = []
    
    # extxyz 파일 목록 가져오기
    extxyz_files = [f for f in os.listdir(extxyz_dir) if f.endswith('.extxyz')]
    extxyz_files.sort(key=lambda x: int(x.split('.')[0]))  # 숫자 순으로 정렬
    
    if max_samples:
        extxyz_files = extxyz_files[:max_samples]
    
    print(f"처리할 파일 수: {len(extxyz_files)}")
    
    for extxyz_file in tqdm(extxyz_files, desc="파일 처리 중"):
        try:
            # 파일 번호 추출
            file_number = extxyz_file.split('.')[0]
            
            # extxyz 파일 경로
            extxyz_path = os.path.join(extxyz_dir, extxyz_file)
            
            # txt 파일 경로
            txt_file = f"{file_number}.txt"
            txt_path = os.path.join(txt_dir, txt_file)
            
            # 에너지 값 읽기
            sid_energy_list = read_all_sid_energy_from_txt(txt_path)
            
            # extxyz 파일에서 모든 구조 읽기
            for i, atoms in enumerate(iread(extxyz_path)):
                
                if i < len(sid_energy_list):
                    system_id, energy = sid_energy_list[i]
                else:
                    system_id, energy = "unknown", np.nan
                
                # Apply canonicalization (center adsorbate in x,y with PBC)
                atoms = center_adsorbate_xy_only(atoms)
                    
                # CSV 행 데이터 생성
                row_data = row_from_extxyz_sample(
                    atoms=atoms,
                    system_id=system_id,
                    energy=energy
                )

                # reordered_atoms = create_reordered_structure(atoms)
                # reordered_tags = reordered_atoms.arrays['tags']
                tags = atoms.arrays['tags']
                
                adsorbate_mask = (tags == 2)
                adsorbate_types = atoms.get_atomic_numbers()[adsorbate_mask]
                adsorbate_positions = atoms.get_positions()[adsorbate_mask]
                adsorbate_local_distances = atoms.get_all_distances()[adsorbate_mask][:, adsorbate_mask]
                adsorbate_num_atoms = len(adsorbate_types)
                
                # Pad adsorbate types
                padded_adsorbate_types, adsorbate_valid_mask = pad_adsorbate_types(adsorbate_types)
                
                # 중심 원자(첫 번째 원자)의 원자번호
                # if len(adsorbate_types) > 0:
                #     anchor_atomic_number = int(adsorbate_types[0])
                # else:
                #     anchor_atomic_number = -1
                
                # Original adsorbate info (for reference)
                row_data['adsorbate_types'] = adsorbate_types.tolist()  # numpy array → Python list
                row_data['adsorbate_positions'] = adsorbate_positions.tolist()  # numpy array → Python list
                row_data['adsorbate_local_distances'] = adsorbate_local_distances.tolist()  # numpy array → Python list
                row_data['adsorbate_num_atoms'] = adsorbate_num_atoms
                
                # Padded adsorbate info
                row_data['adsorbate_types_padded'] = padded_adsorbate_types      # [MAX_ADSO_NUM] - 이미 list
                row_data['adsorbate_types_mask'] = adsorbate_valid_mask          # [MAX_ADSO_NUM] - 이미 list
                
                # row_data['anchor_atomic_number'] = anchor_atomic_number
                
                rows.append(row_data)
                
        except Exception as e:
            print(f"파일 {extxyz_file} 처리 중 에러: {e}")
            continue
    
    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=True)
    print(f"CSV 파일 저장 완료: {output_csv}")
    print(f"총 {len(df)}개 샘플 처리됨")
    
    return df

# 메인 실행 부분
if __name__ == "__main__":
    # 경로 설정
    extxyz_dir = "oc20_s2ef_2M/val_id/extxyz/"
    txt_dir = "oc20_s2ef_2M/val_id/txt/"
    output_csv = "val.csv"
    
    # 처리 실행
    df = process_extxyz_to_csv(
        extxyz_dir=extxyz_dir,
        txt_dir=txt_dir,
        output_csv=output_csv,
        max_samples=None  # 모든 파일 처리, 테스트하려면 숫자 지정
    )
    
    # 결과 확인
    print("\n=== 처리 결과 ===")
    print(f"총 샘플 수: {len(df)}")
    print(f"컬럼: {list(df.columns)}")
    print("\n처음 5개 샘플:")
    print(df.head())
