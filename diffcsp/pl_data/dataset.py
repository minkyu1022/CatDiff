import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode, use_pos_index: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        # build_crystal_graph 호출하지 않고 기본 정보만 추출
        self.extract_basic_info(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def extract_basic_info(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            # CSV 파일 로드
            df = pd.read_csv(self.path)
            
            # CSV에서 직접 정보 추출 (build_crystal 호출하지 않음)
            basic_data = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting basic crystal info"):
                # CSV에서 직접 정보 가져오기
                frac_coords = np.array(eval(row['frac_coords'])) if isinstance(row['frac_coords'], str) else row['frac_coords']
                atom_types = np.array(eval(row['atom_types'])) if isinstance(row['atom_types'], str) else row['atom_types']
                lengths = np.array(eval(row['lengths'])) if isinstance(row['lengths'], str) else row['lengths']
                angles = np.array(eval(row['angles'])) if isinstance(row['angles'], str) else row['angles']

                # Original slab composition (for reference)
                slab_composition = np.array(eval(row['slab_composition'])) if isinstance(row['slab_composition'], str) else row['slab_composition']
                slab_atom_types = np.array(eval(row['slab_atom_types'])) if isinstance(row['slab_atom_types'], str) else row['slab_atom_types']
                slab_num_atoms = row['slab_num_atoms']  # Use the value from CSV directly
                
                # Padded slab composition
                slab_composition_elements = np.array(eval(row['slab_composition_elements'])) if isinstance(row['slab_composition_elements'], str) else row['slab_composition_elements']
                slab_composition_ratios = np.array(eval(row['slab_composition_ratios'])) if isinstance(row['slab_composition_ratios'], str) else row['slab_composition_ratios']
                slab_composition_mask = np.array(eval(row['slab_composition_mask'])) if isinstance(row['slab_composition_mask'], str) else row['slab_composition_mask']
                
                # Edge 정보는 dummy로 설정 (CSPNet에서 재계산)
                num_atoms = len(atom_types)
                # 원래 build_crystal_graph와 동일한 차원으로 dummy 설정
                edge_indices = np.array([], dtype=int).reshape(0, 2)  # (num_edges, 2) 형태
                to_jimages = np.array([], dtype=int).reshape(0, 3)   # (num_edges, 3) 형태
                
                # Original adsorbate 정보
                adsorbate_types = np.array(eval(row['adsorbate_types'])) if isinstance(row['adsorbate_types'], str) else row['adsorbate_types']
                adsorbate_local_distances = np.array(eval(row['adsorbate_local_distances'])) if isinstance(row['adsorbate_local_distances'], str) else row['adsorbate_local_distances']
                adsorbate_num_atoms = row['adsorbate_num_atoms']
                
                # Padded adsorbate 정보
                adsorbate_types_padded = np.array(eval(row['adsorbate_types_padded'])) if isinstance(row['adsorbate_types_padded'], str) else row['adsorbate_types_padded']
                adsorbate_types_mask = np.array(eval(row['adsorbate_types_mask'])) if isinstance(row['adsorbate_types_mask'], str) else row['adsorbate_types_mask']
                
                result_dict = {
                    'system_id': row['system_id'],
                    'cif': row.get('cif', ''),  # CIF가 없을 수도 있음
                    'graph_arrays': (
                        frac_coords, atom_types, lengths, angles, edge_indices,
                        to_jimages, num_atoms, adsorbate_types, adsorbate_local_distances, adsorbate_num_atoms,
                        slab_composition, slab_num_atoms, slab_atom_types
                    ),
                    
                    # Padded slab composition
                    'slab_composition_elements': slab_composition_elements,
                    'slab_composition_ratios': slab_composition_ratios,
                    'slab_composition_mask': slab_composition_mask,
                    'slab_num_atoms': slab_num_atoms,
                    'slab_atom_types': slab_atom_types,
                    
                    # Padded adsorbate info
                    'adsorbate_types_padded': adsorbate_types_padded,
                    'adsorbate_types_mask': adsorbate_types_mask,
                }
                
                # 속성 정보
                if prop in row.keys():
                    result_dict[prop] = row[prop]
                
                # 공간군 정보 (필요시)
                if self.use_space_group:
                    try:
                        # CSV에서 직접 공간군 정보 가져오기
                        result_dict['spacegroup'] = row.get('spacegroup', 1)
                        result_dict['wyckoff_ops'] = np.array(eval(row['wyckoff_ops'])) if 'wyckoff_ops' in row else np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
                        result_dict['anchors'] = np.array(eval(row['anchors'])) if 'anchors' in row else np.array([0])
                    except:
                        result_dict['spacegroup'] = 1
                        result_dict['wyckoff_ops'] = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
                        result_dict['anchors'] = np.array([0])
                else:
                    result_dict['spacegroup'] = 1
                
                basic_data.append(result_dict)
            
            torch.save(basic_data, save_path)
            self.cached_data = basic_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, adsorbate_types, adsorbate_local_distances, adsorbate_num_atoms,
         slab_composition, slab_num_atoms, slab_atom_types) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges) - dummy edge
            to_jimages=torch.LongTensor(to_jimages),  # dummy to_jimages
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],  # dummy 값
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
            
            # Original adsorbate info
            adsorbate_types=torch.LongTensor(adsorbate_types),
            # adsorbate_local_distances=torch.Tensor(adsorbate_local_distances).view(1, -1),  # Temporarily disabled due to variable size
            adsorbate_num_atoms=adsorbate_num_atoms,
            
            # Padded slab composition
            slab_composition_elements=torch.LongTensor(data_dict['slab_composition_elements']),      # [MAX_COMP_NUM]
            slab_composition_ratios=torch.Tensor(data_dict['slab_composition_ratios']),              # [MAX_COMP_NUM] 
            slab_composition_mask=torch.LongTensor(data_dict['slab_composition_mask']),              # [MAX_COMP_NUM]
            slab_num_atoms=data_dict['slab_num_atoms'],                                              # scalar
            slab_atom_types=torch.LongTensor(data_dict['slab_atom_types']),                         # variable length
            
            # Padded adsorbate info
            adsorbate_types_padded=torch.LongTensor(data_dict['adsorbate_types_padded']),            # [MAX_ADSO_NUM]
            adsorbate_types_mask=torch.LongTensor(data_dict['adsorbate_types_mask']),                # [MAX_ADSO_NUM]
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        # preprocess_tensors 호출하지 않고 직접 처리
        self.cached_data = self.extract_basic_info_from_tensors(crystal_array_list)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def extract_basic_info_from_tensors(self, crystal_array_list):
        # crystal_array_list에서 직접 정보 추출
        basic_data = []
        for batch_idx, crystal_array in enumerate(crystal_array_list):
            # 기본 정보 추출
            frac_coords = crystal_array['frac_coords']
            atom_types = crystal_array['atom_types']
            lengths = crystal_array['lengths']
            angles = crystal_array['angles']
            
            # Edge 정보는 dummy로 설정 (CSPNet에서 재계산)
            num_atoms = len(atom_types)
            edge_indices = np.array([], dtype=int).reshape(0, 2)  # (num_edges, 2) 형태
            to_jimages = np.array([], dtype=int).reshape(0, 3)   # (num_edges, 3) 형태
            
            # Adsorbate 정보 (있는 경우)
            adsorbate_types = crystal_array.get('adsorbate_types', [])
            adsorbate_local_distances = crystal_array.get('adsorbate_local_distances', [])
            adsorbate_num_atoms = crystal_array.get('adsorbate_num_atoms', 0)
            
            result_dict = {
                'batch_idx': batch_idx,
                'graph_arrays': (
                    frac_coords, atom_types, lengths, angles, edge_indices,
                    to_jimages, num_atoms, adsorbate_types, adsorbate_local_distances, adsorbate_num_atoms
                )
            }
            
            basic_data.append(result_dict)
        
        return basic_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, adsorbate_types, adsorbate_local_distances, adsorbate_num_atoms,
         slab_composition, slab_num_atoms, slab_atom_types) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges) - dummy edge
            to_jimages=torch.LongTensor(to_jimages),  # dummy to_jimages
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],  # dummy 값
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            adsorbate_types=torch.LongTensor(adsorbate_types),
            # adsorbate_local_distances=torch.Tensor(adsorbate_local_distances).view(1, -1),  # Temporarily disabled due to variable size
            adsorbate_num_atoms=adsorbate_num_atoms,
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
