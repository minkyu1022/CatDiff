"""
OC20 Catalyst Dataset Canonicalization Module

이 모듈은 OC20 catalyst dataset의 canonicalization을 수행합니다.
- Supercell 생성 (2x2x1)
- Adsorbate를 원점으로 이동
- 전체 시스템의 무게중심을 0으로 조정
- 원래 unit cell을 supercell 중앙으로 이동
"""

import numpy as np
from ase.atoms import Atoms
from ase.io import iread, write
from typing import Tuple, Optional
import warnings

class OC20Canonicalizer:
    """OC20 데이터셋을 위한 canonicalization 클래스"""
    
    def __init__(self):
        pass
    
    def identify_adsorbate(self, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slab과 adsorbate를 구분합니다.
        
        Args:
            atoms: ASE Atoms 객체
            
        Returns:
            slab_mask: slab 원자들의 마스크 (True/False 배열)
            adsorbate_mask: adsorbate 원자들의 마스크 (True/False 배열)
        """
        if 'tags' not in atoms.arrays:
            raise ValueError("Tags 정보가 없습니다. OC20 데이터셋에는 tags가 필요합니다.")
        
        tags = atoms.get_array('tags')
        
        # OC20에서 tag 2가 adsorbate, tag 0,1이 slab
        adsorbate_mask = (tags == 2)
        slab_mask = (tags != 2)
        
        return slab_mask, adsorbate_mask
    
    def create_supercell(self, atoms: Atoms, repeat: Tuple[int, int, int] = (2, 2, 1)) -> Atoms:
        """
        Supercell을 생성합니다.
        
        Args:
            atoms: 원본 ASE Atoms 객체
            repeat: 반복 횟수 (nx, ny, nz)
            
        Returns:
            supercell: 생성된 supercell ASE Atoms 객체
        """
        supercell = atoms.repeat(repeat)
        return supercell
    
    def find_adsorbate_positions(self, atoms: Atoms) -> np.ndarray:
        """
        Adsorbate 원자들의 위치를 찾습니다.
        
        Args:
            atoms: ASE Atoms 객체
            
        Returns:
            adsorbate_positions: adsorbate 원자들의 위치 배열
        """
        _, adsorbate_mask = self.identify_adsorbate(atoms)
        
        if not np.any(adsorbate_mask):
            raise ValueError("Adsorbate를 찾을 수 없습니다.")
        
        positions = atoms.get_positions()
        adsorbate_positions = positions[adsorbate_mask]
        
        return adsorbate_positions
    
    def get_adsorbate_center_xy(self, adsorbate_positions: np.ndarray) -> np.ndarray:
        """
        Adsorbate의 x,y 평면에서의 무게중심을 계산합니다.
        
        Args:
            adsorbate_positions: adsorbate 원자들의 위치
            
        Returns:
            center_xy: x,y 평면에서의 무게중심 [x, y]
        """
        center_xy = np.mean(adsorbate_positions[:, :2], axis=0)
        return center_xy
    
    def translate_to_center_adsorbate(self, atoms: Atoms) -> Atoms:
        """
        Adsorbate의 x,y 좌표가 0이 되도록 전체 시스템을 translation합니다.
        
        Args:
            atoms: ASE Atoms 객체
            
        Returns:
            translated_atoms: translation된 ASE Atoms 객체
        """
        # 복사본 생성
        translated_atoms = atoms.copy()
        
        # Adsorbate 위치 찾기
        adsorbate_positions = self.find_adsorbate_positions(atoms)
        
        # Adsorbate의 x,y 무게중심 계산
        adsorbate_center_xy = self.get_adsorbate_center_xy(adsorbate_positions)
        
        # 전체 시스템을 adsorbate 중심만큼 이동 (x,y만, z는 0)
        translation_vector = np.array([-adsorbate_center_xy[0], -adsorbate_center_xy[1], 0.0])
        
        # 모든 원자 위치를 이동
        new_positions = translated_atoms.get_positions() + translation_vector
        translated_atoms.set_positions(new_positions)
        
        return translated_atoms
    
    def center_system_mass(self, atoms: Atoms) -> Atoms:
        """
        전체 시스템의 무게중심을 0으로 맞춥니다.
        
        Args:
            atoms: ASE Atoms 객체
            
        Returns:
            centered_atoms: 무게중심이 조정된 ASE Atoms 객체
        """
        centered_atoms = atoms.copy()
        
        # 전체 시스템의 무게중심 계산
        positions = centered_atoms.get_positions()
        center_of_mass = np.mean(positions, axis=0)
        
        # 무게중심만큼 이동
        new_positions = positions - center_of_mass
        centered_atoms.set_positions(new_positions)
        
        return centered_atoms
    
    def find_closest_adsorbate_to_origin(self, supercell: Atoms) -> Tuple[int, np.ndarray]:
        """
        Supercell에서 원점에 가장 가까운 adsorbate 그룹을 찾습니다.
        
        Args:
            supercell: supercell ASE Atoms 객체
            
        Returns:
            closest_group_idx: 가장 가까운 adsorbate 그룹의 인덱스
            group_center: 해당 그룹의 중심 위치
        """
        _, adsorbate_mask = self.identify_adsorbate(supercell)
        adsorbate_positions = supercell.get_positions()[adsorbate_mask]
        
        if len(adsorbate_positions) == 0:
            raise ValueError("Supercell에서 adsorbate를 찾을 수 없습니다.")
        
        # 원본 unit cell의 adsorbate 개수 추정
        # supercell이 2x2x1이므로 4배가 되어야 함
        original_adsorbate_count = len(adsorbate_positions) // 4
        
        if len(adsorbate_positions) % 4 != 0:
            warnings.warn(f"Adsorbate 개수가 예상과 다릅니다. 총 {len(adsorbate_positions)}개")
            original_adsorbate_count = len(adsorbate_positions) // 4
        
        # 각 adsorbate 그룹의 중심 계산
        group_centers = []
        for i in range(4):  # 2x2 = 4개 그룹
            start_idx = i * original_adsorbate_count
            end_idx = start_idx + original_adsorbate_count
            
            if end_idx <= len(adsorbate_positions):
                group_positions = adsorbate_positions[start_idx:end_idx]
                group_center = np.mean(group_positions, axis=0)
                group_centers.append(group_center)
        
        group_centers = np.array(group_centers)
        
        # 원점에서 가장 가까운 그룹 찾기 (x,y 평면에서만)
        distances_to_origin = np.linalg.norm(group_centers[:, :2], axis=1)
        closest_group_idx = np.argmin(distances_to_origin)
        
        return closest_group_idx, group_centers[closest_group_idx]
    
    def canonicalize_structure(self, atoms: Atoms, repeat: Tuple[int, int, int] = (2, 2, 1)) -> Atoms:
        """
        OC20 구조에 대해 전체 canonicalization을 수행합니다.
        
        단계:
        1. Supercell 생성
        2. Supercell에서 원점에 가장 가까운 adsorbate를 원점으로 이동
        3. 전체 시스템의 무게중심을 0으로 조정
        4. 원래 unit cell을 supercell 중앙으로 이동
        
        Args:
            atoms: 원본 ASE Atoms 객체
            repeat: supercell 반복 횟수
            
        Returns:
            canonical_atoms: canonicalization된 ASE Atoms 객체
        """
        # 1. Supercell 생성
        supercell = self.create_supercell(atoms, repeat)
        
        # 2. 원점에 가장 가까운 adsorbate 그룹 찾기
        closest_group_idx, group_center = self.find_closest_adsorbate_to_origin(supercell)
        
        # 3. 해당 adsorbate 그룹을 원점으로 이동 (x,y만)
        translation_vector = np.array([-group_center[0], -group_center[1], 0.0])
        new_positions = supercell.get_positions() + translation_vector
        supercell.set_positions(new_positions)
        
        # 4. 전체 시스템의 무게중심을 0으로 조정
        canonical_atoms = self.center_system_mass(supercell)
        
        # 5. 원래 unit cell을 supercell 중앙으로 이동
        # supercell의 크기 계산
        cell = canonical_atoms.get_cell()
        cell_center = np.array([cell[0][0]/2, cell[1][1]/2, cell[2][2]/2])
        
        # 원래 unit cell 크기
        original_cell = atoms.get_cell()
        original_cell_size = np.array([original_cell[0][0], original_cell[1][1], original_cell[2][2]])
        
        # 중앙으로 이동하기 위한 추가 translation
        center_translation = cell_center - original_cell_size/2
        
        final_positions = canonical_atoms.get_positions() + center_translation
        canonical_atoms.set_positions(final_positions)
        
        return canonical_atoms
    
    def process_extxyz_file(self, input_file: str, output_file: str, 
                          repeat: Tuple[int, int, int] = (2, 2, 1), 
                          max_structures: Optional[int] = None):
        """
        extxyz 파일 전체를 처리하여 canonicalization을 적용합니다.
        
        Args:
            input_file: 입력 extxyz 파일 경로
            output_file: 출력 extxyz 파일 경로
            repeat: supercell 반복 횟수
            max_structures: 처리할 최대 구조 개수 (None이면 전체)
        """
        canonical_structures = []
        
        print(f"Processing {input_file}...")
        
        for i, atoms in enumerate(iread(input_file)):
            if max_structures is not None and i >= max_structures:
                break
                
            if i % 100 == 0:
                print(f"  Processing structure {i}...")
            
            try:
                canonical_atoms = self.canonicalize_structure(atoms, repeat)
                canonical_structures.append(canonical_atoms)
                
            except Exception as e:
                print(f"  Warning: Failed to process structure {i}: {e}")
                # 원본 구조를 그대로 추가
                canonical_structures.append(atoms)
        
        # 결과 저장
        print(f"Saving {len(canonical_structures)} structures to {output_file}...")
        write(output_file, canonical_structures)
        print("Done!")


def main():
    """사용 예시"""
    # Canonicalizer 초기화
    canonicalizer = OC20Canonicalizer()
    
    # 단일 구조 테스트
    from ase.io import read
    
    # 예시 파일 경로 (실제 경로로 변경 필요)
    test_file = "/home/minkyu/DiffCSP/oc20_s2ef_2M/train/extxyz/0.extxyz"
    
    # 첫 번째 구조 읽기
    atoms_iter = iread(test_file)
    atoms = next(atoms_iter)
    
    print("=== 원본 구조 정보 ===")
    print(f"원자 수: {len(atoms)}")
    print(f"화학식: {atoms.get_chemical_formula()}")
    print(f"셀 크기: {atoms.get_cell().cellpar()}")
    
    # Canonicalization 적용
    canonical_atoms = canonicalizer.canonicalize_structure(atoms)
    
    print("\n=== Canonicalized 구조 정보 ===")
    print(f"원자 수: {len(canonical_atoms)}")
    print(f"화학식: {canonical_atoms.get_chemical_formula()}")
    print(f"셀 크기: {canonical_atoms.get_cell().cellpar()}")
    
    # Adsorbate 위치 확인
    _, adsorbate_mask = canonicalizer.identify_adsorbate(canonical_atoms)
    adsorbate_positions = canonical_atoms.get_positions()[adsorbate_mask]
    adsorbate_center_xy = canonicalizer.get_adsorbate_center_xy(adsorbate_positions)
    
    print(f"Adsorbate 중심 (x,y): {adsorbate_center_xy}")
    
    # 전체 시스템 무게중심 확인
    all_positions = canonical_atoms.get_positions()
    system_center = np.mean(all_positions, axis=0)
    print(f"전체 시스템 무게중심: {system_center}")


if __name__ == "__main__":
    main()

