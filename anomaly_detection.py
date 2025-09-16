from ase.io import iread
from tqdm import tqdm
import numpy as np
import os

def check_tag_anomaly(atoms):
    """
    Check if tag value 2 (adsorbate) is at the end of the tags array
    Returns True if anomaly detected (tag 2 not at end), False otherwise
    """
    if 'tags' not in atoms.arrays:
        return False
    
    tags = atoms.arrays['tags']
    
    # tag 2가 없는 경우는 정상으로 간주
    if 2 not in tags:
        return False
    
    # tag 2의 마지막 위치 찾기
    last_tag2_idx = np.where(tags == 2)[0][-1]
    
    # 마지막 tag 2 이후에 다른 tag가 있는지 확인
    if last_tag2_idx < len(tags) - 1:
        # 마지막 tag 2 이후에 다른 값이 있으면 anomaly
        return True
    
    return False

def analyze_extxyz_file(file_path):
    """
    Analyze extxyz file for tag anomalies
    """
    anomalies = []
    total_structures = 0
    
    print(f"분석 중인 파일: {file_path}")
    
    for i, atoms in enumerate(tqdm(iread(file_path), desc="구조 분석 중")):
        total_structures += 1
        
        if check_tag_anomaly(atoms):
            # anomaly 정보 수집
            tags = atoms.arrays['tags']
            last_tag2_idx = np.where(tags == 2)[0][-1]
            
            # atom_types 정보 가져오기 (symbols 사용)
            atom_types = atoms.get_atomic_numbers()
            
            anomaly_info = {
                'structure_idx': i,
                'total_atoms': len(atoms),
                'tags': tags.copy(),
                'atom_types': atom_types,
                'last_tag2_position': last_tag2_idx,
                'tags_after_tag2': tags[last_tag2_idx+1:],
                'chemical_formula': atoms.get_chemical_formula()
            }
            anomalies.append(anomaly_info)
    
    return anomalies, total_structures

def print_anomaly_summary(anomalies, total_structures):
    """
    Print summary of detected anomalies
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION 결과")
    print("="*60)
    print(f"총 구조 수: {total_structures}")
    print(f"Anomaly 수: {len(anomalies)}")
    print(f"Anomaly 비율: {len(anomalies)/total_structures*100:.2f}%")
    
    if anomalies:
        print(f"\n첫 번째 5개 Anomaly 상세 정보:")
        for i, anomaly in enumerate(anomalies[:5]):
            print(f"\n--- Anomaly {i+1} ---")
            print(f"구조 인덱스: {anomaly['structure_idx']}")
            print(f"화학식: {anomaly['chemical_formula']}")
            print(f"총 원자 수: {anomaly['total_atoms']}")
            print(f"마지막 tag 2 위치: {anomaly['last_tag2_position']}")
            print(f"tag 2 이후 태그들: {anomaly['tags_after_tag2']}")
            print(f"전체 tags: {anomaly['tags']}")
            print(f"전체 atom_types: {anomaly['atom_types']}")
    
    return len(anomalies)

def analyze_multiple_files(directory):
    """
    Analyze multiple extxyz files in a directory
    """
    all_anomalies = []
    total_structures_all = 0
    
    # extxyz 파일들 찾기
    extxyz_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.extxyz'):
                extxyz_files.append(os.path.join(root, file))
    
    print(f"발견된 extxyz 파일 수: {len(extxyz_files)}")
    
    for file_path in extxyz_files:
        print(f"\n파일 분석 중: {file_path}")
        anomalies, total_structures = analyze_extxyz_file(file_path)
        
        # 파일별 결과 출력
        print(f"  - 총 구조: {total_structures}, Anomaly: {len(anomalies)}")
        
        all_anomalies.extend(anomalies)
        total_structures_all += total_structures
    
    # 전체 결과 요약
    print("\n" + "="*60)
    print("전체 데이터셋 Anomaly 요약")
    print("="*60)
    print(f"총 구조 수: {total_structures_all}")
    print(f"총 Anomaly 수: {len(all_anomalies)}")
    print(f"전체 Anomaly 비율: {len(all_anomalies)/total_structures_all*100:.2f}%")
    
    return all_anomalies, total_structures_all

if __name__ == "__main__":
    # 단일 파일 분석
    extxyz_file = "oc20_s2ef_2M/train/extxyz/2.extxyz"
    
    if os.path.exists(extxyz_file):
        print("단일 파일 분석 모드")
        anomalies, total_structures = analyze_extxyz_file(extxyz_file)
        print_anomaly_summary(anomalies, total_structures)
    else:
        print("전체 디렉토리 분석 모드")
        # 전체 디렉토리 분석
        anomalies, total_structures = analyze_multiple_files("oc20_s2ef_2M")
