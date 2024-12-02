import os
import scipy.io
import pickle
import numpy as np
import re
from tqdm import tqdm

# 데이터가 있는 경로 설정
data_dir = "/media/dataset2/joungbin/VHAP/data/obama/obama_whiteBg_staticOffset_maskBelowLine/flame_param"
output_file = "/media/dataset2/joungbin/VHAP/data/obama/obama_whiteBg_staticOffset_maskBelowLine/flame_camera_frames.pkl"

# 각 키별 데이터를 담을 dict 초기화 (각각 numpy 배열로 초기화)
all_data = {
    'shape': [],
    'tex': [],
    'exp': [],
    'pose': [],
    'cam': [],
    'light': [],
    'images': [],
    'detail': []
}

# 파일명에서 숫자를 추출하여 정렬하는 함수
def sort_by_number(file_name):
    # 숫자 패턴을 추출 (예: '123'을 추출)
    numbers = re.findall(r'\d+', file_name)
    # 숫자가 있으면 첫 번째 숫자를 기준으로, 없으면 기본적으로 0으로 반환
    return int(numbers[0]) if numbers else 0

# 디렉토리 내 모든 .mat 파일을 숫자 순서대로 정렬
mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")], key=sort_by_number)

# 디렉토리 내 모든 .mat 파일을 읽기
for file_name in tqdm(mat_files):
    # mat 파일의 전체 경로
    file_path = os.path.join(data_dir, file_name)
    
    # mat 파일 로드
    # content = scipy.io.loadmat(file_path)
    content = np.load(file_path)
    
    # 각 키에 해당하는 데이터들을 리스트에 추가
    # all_data['shape'].append(content.get('shape'))
    # all_data['tex'].append(content.get('tex'))
    # all_data['exp'].append(content.get('exp'))
    # all_data['pose'].append(content.get('pose'))
    # all_data['cam'].append(content.get('cam'))
    # all_data['light'].append(content.get('light'))
    # all_data['images'].append(content.get('images'))
    # all_data['detail'].append(content.get('detail'))
    
    all_data['shape'].append(content['shape'])
    all_data['tex'].append(content['tex'])
    all_data['exp'].append(content['exp'])
    all_data['pose'].append(content['pose'])
    all_data['cam'].append(content['cam'])
    all_data['light'].append(content['light'])
    all_data['images'].append(content['images'])
    all_data['detail'].append(content['detail'])
    breakpoint()
# 이제 각 데이터를 축 1로 concatenate
for key in tqdm(all_data):
    all_data[key] = np.concatenate(all_data[key], axis=0)

breakpoint()
# 최종 dict를 pkl 파일로 저장
with open(output_file, 'wb') as f:
    pickle.dump(all_data, f)

print(f"Data successfully concatenated and saved to {output_file}")