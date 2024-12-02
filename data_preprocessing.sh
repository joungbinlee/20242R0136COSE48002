# wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install tensorflow-gpu==2.8.0
# pip install --upgrade "protobuf<=3.20.1"

# cd data_utils/face_tracking
# python convert_BFM.py
# cd ../../
# python data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4 

# python data_utils/process.py /media/dataset2/joungbin/data/Adam_Schiff/Adam_Schiff.mp4 --task 7
# python data_utils/process.py /media/dataset2/joungbin/data/Adam_Schiff/Adam_Schiff.mp4 --task 8
# python data_utils/process.py /media/dataset2/joungbin/data/Adam_Schiff/Adam_Schiff.mp4 --task 9

# CUDA_VISIBLE_DEVICES=1 python data_utils/process.py /media/dataset2/joungbin/data/Adam_Schiff/Adam_Schiff.mp4

# DATA_DIR="/media/dataset1/HDTF/data"

# # 모든 폴더에 대해 이름순으로 정렬 후 루프 실행
# for folder in $(ls -d "$DATA_DIR"/* | sort); do
#   # 폴더인지 확인
#   if [ -d "$folder" ]; then
#     # 폴더 이름 추출
#     folder_name=$(basename "$folder")
    
#     # mp4 파일 경로 생성
#     video_path="$folder/$folder_name.mp4"
    
#     # 명령어 실행
#     echo "Processing $video_path..."
#     CUDA_VISIBLE_DEVICES=1 python data_utils/process.py "$video_path"
    
#     # 실행 결과 확인
#     if [ $? -ne 0 ]; then
#       echo "Error processing $video_path"
#     fi
#   fi
# done

DATA_DIR="/media/dataset1/HDTF/data"

# 모든 폴더 목록을 이름순으로 가져오기
folders=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# 전체 폴더 수 계산
total_folders=$(echo "$folders" | wc -l)
echo "Total folders: $total_folders"
# 현재 처리 중인 순서를 저장할 변수
current_count=0

# 모든 폴더에 대해 이름순으로 정렬 후 루프 실행
for folder in $folders; do
  current_count=$((current_count + 1))
  
  # 폴더 안에 있는 파일의 개수 확인
  file_count=$(find "$folder" -maxdepth 1 -type f | wc -l)
  echo "folder: $folder"

  echo "file_count: $file_count"
  # 파일이 1개일 때만 실행
  # if [ "$file_count" -eq 1 ]; then
  if [ "$file_count" -eq 5 ]; then
    # 폴더 이름 추출
    folder_name=$(basename "$folder")
    
    # mp4 파일 경로 생성
    video_path="$folder/$folder_name.mp4"
    
    # mp4 파일이 존재하는지 확인
    if [ -f "$video_path" ]; then
      # 현재 진행 상황 출력
      echo "Processing folder $current_count of $total_folders: $video_path"
      
      # 명령어 실행
      CUDA_VISIBLE_DEVICES=1 python data_utils/process.py "$video_path"
      
      # 실행 결과 확인
      if [ $? -ne 0 ]; then
        echo "Error processing $video_path"
      fi
    else
      echo "No mp4 file found in $folder"
    fi
  fi
done
