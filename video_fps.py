import cv2
import subprocess
from moviepy.editor import VideoFileClip
def change_fps(input_video_path, output_video_path, target_fps):
    # 동영상 파일 읽기
    cap = cv2.VideoCapture(input_video_path)

    # 원본 동영상의 속성 가져오기
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    # VideoWriter 객체 생성 (출력 파일, 코덱, FPS, 크기)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 코덱 설정
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))

    # 프레임을 읽어서 새로운 FPS로 저장
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # 자원 해제
    cap.release()
    out.release()

def change_fps_ffmpeg(input_video_path, output_video_path, target_fps):
    # FFmpeg 명령어를 실행하여 FPS 변경
    # command = [
    #     'ffmpeg',
    #     '-i', input_video_path, 
    #     '-r', str(target_fps), 
    #     '-c:v', 'libx264',  # H.264 코덱 사용
    #     '-crf', '18',       # 화질 설정 (낮을수록 화질이 좋음)
    #     '-loglevel', 'quiet',  # 로그 출력 최소화
    #     '-preset', 'veryslow',  # 인코딩 속도 (느릴수록 효율적인 압축)
    #     output_video_path
    # ]
    command = f"ffmpeg -i '{input_video_path}' -r {target_fps} -c:v libx264 -crf 18 -loglevel quiet '{output_video_path}'"


    subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)


def change_fps_moviepy(input_video_path, output_video_path, target_fps):
    # 동영상 파일을 불러오기
    clip = VideoFileClip(input_video_path)
    
    # FPS 변경
    new_clip = clip.set_fps(target_fps)
    
    # 동영상 저장
    new_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', verbose=False)

# import os
# from tqdm import tqdm
# video_list = os.listdir('/media/dataset1/HDTF/original_video')
# # 사용 예시
# video_list = sorted(video_list)
# target_fps = 25  # 원하는 FPS로 설정
# for video in tqdm(video_list):
#     input_video = os.path.join('/media/dataset1/HDTF/original_video',video)
#     output_video = os.path.join('/media/dataset1/HDTF/25fps_video',video)
#     change_fps_moviepy(input_video, output_video, target_fps)
    
# input_video = 'input_video.mp4'
# output_video = 'output_video.mp4'

# ffmpeg -i '{input_video_path}' -r {target_fps} -c:v libx264 -crf 18 -loglevel quiet -preset veryslow '{output_video_path}'"


import os
from tqdm import tqdm
import shutil

# 파일 복사 함수
def copy_file(source_path, destination_path):
    shutil.copy(source_path, destination_path)  # 파일 복사
    # print(f"File copied from {source_path} to {destination_path}")

# 사용 예시
source = 'source_file.txt'  # 복사할 파일 경로
destination = 'destination_folder/copy_file.txt'  # 복사한 파일의 대상 경로



video_list = os.listdir('/media/dataset1/HDTF/original_video')
video_list = sorted(video_list)
for video in tqdm(video_list):
    os.makedirs(os.path.join('/media/dataset1/HDTF/data',video.rstrip('.mp4')), exist_ok=True)
    # 사용 예시
    source = os.path.join('/media/dataset1/HDTF/25fps_video',video) # 복사할 파일 경로
    destination = os.path.join('/media/dataset1/HDTF/data',video.rstrip('.mp4'),video)  # 복사한 파일의 대상 경로
    copy_file(source, destination)