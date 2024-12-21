from fastapi import FastAPI, UploadFile, WebSocket, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
import shutil
import os
import subprocess
import torch
from datetime import datetime
import glob

app = FastAPI()

# Define paths for input and output
BASE_OUTPUT_DIR = "output_files"
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

# Get available GPU
def get_available_gpu():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            if device_properties:
                return str(i)  # Return the first available GPU index as a string
    return None

# Example: Use an external AI model for video generation
def ai_model_process(image_path: str, audio_path: str, output_dir: str):
    try:
        # Set CUDA_VISIBLE_DEVICES environment variable dynamically
        available_gpu = get_available_gpu()
        if available_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
        else:
            raise RuntimeError("No available GPU detected.")

        # Replace the following line with the actual command to run your AI model
        subprocess.run([
            "python", "inference.py", \
            "--source_image", image_path, \
            "--driven_audio", audio_path, \
            "--preprocess", "full", \
            "--enhancer", "gfpgan", \
            "--result_dir", output_dir
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"AI model processing failed: {e}")

@app.get("/")
async def get_webpage():
    webpage = """
    <h1>영상 스트리밍 서비스</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <label for="image">이미지 업로드:</label>
        <input type="file" id="image" name="image" accept="image/*" required><br><br>
        <label for="audio">오디오 업로드:</label>
        <input type="file" id="audio" name="audio" accept="audio/*" required><br><br>
        <button type="submit">업로드 및 처리</button>
    </form>
    <div id="video-container" style="display:none;">
        <h2>생성된 비디오</h2>
        <video controls autoplay>
            <source src="" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <script>
        async function handleFormSubmit(event) {
            event.preventDefault();
            const form = event.target;
            const formData = new FormData(form);

            const response = await fetch(form.action, {
                method: form.method,
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                const videoContainer = document.getElementById('video-container');
                const videoSource = videoContainer.querySelector('source');
                videoSource.src = `/stream?output_dir=${encodeURIComponent(data.output_dir)}`;
                videoContainer.style.display = 'block';
                videoContainer.querySelector('video').load();
            } else {
                alert('파일 업로드 및 처리 중 문제가 발생했습니다.');
            }
        }

        document.querySelector('form').addEventListener('submit', handleFormSubmit);
    </script>
    """
    return HTMLResponse(content=webpage, status_code=200)

@app.post("/upload")
async def upload_files(image: UploadFile = File(...), audio: UploadFile = File(...)):
    # Generate unique output directory based on the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded files
    image_path = os.path.join(output_dir, "input_image.jpg")
    audio_path = os.path.join(output_dir, "input_audio.wav")
    with open(image_path, "wb") as img_file:
        shutil.copyfileobj(image.file, img_file)

    with open(audio_path, "wb") as audio_file:
        shutil.copyfileobj(audio.file, audio_file)

    # Process files with AI model
    try:
        ai_model_process(image_path, audio_path, output_dir)
    except RuntimeError as e:
        return {"error": str(e)}

    return {"message": "Files uploaded and processed successfully.", "output_dir": output_dir}

@app.get("/stream")
async def stream_video(output_dir: str):
    video_path = glob.glob(f"{output_dir}/*.mp4")[0]
    if not os.path.exists(video_path):
        return HTMLResponse(content="<h1>오류: 먼저 파일을 업로드하세요.</h1>", status_code=400)

    return StreamingResponse(open(video_path, "rb"), media_type="video/mp4")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        await websocket.send_text(f"Received message: {message}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI, UploadFile, WebSocket, File, Form, Request
# from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
# import shutil
# import os
# import subprocess
# import torch
# from datetime import datetime
# import glob
# import tempfile
# import psutil
# import cv2

# app = FastAPI()

# # Define paths for input and output
# BASE_OUTPUT_DIR = "output_files"
# if not os.path.exists(BASE_OUTPUT_DIR):
#     os.makedirs(BASE_OUTPUT_DIR)

# # Middleware to handle large file upload timeout
# @app.middleware("http")
# async def limit_upload_size(request: Request, call_next):
#     if "content-length" in request.headers:
#         content_length = int(request.headers["content-length"])
#         if content_length > 100 * 1024 * 1024:  # Limit file upload to 100MB
#             return JSONResponse(status_code=413, content={"detail": "File size exceeds limit."})
#     return await call_next(request)

# # Function to clear cache and temporary files
# def clear_cache_and_temp_files():
#     temp_dir = tempfile.gettempdir()
#     for root, dirs, files in os.walk(temp_dir):
#         for file in files:
#             try:
#                 os.remove(os.path.join(root, file))
#             except Exception as e:
#                 pass
#     # Optionally clear application-specific output files
#     for root, dirs, files in os.walk(BASE_OUTPUT_DIR):
#         for file in files:
#             try:
#                 os.remove(os.path.join(root, file))
#             except Exception as e:
#                 pass

# # Function to find child processes of a given process
# def find_child_processes(parent_pid):
#     try:
#         parent = psutil.Process(parent_pid)
#         children = parent.children(recursive=True)
#         return [child.pid for child in children]
#     except psutil.NoSuchProcess:
#         return []

# # Kill child processes immediately on startup
# def kill_child_processes_on_startup():
#     current_pid = os.getpid()
#     child_pids = find_child_processes(current_pid)
#     for pid in child_pids:
#         try:
#             os.kill(pid, 9)  # Force kill
#         except Exception as e:
#             pass

# kill_child_processes_on_startup()

# # Get available GPU
# def get_available_gpu():
#     if torch.cuda.is_available():
#         for i in range(torch.cuda.device_count()):
#             device_properties = torch.cuda.get_device_properties(i)
#             if device_properties:
#                 return str(i)  # Return the first available GPU index as a string
#     return None

# # Example: Use OpenCV for video encoding
# # Ensures input dimensions are divisible by macro_block_size
# def encode_video_with_opencv(image_path: str, audio_path: str, output_path: str):
#     try:
#         # Read image and adjust dimensions for compatibility
#         frame = cv2.imread(image_path)
#         height, width, _ = frame.shape
#         macro_block_size = 16
#         adjusted_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
#         adjusted_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size

#         if (width != adjusted_width or height != adjusted_height):
#             frame = cv2.resize(frame, (adjusted_width, adjusted_height))

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(output_path, fourcc, 30.0, (adjusted_width, adjusted_height))

#         for _ in range(60):  # Generate 60 frames
#             out.write(frame)

#         out.release()
#     except Exception as e:
#         raise RuntimeError(f"Video encoding failed: {e}")

# @app.get("/video")
# def get_video():
#     file_path = "path/to/large_video.mp4"
#     if not os.path.exists(file_path):
#         return JSONResponse(content={"error": "File not found."}, status_code=404)
#     return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")

# @app.get("/")
# async def get_webpage():
#     webpage = """
#     <h1>영상 스트리밍 서비스</h1>
#     <form action="/upload" method="post" enctype="multipart/form-data">
#         <label for="image">이미지 업로드:</label>
#         <input type="file" id="image" name="image" accept="image/*" required><br><br>
#         <label for="audio">오디오 업로드:</label>
#         <input type="file" id="audio" name="audio" accept="audio/*" required><br><br>
#         <button type="submit">업로드 및 처리</button>
#     </form>
#     <div id="video-container" style="display:none;">
#         <h2>생성된 비디오</h2>
#         <video controls autoplay>
#             <source src="" type="video/mp4">
#             Your browser does not support the video tag.
#         </video>
#     </div>
#     <script>
#         async function handleFormSubmit(event) {
#             event.preventDefault();
#             const form = event.target;
#             const formData = new FormData(form);

#             const response = await fetch(form.action, {
#                 method: form.method,
#                 body: formData
#             });

#             if (response.ok) {
#                 const data = await response.json();
#                 const videoContainer = document.getElementById('video-container');
#                 const videoSource = videoContainer.querySelector('source');
#                 videoSource.src = `/stream?output_dir=${encodeURIComponent(data.output_dir)}`;
#                 videoContainer.style.display = 'block';
#                 videoContainer.querySelector('video').load();
#             } else {
#                 alert('파일 업로드 및 처리 중 문제가 발생했습니다.');
#             }
#         }

#         document.querySelector('form').addEventListener('submit', handleFormSubmit);
#     </script>
#     """
#     return HTMLResponse(content=webpage, status_code=200)

# @app.post("/upload")
# async def upload_files(image: UploadFile = File(...), audio: UploadFile = File(...)):
#     # Clear cache and temp files before processing
#     clear_cache_and_temp_files()

#     # Generate unique output directory based on the current date and time
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(BASE_OUTPUT_DIR, timestamp)
#     os.makedirs(output_dir, exist_ok=True)

#     # Save uploaded files
#     image_path = os.path.join(output_dir, "input_image.jpg")
#     audio_path = os.path.join(output_dir, "input_audio.wav")
#     with open(image_path, "wb") as img_file:
#         shutil.copyfileobj(image.file, img_file)

#     with open(audio_path, "wb") as audio_file:
#         shutil.copyfileobj(audio.file, audio_file)

#     # Process files with OpenCV
#     output_video_path = os.path.join(output_dir, "output_video.mp4")
#     try:
#         encode_video_with_opencv(image_path, audio_path, output_video_path)
#     except RuntimeError as e:
#         return {"error": str(e)}

#     return {"message": "Files uploaded and processed successfully.", "output_dir": output_dir}

# @app.get("/stream")
# async def stream_video(output_dir: str):
#     video_path = glob.glob(f"{output_dir}/*.mp4")[0]
#     if not os.path.exists(video_path):
#         return HTMLResponse(content="<h1>오류: 먼저 파일을 업로드하세요.</h1>", status_code=400)

#     return StreamingResponse(open(video_path, "rb"), media_type="video/mp4")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         message = await websocket.receive_text()
#         await websocket.send_text(f"Received message: {message}")

# @app.get("/child_processes/{pid}")
# def get_child_processes(pid: int):
#     children = find_child_processes(pid)
#     return {"parent_pid": pid, "child_pids": children}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)





    