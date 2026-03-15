import os
import shutil
import base64
import json
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom utilities (to be implemented)
from utils import Sam2Predictor, WhisperTranscriber, QwenVLGenerator
from plugin_system import PluginManager

from fastapi.staticfiles import StaticFiles

from starlette.requests import Request
import time

app = FastAPI()

# Initialize Plugin Manager
plugin_manager = PluginManager()

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    print(f"[ACCESS] Incoming request: {method} {path}")
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        print(f"[ACCESS] Completed: {method} {path} - Status: {response.status_code} - Time: {process_time:.2f}ms")
        return response
    except Exception as e:
        print(f"[ACCESS ERROR] Request failed: {method} {path} - Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise e

# Ensure temp directory exists before mounting
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/healthz")
async def health_check():
    return {"status": "ok", "transformers": os.environ.get("TRANSFORMERS_VERSION", "unknown")}

# Mount temp directory for static access
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize models (lazy loading or startup)
sam2_predictor = None
whisper_transcriber = None
qwen_vl_generator = None

@app.on_event("startup")
async def startup_event():
    global sam2_predictor, whisper_transcriber, qwen_vl_generator
    print("Loading models...")
    # Initialize your models here
    # sam2_predictor = Sam2Predictor()
    # whisper_transcriber = WhisperTranscriber()
    # qwen_vl_generator = QwenVLGenerator()
    print("Models loaded (placeholders active).")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.post("/process")
async def process_video(
    video_path: str = Form(...),
    points_json: str = Form(...),
    labels_json: str = Form(...),
    timestamp: float = Form(...),
    frame_width: int = Form(...),
    frame_height: int = Form(...),
    start_time: float = Form(0.0),
    end_time: float = Form(10.0),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    qwen_model: str = Form(os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")),
    sam2_model: str = Form(os.getenv("SAM2_MODEL", "facebook/sam2-hiera-tiny"))
):
    global sam2_predictor, whisper_transcriber, qwen_vl_generator

    # 1. Initialize models
    if sam2_predictor is None or sam2_predictor.model_id != sam2_model:
        sam2_predictor = Sam2Predictor(model_id=sam2_model)
    if whisper_transcriber is None:
        whisper_transcriber = WhisperTranscriber()
    if qwen_vl_generator is None:
        qwen_vl_generator = QwenVLGenerator()

    # 2. Extract Frame at Timestamp for Visual Prompt
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=400, detail="Could not read frame")

    # 3. Coordinate Scaling
    orig_h, orig_w = frame.shape[:2]
    scale_x = orig_w / frame_width
    scale_y = orig_h / frame_height
    
    raw_points = json.loads(points_json)
    raw_labels = json.loads(labels_json)
    
    scaled_points = []
    for p in raw_points:
        scaled_points.append([p[0] * scale_x, p[1] * scale_y])

    # 4. SAM2 Video Propagation (Constrained by Time Range)
    # We need to pass start/end frames to predictor
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"Processing video from {start_time}s to {end_time}s (Frames {start_frame}-{end_frame})")
    
    mask_video_path, mask_data = sam2_predictor.propagate_in_video(
        video_path, 
        points=scaled_points, 
        labels=raw_labels, 
        frame_idx=frame_idx,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    # 5. Whisper Transcription (Constrained by Time Range)
    # Extract audio segment first? Or just transcribe whole and filter?
    # Whisper usually takes audio file.
    # Let's extract audio segment using moviepy
    try:
        from moviepy import VideoFileClip
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            print("Warning: MoviePy not found for audio extraction")
            VideoFileClip = None
    
    try:
        video = VideoFileClip(video_path)
        # Ensure subclip is valid
        duration = video.duration
        t1 = max(0, start_time)
        t2 = min(duration, end_time)
        
        audio_segment_path = os.path.join("temp", f"audio_{int(time.time())}.mp3")
        
        # Handle MoviePy 1.x vs 2.x subclip
        if hasattr(video, 'subclipped'):
            segment = video.subclipped(t1, t2)
        else:
            segment = video.subclip(t1, t2)
            
        segment.audio.write_audiofile(audio_segment_path, logger=None)
        
        transcription = whisper_transcriber.transcribe(audio_segment_path)
        
        # Cleanup audio
        if os.path.exists(audio_segment_path):
            os.remove(audio_segment_path)
            
    except Exception as e:
        print(f"Audio processing failed: {e}")
        transcription = "(Audio transcription failed)"

    # 6. Qwen VL Encyclopedia
    # Use the static frame we extracted earlier
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    encyclopedia = qwen_vl_generator.generate(
        pil_image, 
        context_text=transcription,
        api_key=api_key,
        base_url=base_url,
        model_name=qwen_model
    )
    
    # Return result
    # We return the path to the MASKED video segment
    # mask_video_path is absolute, make it relative to serve
    relative_video_url = f"/temp/{os.path.basename(mask_video_path)}"
    
    return {
        "video_url": relative_video_url,
        "transcription": transcription,
        "encyclopedia": encyclopedia
    }

class ScreenAnalysisRequest(BaseModel):
    image: str # Base64 encoded
    click_x: int
    click_y: int
    mode: str = "identify" # "segment" or "identify"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    qwen_model: Optional[str] = None

@app.post("/analyze/screen")
async def analyze_screen(request: ScreenAnalysisRequest):
    global sam2_predictor, whisper_transcriber, qwen_vl_generator

    # Auto-load models if not already loaded (HUD specific)
    if sam2_predictor is None:
        print("[HUD] Lazy loading SAM2 model...")
        try:
             # Use global config sam2 model if available
             model_id = global_config.get("sam2_model", "facebook/sam2-hiera-tiny")
             sam2_predictor = Sam2Predictor(model_id=model_id)
        except Exception as e:
            print(f"[HUD] Failed to load SAM2: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load SAM2: {e}")

    if not sam2_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Decode Image
        if "," in request.image:
            header, encoded = request.image.split(",", 1)
        else:
            encoded = request.image
            
        image_data = base64.b64decode(encoded)
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        frame = np.array(pil_image)
        # Convert RGB to BGR for OpenCV/SAM2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 2. SAM2 Segmentation
        # Points: [[x, y]], Labels: [1] (positive click)
        # Use simple predict for single frame
        mask, _ = sam2_predictor.predict(
            frame, 
            points=[[request.click_x, request.click_y]], 
            labels=[1]
        )
        
        # 3. Generate SVG Path from Mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        svg_path = ""
        bbox = [0, 0, 0, 0]
        
        if contours:
            # Get the largest contour
            c = max(contours, key=cv2.contourArea)
            
            # Calculate Bounding Box
            x, y, w, h = cv2.boundingRect(c)
            bbox = [x, y, w, h]
            
            # Create SVG Path
            # Move to first point
            if len(c) > 0:
                svg_path = f"M {c[0][0][0]} {c[0][0][1]} "
                for point in c[1:]:
                    svg_path += f"L {point[0][0]} {point[0][1]} "
                svg_path += "Z"

        # 4. Qwen-VL Identification (if mode is identify)
        description = ""
        if request.mode == "identify" and bbox[2] > 0 and bbox[3] > 0:
            # Crop image with padding
            padding = 50
            h_img, w_img = frame.shape[:2]
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(w_img, bbox[0] + bbox[2] + padding)
            y2 = min(h_img, bbox[1] + bbox[3] + padding)
            
            cropped_frame = frame[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            
            # Initialize Qwen if needed
            global qwen_vl_generator
            if qwen_vl_generator is None:
                qwen_vl_generator = QwenVLGenerator()
            
            # Use model from request, or global config, or default
            model_name = request.qwen_model or global_config.get("qwen_model") or "Qwen/Qwen2-VL-7B-Instruct"
            
            description = qwen_vl_generator.generate(
                cropped_pil, 
                context_text="User clicked on screen to identify this object.",
                api_key=request.api_key,
                base_url=request.base_url,
                model_name=model_name
            )
            
        # 5. Plugin Matching
        available_plugins = []
        if description:
            available_plugins = plugin_manager.match_plugins(description)

        return {
            "svg_path": svg_path,
            "bbox": bbox,
            "description": description,
            "plugins": available_plugins
        }

    except Exception as e:
        print(f"Screen analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


import subprocess

# Global config store
global_config = {
    "api_key": os.getenv("DASHSCOPE_API_KEY"),
    "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://api.siliconflow.cn/v1"),
    "qwen_model": os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct"),
    "sam2_model": os.getenv("SAM2_MODEL", "facebook/sam2-hiera-tiny")
}

class LaunchHUDRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    qwen_model: Optional[str] = None
    sam2_model: Optional[str] = None

@app.post("/system/launch-hud")
async def launch_hud(request: LaunchHUDRequest):
    try:
        # Update global config with values from frontend
        global global_config
        if request.api_key: global_config["api_key"] = request.api_key
        if request.base_url: global_config["base_url"] = request.base_url
        if request.qwen_model: global_config["qwen_model"] = request.qwen_model
        if request.sam2_model: global_config["sam2_model"] = request.sam2_model
        
        print(f"[System] HUD Config Updated: BaseURL={global_config['base_url']}, Model={global_config['qwen_model']}")

        # Assuming backend is running in D:\Qwen-SAM2\backend
        # And electron-hud is in D:\Qwen-SAM2\electron-hud
        current_dir = os.getcwd()
        project_root = os.path.dirname(current_dir) # Go up one level
        hud_dir = os.path.join(project_root, "electron-hud")
        
        print(f"[System] Launching HUD from {hud_dir}...")
        
        if os.name == 'nt': # Windows
            cmd = f'start "HUD" cmd /c "npm start"'
            subprocess.Popen(cmd, shell=True, cwd=hud_dir)
        else: # Linux/Mac
            subprocess.Popen(["npm", "start"], cwd=hud_dir)
            
        return {"status": "success", "message": "HUD launching in background"}
    except Exception as e:
        print(f"[System] Failed to launch HUD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/config")
async def get_config():
    return global_config

class PluginExecuteRequest(BaseModel):
    plugin_id: str
    context: dict

@app.post("/plugin/execute")
async def execute_plugin(request: PluginExecuteRequest):
    return plugin_manager.execute_plugin(request.plugin_id, request.context)

class ChatRequest(BaseModel):
    image: str # Base64 encoded (Full screenshot)
    bbox: List[int] # [x, y, w, h]
    messages: List[dict] # [{"role": "user", "content": "..."}]
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    qwen_model: Optional[str] = None

@app.post("/analyze/chat")
async def analyze_chat(request: ChatRequest):
    global qwen_vl_generator
    try:
        # Decode Image
        if "," in request.image:
            header, encoded = request.image.split(",", 1)
        else:
            encoded = request.image
        image_data = base64.b64decode(encoded)
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        frame = np.array(pil_image)
        # Convert RGB to BGR for processing (if needed) but PIL is RGB
        
        # Crop Image based on BBox
        x, y, w, h = request.bbox
        if w > 0 and h > 0:
             # Add padding
            padding = 50
            h_img, w_img = frame.shape[:2]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            cropped_frame = frame[y1:y2, x1:x2]
            cropped_pil = Image.fromarray(cropped_frame) # frame is RGB from PIL
        else:
            cropped_pil = pil_image

        # Initialize Qwen if needed
        if qwen_vl_generator is None:
            qwen_vl_generator = QwenVLGenerator()
            
        model_name = request.qwen_model or global_config.get("qwen_model") or "Qwen/Qwen2-VL-7B-Instruct"
        
        # Construct Prompt from messages
        # QwenVLGenerator.generate currently only takes a single prompt context_text.
        # We need to adapt it or extend it. 
        # For now, let's take the LAST user message and prepend previous context as text.
        
        last_user_msg = request.messages[-1]['content']
        history_text = ""
        if len(request.messages) > 1:
            history_text = "Previous conversation:\n"
            for msg in request.messages[:-1]:
                history_text += f"{msg['role']}: {msg['content']}\n"
        
        final_prompt = f"{history_text}\nUser Question: {last_user_msg}\nAnswer concisely."

        reply = qwen_vl_generator.generate(
            cropped_pil, 
            context_text=final_prompt,
            api_key=request.api_key,
            base_url=request.base_url,
            model_name=model_name
        )
        
        return {"reply": reply}

    except Exception as e:
        print(f"Chat failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
