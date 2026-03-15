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

# Import custom utilities (to be implemented)
from utils import Sam2Predictor, WhisperTranscriber, QwenVLGenerator

from fastapi.staticfiles import StaticFiles

from starlette.requests import Request
import time

app = FastAPI()

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

@app.post("/predict")
async def predict(
    video_path: str = Form(...),
    x: Optional[float] = Form(None),
    y: Optional[float] = Form(None),
    points_json: Optional[str] = Form(None),
    labels_json: Optional[str] = Form(None),
    timestamp: float = Form(...),  # Time in seconds
    frame_width: int = Form(...),
    frame_height: int = Form(...),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
    qwen_model: str = Form("Qwen/Qwen2-VL-7B-Instruct"),
    sam2_model: str = Form("facebook/sam2-hiera-tiny")
):
    global sam2_predictor, whisper_transcriber, qwen_vl_generator
    
    try:
        # 1. Initialize models if needed
        if sam2_predictor is None:
             sam2_predictor = Sam2Predictor(model_id=sam2_model)
        elif sam2_predictor.model_id != sam2_model:
             print(f"Switching SAM2 model from {sam2_predictor.model_id} to {sam2_model}")
             sam2_predictor = Sam2Predictor(model_id=sam2_model)

        if whisper_transcriber is None:
             whisper_transcriber = WhisperTranscriber()
        if qwen_vl_generator is None:
             qwen_vl_generator = QwenVLGenerator()

        # 2. Extract Frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Could not read frame")

        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / frame_width
        scale_y = orig_h / frame_height
        
        final_points = []
        final_labels = []

        # Process new points/labels format
        if points_json and labels_json:
            try:
                raw_points = json.loads(points_json)
                raw_labels = json.loads(labels_json)
                
                if len(raw_points) != len(raw_labels):
                    raise ValueError("Points and labels length mismatch")
                
                # Optimization: Downsample points if too many (e.g. scribble)
                # SAM2 works best with fewer, well-placed points.
                # If we have > 30 points, we take every K-th point.
                if len(raw_points) > 30:
                    step = len(raw_points) // 20
                    raw_points = raw_points[::step]
                    raw_labels = raw_labels[::step]
                    print(f"[Main] Downsampled points from {len(raw_points)*step} to {len(raw_points)}")

                for p in raw_points:
                    final_points.append([int(p[0] * scale_x), int(p[1] * scale_y)])
                final_labels = raw_labels
                
                print(f"[Main] Received {len(final_points)} points via JSON.")
            except Exception as e:
                print(f"[Main] Error parsing points_json/labels_json: {e}")
        
        # Fallback to single point if no list provided
        if not final_points and x is not None and y is not None:
            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)
            final_points = [[actual_x, actual_y]]
            final_labels = [1]
            print(f"[Main] Using legacy single point: ({actual_x}, {actual_y})")
            
        if not final_points:
            raise HTTPException(status_code=400, detail="No points provided")
        
        print(f"--- Processing Start ---")
        
        # 3. SAM2 Inference
        print(f"[Main] Step 2: Running SAM2 Video Segmentation with {len(final_points)} points...")
        # Old single frame call:
        # mask, masked_image = sam2_predictor.predict(frame, final_points, final_labels)
        
        # New video call:
        output_video_path = sam2_predictor.predict_video(video_path, final_points, final_labels, timestamp)
        
        # We still need a single mask for QwenVL?
        # QwenVL usually describes the object. 
        # Let's use the mask from the *prompt frame* for QwenVL generation.
        # So we can call predict() once for the prompt frame to get the mask/image for Qwen.
        mask, masked_image = sam2_predictor.predict(frame, final_points, final_labels)
        
        # 4. Whisper Transcription
        print(f"[Main] Step 3: Running Whisper Transcription at {timestamp}s...")
        audio_text = whisper_transcriber.transcribe_segment(video_path, timestamp, duration=5.0)
        print(f"[Main] Whisper Result: {audio_text}")
        
        # 5. Qwen VL Generation
        print(f"[Main] Step 4: Running Qwen VL Encyclopedia Generation...")
        encyclopedia_text = qwen_vl_generator.generate(
            masked_image, 
            audio_text, 
            api_key=api_key, 
            base_url=base_url,
            model_name=qwen_model
        )
        print(f"[Main] Qwen Result: {encyclopedia_text[:100]}...")
        print(f"--- Processing End ---")
        
        # 6. Encode mask for response
        # We return the video path now.
        # But frontend expects "mask" as base64 image.
        # We should update frontend to accept video url.
        # For backward compatibility or immediate display, we can still return the frame mask.
        # And ADD the video url.
        
        _, buffer = cv2.imencode('.png', (mask * 255).astype(np.uint8))
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create a temp URL for the output video
        output_filename = os.path.basename(output_video_path)
        video_url = f"http://localhost:8000/temp/{output_filename}"
        
        return JSONResponse({
            "mask": f"data:image/png;base64,{mask_base64}", # Keep for Qwen logic/compatibility
            "transcription": audio_text,
            "encyclopedia": encyclopedia_text,
            "segmented_video_url": video_url # New field
        })

    except Exception as e:
        import traceback
        error_msg = f"Prediction failed: {str(e)}"
        print(f"[CRITICAL ERROR] {error_msg}")
        print(traceback.format_exc())
        # Return 500 with details for debugging
        return JSONResponse(
            status_code=500,
            content={
                "detail": error_msg,
                "traceback": traceback.format_exc()
            }
        )

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

        return {
            "svg_path": svg_path,
            "bbox": bbox,
            "description": description
        }

    except Exception as e:
        print(f"Screen analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


import subprocess

# Global config store
global_config = {
    "api_key": None,
    "base_url": "https://api.siliconflow.cn/v1",
    "qwen_model": "Qwen/Qwen2-VL-7B-Instruct",
    "sam2_model": "facebook/sam2-hiera-tiny"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
