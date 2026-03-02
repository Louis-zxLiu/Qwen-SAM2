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
    x: float = Form(...),
    y: float = Form(...),
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
        
        actual_x = int(x * scale_x)
        actual_y = int(y * scale_y)
        
        print(f"--- Processing Start ---")
        print(f"[Main] Click at ({x}, {y}) on {frame_width}x{frame_height} video.")
        print(f"[Main] Mapped to frame coords: ({actual_x}, {actual_y}) in {orig_w}x{orig_h} frame.")
        
        # 3. SAM2 Inference
        print(f"[Main] Step 2: Running SAM2 Segmentation...")
        mask, masked_image = sam2_predictor.predict(frame, (actual_x, actual_y))
        
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
        _, buffer = cv2.imencode('.png', (mask * 255).astype(np.uint8))
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "mask": f"data:image/png;base64,{mask_base64}",
            "transcription": audio_text,
            "encyclopedia": encyclopedia_text
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
