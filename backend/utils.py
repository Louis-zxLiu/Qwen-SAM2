import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import transformers
from transformers import (
    Sam2Processor, 
    Sam2Model,
    WhisperProcessor, 
    WhisperForConditionalGeneration
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Explicit check for transformers version
TRANSFORMERS_VERSION = transformers.__version__
print(f"Current Transformers version: {TRANSFORMERS_VERSION}")

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Moviepy v2.0+ compatibility
    from moviepy import VideoFileClip

# ... (rest of imports)

class Sam2Predictor:
    def __init__(self, model_id=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id or os.getenv("SAM2_MODEL", "facebook/sam2-hiera-tiny")
        print(f"Loading SAM2 model: {self.model_id} on {self.device}...")
        
        # SAM2 requirement: transformers >= 4.45.0
        # Reference: https://huggingface.co/facebook/sam2-hiera-large
        try:
            print(f"[SAM2] Loading Model: {self.model_id}")
            self.model = Sam2Model.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print(f"[SAM2] Loading Processor...")
            self.processor = Sam2Processor.from_pretrained(self.model_id, trust_remote_code=True)
            
            print(f"Successfully loaded SAM2 from {self.model_id} using Sam2Model/Sam2Processor")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load SAM2.")
            print(f"Transformers: {TRANSFORMERS_VERSION}, Torch: {torch.__version__}")
            print(f"Detailed Error: {e}")
            raise RuntimeError(f"SAM2 Loading Failed: {e}. Please ensure model files are fully downloaded.")

    def predict(self, frame, points, labels):
        """
        frame: numpy array (H, W, 3) BGR (cv2 default)
        points: list of [x, y] coordinates
        labels: list of int (1 for positive, 0 for negative)
        Returns: mask (H, W) binary, masked_image (PIL Image)
        """
        # Prepare inputs
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        # Reference official docs for dimension requirements:
        # input_points: 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
        # input_labels: 3 dimensions (image_dim, object_dim, point_label)
        
        # Ensure points and labels are properly formatted
        # points should be list of lists: [[x1, y1], [x2, y2], ...]
        # labels should be list of ints: [1, 0, ...]
        
        # FIX: Ensure all points and labels are standard Python types (not numpy)
        # Convert points to list of lists of floats
        safe_points = []
        for p in points:
            safe_points.append([float(p[0]), float(p[1])])
            
        # Convert labels to list of ints
        safe_labels = [int(l) for l in labels]
        
        input_points = [[safe_points]] # 4D: (1, 1, N, 2)
        input_labels = [[safe_labels]] # 3D: (1, 1, N)
        
        try:
            # print(f"[SAM2] Predict called for {len(safe_points)} points") # Suppress noisy log
            inputs = self.processor(
                images=image, 
                input_points=input_points, 
                input_labels=input_labels, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process masks
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu()
            )[0]
            
            predicted_masks = masks[0] # (num_masks, H, W)
            scores = outputs.iou_scores.cpu().numpy()
            
            # Handle score dimensions
            if len(scores.shape) == 3: # (batch, objects, masks)
                iou_scores = scores[0][0]
            elif len(scores.shape) == 2: # (batch, masks)
                iou_scores = scores[0]
            else:
                iou_scores = scores.flatten()
                
            best_mask_idx = np.argmax(iou_scores)
            best_mask = predicted_masks[best_mask_idx].numpy() # (H, W) boolean
            
            return best_mask.astype(np.uint8), image
        except Exception as e:
            print(f"[SAM2 ERROR] Inference failed: {e}")
            import traceback
            print(traceback.format_exc())
            raise e

    def propagate_in_video(self, video_path, points, labels, frame_idx, start_frame=0, end_frame=None):
        """
        Propagate segmentation mask in video within a specific frame range.
        
        Args:
            video_path: Path to input video
            points: Initial points for SAM2
            labels: Initial labels for SAM2
            frame_idx: Index of the frame where points were clicked
            start_frame: Index to start processing (inclusive)
            end_frame: Index to end processing (exclusive), None for end of video
            
        Returns: 
            (output_video_path, mask_data)
        """
        import tempfile
        try:
            # Try MoviePy v2.0+ imports (direct import)
            from moviepy import ImageSequenceClip, AudioFileClip, VideoFileClip
        except ImportError:
            # Fallback to MoviePy v1.0 imports (moviepy.editor)
            try:
                from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
            except ImportError:
                raise ImportError("MoviePy is not installed correctly. Please install moviepy.")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
            
        # Ensure range is valid
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        print(f"[SAM2 Video] Processing range: Frames {start_frame} to {end_frame} (Total video: {total_frames})")
        print(f"[SAM2 Video] Prompt at frame {frame_idx}")
        
        # We need to process frames sequentially from the prompt frame
        # Strategy:
        # 1. Forward pass: frame_idx -> end_frame
        # 2. Backward pass: frame_idx -> start_frame
        # But since we use a simple image-to-image tracking (simulated video prop), 
        # we might just do forward for prototype simplicity if prompt is at start.
        # If prompt is in middle, we need bidirectional.
        
        # For this prototype, we will implement a simple tracker:
        # We start from the prompt frame and track forward to end_frame.
        # Then we start from prompt frame and track backward to start_frame.
        # Then we combine.
        
        processed_frames = {} # frame_idx -> annotated_frame (RGB numpy)
        
        # 1. Get Prompt Frame Mask
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: raise RuntimeError("Failed to read prompt frame")
        
        # Initial prediction
        mask, _ = self.predict(frame, points, labels) # mask is (H, W) binary
        
        # Draw on prompt frame
        annotated_prompt = self.overlay_mask(frame, mask)
        processed_frames[frame_idx] = cv2.cvtColor(annotated_prompt, cv2.COLOR_BGR2RGB)
        
        current_mask = mask
        current_box = self.mask_to_box(mask) # [x, y, w, h]
        
        # 2. Forward Tracking (frame_idx + 1 -> end_frame)
        # We use a simple bbox tracking or re-prompting with center of mass for next frame
        # This is a heuristic since we don't have the stateful video predictor API in transformers yet.
        # Real SAM2 Video API would maintain memory bank.
        
        # Optimization: We only process if we have a valid mask
        if current_box:
            # Re-open cap to ensure seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
            
            for i in range(frame_idx + 1, end_frame):
                ret, frame = cap.read()
                if not ret: break
                
                # Use previous mask's center as a prompt for this frame?
                # Or use the whole box as prompt?
                # Let's use the center of the previous mask as a positive point.
                # This is a naive tracker.
                
                if current_box:
                    cx = current_box[0] + current_box[2] // 2
                    cy = current_box[1] + current_box[3] // 2
                    
                    # Predict with new point prompt
                    # We assume object doesn't move too fast
                    new_mask, _ = self.predict(frame, [[cx, cy]], [1])
                    
                    # Update state
                    current_mask = new_mask
                    current_box = self.mask_to_box(new_mask)
                    
                    # Overlay
                    annotated = self.overlay_mask(frame, new_mask)
                    processed_frames[i] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                else:
                    # Lost track
                    processed_frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Backward Tracking (frame_idx - 1 -> start_frame)
        # Reset to prompt mask
        current_mask = mask
        current_box = self.mask_to_box(mask)
        
        if current_box and start_frame < frame_idx:
             # We need to read backwards. OpenCV doesn't support reverse reading efficiently.
             # We iterate backwards by setting pos frames.
             for i in range(frame_idx - 1, start_frame - 1, -1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break
                
                if current_box:
                    cx = current_box[0] + current_box[2] // 2
                    cy = current_box[1] + current_box[3] // 2
                    
                    new_mask, _ = self.predict(frame, [[cx, cy]], [1])
                    
                    current_mask = new_mask
                    current_box = self.mask_to_box(new_mask)
                    
                    annotated = self.overlay_mask(frame, new_mask)
                    processed_frames[i] = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                else:
                    processed_frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()
        
        # 4. Assemble Video
        # We only have processed frames. What about the rest?
        # The user only wants the segment? Or the whole video with segment highlighted?
        # Usually "Trim" implies we only return the segment.
        # Let's return ONLY the segmented clip.
        
        output_frames = []
        for i in range(start_frame, end_frame):
            if i in processed_frames:
                output_frames.append(processed_frames[i])
            else:
                # If frame wasn't processed (e.g. tracking lost or gap), read raw
                cap_fill = cv2.VideoCapture(video_path)
                cap_fill.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, raw = cap_fill.read()
                cap_fill.release()
                if raw is not None:
                    output_frames.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        
        if not output_frames:
            raise RuntimeError("No frames processed")
            
        output_path = os.path.join("temp", f"segment_{int(time.time())}.mp4")
        clip = ImageSequenceClip(output_frames, fps=fps)
        
        # Add audio
        try:
            original_clip = VideoFileClip(video_path)
            # Cut audio to match range
            # MoviePy 2.0+ uses 'subclipped' or slice syntax, 1.0 uses 'subclip'
            # Let's try robust approach
            t_start = start_frame / fps
            t_end = end_frame / fps
            
            if hasattr(original_clip.audio, 'subclipped'):
                audio = original_clip.audio.subclipped(t_start, t_end)
            elif hasattr(original_clip.audio, 'subclip'):
                audio = original_clip.audio.subclip(t_start, t_end)
            else:
                # Fallback for some versions
                audio = original_clip.audio
                
            clip = clip.set_audio(audio)
        except Exception as e:
            print(f"Warning: Could not add audio to segment: {e}")
            
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        
        return output_path, "mask_data_placeholder"

    def mask_to_box(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return [xmin, ymin, xmax - xmin, ymax - ymin] # x, y, w, h

    def overlay_mask(self, image, mask, color=(0, 255, 0), alpha=0.5):
        # image is BGR
        masked = image.copy()
        masked[mask > 0] = masked[mask > 0] * (1 - alpha) + np.array(color) * alpha
        return masked.astype(np.uint8)
        # Let's read them into a list if memory allows (SAM2 usually requires high memory anyway).
        # For long videos, this might crash. But let's assume reasonable size for now.
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise RuntimeError("No frames read from video")

        # Initialize masks array
        masks = [None] * len(frames)
        
        # 1. Segment Prompt Frame
        print(f"[SAM2 Video] Segmenting prompt frame {prompt_frame_idx}...")
        prompt_mask, _ = self.predict(frames[prompt_frame_idx], points, labels)
        masks[prompt_frame_idx] = prompt_mask
        
        # 2. Propagate Forward
        current_mask = prompt_mask
        for i in range(prompt_frame_idx + 1, len(frames)):
            if i % 10 == 0: print(f"[SAM2 Video] Propagating forward frame {i}/{len(frames)}")
            
            # Use previous mask to get a bounding box or mask prompt
            # Using mask directly as prompt is better if supported.
            # Transformers Sam2Model supports 'input_masks' (batch, num_masks, H, W)
            # But we need to ensure dimensions are correct.
            # The model expects raw mask logits usually? Or boolean?
            # Actually, standard SAM uses low-res mask (256x256).
            # If we pass full res mask, we might need to resize.
            # For simplicity in this "hacky" video loop:
            # We calculate the bounding box of the previous mask and use it as a box prompt.
            # This is robust for tracking.
            
            rows, cols = np.where(current_mask > 0)
            if len(rows) > 0:
                y_min, y_max = np.min(rows), np.max(rows)
                x_min, x_max = np.min(cols), np.max(cols)
                # Add margin
                margin = 10
                # Ensure values are standard python int, not numpy.int64
                box = [
                    int(max(0, x_min - margin)),
                    int(max(0, y_min - margin)),
                    int(min(width, x_max + margin)),
                    int(min(height, y_max + margin))
                ]
                # Box format for SAM2: [[x1, y1, x2, y2]]
                # Processor expects input_boxes=[[[x1, y1, x2, y2]]]
                # Error says: expected 3 levels [image level, box level, box coordinates], got 4.
                # So we should pass input_boxes=[[box]]
                # Because images=image (1 image), so outer list is for image batch.
                # Inside is list of boxes for that image.
                # Each box is [x1, y1, x2, y2].
                
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                
                inputs = self.processor(
                    images=image, 
                    input_boxes=[[box]], 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post process
                m = self.processor.post_process_masks(
                    outputs.pred_masks.cpu(), 
                    inputs["original_sizes"].cpu()
                )[0]
                
                # Take best mask
                iou = outputs.iou_scores.cpu().numpy()
                if len(iou.shape) == 3: iou = iou[0][0]
                else: iou = iou.flatten()
                
                best_idx = np.argmax(iou)
                current_mask = m[0][best_idx].numpy().astype(np.uint8)
            else:
                # Object lost
                current_mask = np.zeros((height, width), dtype=np.uint8)
                
            masks[i] = current_mask

        # 3. Propagate Backward
        current_mask = prompt_mask
        for i in range(prompt_frame_idx - 1, -1, -1):
            if i % 10 == 0: print(f"[SAM2 Video] Propagating backward frame {i}")
            
            rows, cols = np.where(current_mask > 0)
            if len(rows) > 0:
                y_min, y_max = np.min(rows), np.max(rows)
                x_min, x_max = np.min(cols), np.max(cols)
                margin = 10
                # Ensure values are standard python int, not numpy.int64
                box = [
                    int(max(0, x_min - margin)),
                    int(max(0, y_min - margin)),
                    int(min(width, x_max + margin)),
                    int(min(height, y_max + margin))
                ]
                
                # Backward propagation box input
                rgb_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                
                inputs = self.processor(
                    images=image, 
                    input_boxes=[[box]], 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                m = self.processor.post_process_masks(
                    outputs.pred_masks.cpu(), 
                    inputs["original_sizes"].cpu()
                )[0]
                
                iou = outputs.iou_scores.cpu().numpy()
                if len(iou.shape) == 3: iou = iou[0][0]
                else: iou = iou.flatten()
                best_idx = np.argmax(iou)
                current_mask = m[0][best_idx].numpy().astype(np.uint8)
            else:
                current_mask = np.zeros((height, width), dtype=np.uint8)
                
            masks[i] = current_mask
            
        # 4. Write Video with Overlay using MoviePy (Better compatibility)
        print(f"[SAM2 Video] Writing output video with MoviePy...")
        
        # Try import ImageSequenceClip
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                # Fallback or error
                raise RuntimeError("Could not import ImageSequenceClip from moviepy")

        output_frames = []
        for i, frame in enumerate(frames):
            mask = masks[i]
            if mask is not None and np.sum(mask) > 0:
                # Create colored mask overlay (e.g., green)
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = mask * 255 # Green channel
                
                # Blend
                overlay = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
                # Convert BGR to RGB for MoviePy
                rgb_frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                output_frames.append(rgb_frame)
            else:
                # Convert original frame BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_frames.append(rgb_frame)
                
        # Create clip and write
        clip = ImageSequenceClip(output_frames, fps=fps)
        
        # Add audio from original video
        try:
            original_clip = VideoFileClip(video_path)
            if original_clip.audio:
                clip = clip.set_audio(original_clip.audio)
        except Exception as e:
            print(f"[SAM2 Video] Warning: Could not add audio: {e}")
        
        # Write to file using libx264 which is standard for web
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
        
        print(f"[SAM2 Video] Saved to {output_path}")
        return output_path

class WhisperTranscriber:
    def __init__(self, model_id=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id or os.getenv("WHISPER_MODEL", "openai/whisper-tiny")
        print(f"Loading Whisper model: {self.model_id} on {self.device}...")
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load Whisper: {e}")
            raise RuntimeError(f"Whisper Loading Failed: {e}. Please check your environment.")

    def transcribe_segment(self, video_path, start_time, duration=5.0):
        # Removed mock fallback, this will only be called if self.model exists
        # Extract audio using moviepy (no system ffmpeg required)
        try:
            video = VideoFileClip(video_path)
            # Ensure start_time and duration are within bounds
            end_time = min(start_time + duration, video.duration)
            if start_time >= video.duration:
                 start_time = max(0, video.duration - duration)
            
            # Extract subclip audio
            try:
                print(f"[Whisper] Extracting audio from {start_time:.2f}s to {end_time:.2f}s")
                # Moviepy 2.0+ uses 'subclipped' returning a copy, or 'subclip' (if available)
                if hasattr(video, 'subclipped'):
                    audio = video.subclipped(start_time, end_time).audio
                else:
                    audio = video.subclip(start_time, end_time).audio
            except Exception as e_subclip:
                print(f"[Whisper] Subclip failed: {e_subclip}")
                # Fallback: maybe just take the audio and cut it?
                audio = video.audio.subclip(start_time, end_time)
            
            # Write to temporary file (Whisper usually handles files best or raw arrays)
            # But WhisperProcessor expects raw waveform at 16kHz
            # Moviepy can export to array, but format is stereo 44.1kHz usually.
            # Easiest: save to temp wav, load with librosa? No, user wants no librosa.
            # Use moviepy to save as wav, then read with soundfile? No soundfile.
            # Use moviepy to get numpy array, resample?
            
            # Actually, moviepy's `to_soundarray` returns numpy array.
            # audio_array = audio.to_soundarray(fps=16000)
            # audio_array is (N, 2) usually. We need mono (N,).
            
            if audio is None:
                print(f"[Whisper] Error: No audio track found in {video_path}")
                return "No audio track found."

            audio_array = audio.to_soundarray(fps=16000) # Resample to 16k
            print(f"[Whisper] Audio array shape: {audio_array.shape}")
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1) # Convert to mono
            
            # Cleanup
            video.close()

            # Process
            print(f"[Whisper] Running model inference...")
            inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                # Fix for Transformers >= 4.38.0: Explicitly set language or task to avoid warnings/errors
                # Also handle attention_mask warning explicitly by nature of input_features? 
                # WhisperForConditionalGeneration.generate() usually handles this, but let's be explicit.
                # Use forced_decoder_ids for language='en' or let it detect.
                # To suppress "attention mask not set" warning, we might need to pass attention_mask if using inputs_embeds,
                # but for input_features it should be fine. The warning might be from the decoder side?
                
                # Force English transcription for consistency with Qwen prompt, or let it detect?
                # User prompt says "translate to English... pass language='en'". 
                # Let's try to detect first, but if it fails, default to English. 
                # Actually, Qwen prompt is in English ("Identify the main object..."), so English transcription is better.
                
                predicted_ids = self.model.generate(
                    input_features, 
                    attention_mask=attention_mask,
                    language="en",
                    task="transcribe",
                    forced_decoder_ids=None # Explicitly set to None to avoid conflicts with task="transcribe"
                )
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Post-process transcription to remove common hallucinations
            # Whisper often outputs "you", "Thank you", or "Bye" on silent audio.
            clean_text = transcription.strip()
            if clean_text.lower() in ["you", "thank you.", "thank you", "bye", "you."]:
                print(f"[Whisper] Detected hallucination '{clean_text}', filtering out.")
                return ""
                
            return transcription
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return f"Audio extraction failed: {str(e)}"

import base64
from io import BytesIO

from openai import OpenAI
import httpx

class QwenVLGenerator:
    def __init__(self):
        pass

    def generate(self, image: Image.Image, context_text: str, api_key: str = None, base_url: str = None, model_name: str = None):
        # Prioritize passed api_key, then env var
        final_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        # Clean API Key
        if final_api_key:
            final_api_key = final_api_key.strip()
            # Remove any potential surrounding quotes that might have been pasted
            final_api_key = final_api_key.strip('"').strip("'")
        
        if not final_api_key:
            return f"Mock Encyclopedia Entry: (No API Key provided) Based on the visual analysis and audio context '{context_text}', this appears to be an object of interest. Please enter a valid Dashscope API Key in the frontend."

        # Convert image to Data URI (base64)
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_data_uri = f"data:image/png;base64,{img_str}"
        except Exception as e:
            return f"Image processing failed: {e}"
        
        # Use OpenAI compatible client instead of Dashscope SDK
        # This provides better compatibility with custom Base URLs (e.g. OneAPI, proxies)
        try:
            # Determine Base URL
            # Default to SiliconFlow if no base_url provided, as Aliyun is explicitly removed
            final_base_url = base_url or os.getenv("DASHSCOPE_BASE_URL", "https://api.siliconflow.cn/v1")
            if final_base_url:
                clean_base_url = final_base_url.strip()
                if clean_base_url:
                    # Remove trailing slashes to avoid // in URL
                    final_base_url = clean_base_url.rstrip('/')
            
            final_model_name = model_name or os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

            print(f"[QwenVL] Connecting to: {final_base_url}")
            print(f"[QwenVL] Model: {final_model_name}")
            # Do NOT print the full API Key for security, but print length or first few chars
            masked_key = f"{final_api_key[:8]}...{final_api_key[-4:]}" if len(final_api_key) > 12 else "***"
            print(f"[QwenVL] API Key: {masked_key}")

            client = OpenAI(
                api_key=final_api_key,
                base_url=final_base_url,
                http_client=httpx.Client(verify=False) # Disable SSL verification for proxies/local tests if needed
            )

            # Map model names if necessary (e.g., qwen-vl-max -> Pro/Qwen/Qwen2-VL-7B-Instruct)
            # But user can select/type model name in frontend.
            
            prompt = f"Context from audio: {context_text}. Identify the main object in this image. Please keep the answer concise, under 100 words. Format the response with a short title, followed by 3 key bullet points."
            print(f"[QwenVL] Prompt: {prompt}")
            
            response = client.chat.completions.create(
                model=final_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_data_uri
                                }
                            }
                        ]
                    }
                ],
                # Add max_tokens to avoid timeouts or large responses
                max_tokens=256,
                stream=False
            )
            
            content = response.choices[0].message.content
            print(f"[QwenVL] Received response length: {len(content)}")
            return content
            
        except Exception as e:
            return f"Qwen VL API Error (OpenAI Client): {str(e)}"
