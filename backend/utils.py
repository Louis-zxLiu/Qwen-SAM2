import os
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
    def __init__(self, model_id="facebook/sam2-hiera-tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        print(f"Loading SAM2 model: {model_id} on {self.device}...")
        
        # SAM2 requirement: transformers >= 4.45.0
        # Reference: https://huggingface.co/facebook/sam2-hiera-large
        try:
            print(f"[SAM2] Loading Model: {model_id}")
            self.model = Sam2Model.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print(f"[SAM2] Loading Processor...")
            self.processor = Sam2Processor.from_pretrained(model_id, trust_remote_code=True)
            
            print(f"Successfully loaded SAM2 from {model_id} using Sam2Model/Sam2Processor")
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
        
        input_points = [[points]] # 4D: (1, 1, N, 2)
        input_labels = [[labels]] # 3D: (1, 1, N)
        
        try:
            print(f"[SAM2] Predict called for {len(points)} points")
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

    def predict_video(self, video_path, points, labels, timestamp):
        """
        Perform video object segmentation.
        Since we don't have the full Sam2VideoPredictor from the official repo (we are using transformers),
        we implement a frame-by-frame tracking loop using the image model.
        This is a simplified approach:
        1. Segment the prompt frame.
        2. Use the mask from frame T as a prompt (box or mask) for frame T+1.
        3. Repeat for the whole video.
        
        Returns: Path to the output video with mask overlay.
        """
        import tempfile
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prompt_frame_idx = int(timestamp * fps)
        
        # Output video
        output_path = video_path.replace(".mp4", "_segmented.mp4")
        # Use moviepy for better compatibility
        # fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"[SAM2 Video] Processing {total_frames} frames. Prompt at frame {prompt_frame_idx}")
        
        # Read all frames to memory? Video might be large.
        # But we need random access for propagation (start from middle).
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
    def __init__(self, model_id="openai/whisper-tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model: {model_id} on {self.device}...")
        try:
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
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

    def generate(self, image: Image.Image, context_text: str, api_key: str = None, base_url: str = None, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
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
            final_base_url = "https://api.siliconflow.cn/v1" 
            if base_url:
                clean_base_url = base_url.strip()
                if clean_base_url:
                    # Remove trailing slashes to avoid // in URL
                    final_base_url = clean_base_url.rstrip('/')
            
            print(f"[QwenVL] Connecting to: {final_base_url}")
            print(f"[QwenVL] Model: {model_name}")
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
                model=model_name,
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
