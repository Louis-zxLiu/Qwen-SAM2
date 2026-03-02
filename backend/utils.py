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

    def predict(self, frame, point):
        """
        frame: numpy array (H, W, 3) BGR (cv2 default)
        point: (x, y) tuple
        Returns: mask (H, W) binary, masked_image (PIL Image)
        """
        # Prepare inputs
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        # Reference official docs for dimension requirements:
        # input_points: 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
        # input_labels: 3 dimensions (image_dim, object_dim, point_label)
        
        input_points = [[[[point[0], point[1]]]]] # 4D: (1, 1, 1, 2)
        input_labels = [[[1]]]                    # 3D: (1, 1, 1)
        
        print(f"[SAM2] Predict called for point: {point}")
        try:
            inputs = self.processor(
                images=image, 
                input_points=input_points, 
                input_labels=input_labels, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process masks
            # masks is a list of tensors
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu()
            )[0] # Get the first (and only) image's masks
            
            # Each object gets its own masks. Shape: (num_objects, num_masks, H, W)
            # For single point click, num_objects = 1.
            predicted_masks = masks[0] # (num_masks, H, W)
            
            # iou_scores shape: (batch_size, num_objects, num_masks)
            # We need to handle different possible output shapes from different SAM2 versions
            scores = outputs.iou_scores.cpu().numpy()
            print(f"[SAM2] IOU scores shape: {scores.shape}")
            
            # Navigate to the correct dimension based on shape
            if len(scores.shape) == 3: # (batch, objects, masks)
                iou_scores = scores[0][0]
            elif len(scores.shape) == 2: # (batch, masks)
                iou_scores = scores[0]
            else:
                iou_scores = scores.flatten()
                
            best_mask_idx = np.argmax(iou_scores)
            best_mask = predicted_masks[best_mask_idx].numpy() # (H, W) boolean
            
            print(f"[SAM2] Best mask index: {best_mask_idx}, IOU score: {iou_scores[best_mask_idx]:.4f}")
            print(f"[SAM2] Mask pixel count: {np.sum(best_mask)}")
            
            return best_mask.astype(np.uint8), image
        except Exception as e:
            print(f"[SAM2 ERROR] Inference failed: {e}")
            import traceback
            print(traceback.format_exc())
            raise e

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
            
            prompt = f"Context from audio: {context_text}. Identify the main object in this image and provide a brief encyclopedia summary."
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
                max_tokens=512,
                stream=False
            )
            
            content = response.choices[0].message.content
            print(f"[QwenVL] Received response length: {len(content)}")
            return content
            
        except Exception as e:
            return f"Qwen VL API Error (OpenAI Client): {str(e)}"
