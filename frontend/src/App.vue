<template>
  <div class="container">
    <el-card class="upload-section" v-if="!videoSource">
      <template #header>
        <div class="card-header">
          <span>Upload Video for Prototype Demo</span>
        </div>
      </template>
      <div style="margin-bottom: 20px;">
        <el-collapse v-model="activeCollapse">
          <el-collapse-item title="API & Model Configuration" name="1">
            <el-form label-width="120px" status-icon>
              <el-form-item label="Base URL">
                <el-input v-model="baseUrl" placeholder="https://api.siliconflow.cn/v1 (or other OpenAI-compatible URL)" clearable />
              </el-form-item>
              <el-form-item label="API Key">
                <el-input v-model="apiKey" placeholder="Enter API Key (sk-...)" type="password" show-password clearable />
              </el-form-item>
              <el-form-item label="VLM Model">
                <el-select v-model="qwenModel" placeholder="Select or Enter Model Name" filterable allow-create style="width: 100%">
                  <el-option label="Qwen/Qwen2-VL-7B-Instruct (SiliconFlow)" value="Qwen/Qwen2-VL-7B-Instruct" />
                  <el-option label="Pro/Qwen/Qwen2-VL-7B-Instruct (SiliconFlow Pro)" value="Pro/Qwen/Qwen2-VL-7B-Instruct" />
                  <el-option label="qwen-vl-max (Aliyun)" value="qwen-vl-max" />
                  <el-option label="qwen-vl-plus (Aliyun)" value="qwen-vl-plus" />
                </el-select>
              </el-form-item>
              <el-form-item label="SAM2 Model">
                <el-select v-model="sam2Model" placeholder="Select SAM2 Model" filterable allow-create style="width: 100%">
                  <el-option label="facebook/sam2-hiera-tiny" value="facebook/sam2-hiera-tiny" />
                  <el-option label="facebook/sam2-hiera-small" value="facebook/sam2-hiera-small" />
                  <el-option label="facebook/sam2-hiera-large" value="facebook/sam2-hiera-large" />
                </el-select>
              </el-form-item>
            </el-form>
          </el-collapse-item>
        </el-collapse>
      </div>
      <el-upload
        class="upload-demo"
        drag
        action="/api/upload"
        :on-success="handleUploadSuccess"
        :on-error="handleUploadError"
        :on-progress="handleUploadProgress"
        :show-file-list="false"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          Drop file here or <em>click to upload</em>
        </div>
      </el-upload>
      <div v-if="uploadProgress > 0 && uploadProgress < 100" style="margin-top: 15px;">
        <el-progress :percentage="uploadProgress" :status="uploadProgress === 100 ? 'success' : ''" />
      </div>
    </el-card>

    <div v-else class="video-workspace">
      <el-row :gutter="20">
        <el-col :span="16">
          <el-card>
            <template #header>
               <div class="card-header">
                 <span>Video Analysis</span>
                 <el-button link @click="videoSource = null">Back to Upload</el-button>
               </div>
            </template>
            <div class="video-wrapper" ref="videoWrapper">
              <!-- Video Player -->
              <video
                ref="videoElement"
                controls
                class="video-player"
                :src="videoSource"
                @click="handleVideoClick"
              ></video>
              
              <!-- Mask Overlay -->
              <canvas ref="maskCanvas" class="mask-overlay"></canvas>
            </div>
            <div style="margin-top: 10px; text-align: center;">
              <el-text type="info">Click on an object in the video to analyze it.</el-text>
            </div>
          </el-card>
        </el-col>
        
        <el-col :span="8">
          <el-card class="info-card" v-loading="loading">
            <template #header>
              <div class="card-header">
                <span>Analysis Result</span>
                <el-button v-if="encyclopedia" size="small" @click="reset">Reset</el-button>
              </div>
            </template>
            
            <div v-if="encyclopedia">
              <h4>Audio Transcription (Whisper)</h4>
              <p class="transcription-text">{{ transcription }}</p>
              <el-divider />
              <h4>Encyclopedia (Qwen VL)</h4>
              <p class="encyclopedia-text">{{ encyclopedia }}</p>
            </div>
            <div v-else>
              <el-empty description="Click video to start analysis" />
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import axios from 'axios'

const videoSource = ref(null)
const apiKey = ref('')
const baseUrl = ref('')
const qwenModel = ref('Qwen/Qwen2-VL-7B-Instruct')
const sam2Model = ref('facebook/sam2-hiera-tiny')
const serverVideoPath = ref(null)
const uploadProgress = ref(0)
const activeCollapse = ref(['1'])
const videoElement = ref(null)
const videoWrapper = ref(null)
const maskCanvas = ref(null)
const loading = ref(false)
const encyclopedia = ref(null)
const transcription = ref(null)

const handleUploadSuccess = (response) => {
  // response is { filename: "...", path: "..." }
  // Backend mounts /temp at http://localhost:8000/temp
  videoSource.value = `http://localhost:8000/temp/${response.filename}`
  serverVideoPath.value = response.path
  uploadProgress.value = 100
  setTimeout(() => { uploadProgress.value = 0 }, 1000)
}

const handleUploadError = (err) => {
  console.error("Upload failed", err)
  alert("Upload failed. Please check backend is running and CORS is configured.")
  uploadProgress.value = 0
}

const handleUploadProgress = (event) => {
  uploadProgress.value = Math.floor(event.percent)
}

const handleVideoClick = async (event) => {
  if (loading.value) return

  const video = videoElement.value
  const rect = video.getBoundingClientRect()
  
  // Calculate displayed video content dimensions (handling object-fit: contain)
  const videoRatio = video.videoWidth / video.videoHeight
  const elementRatio = rect.width / rect.height
  
  let displayedWidth, displayedHeight, offsetX, offsetY
  
  if (elementRatio > videoRatio) {
    // Letterboxed on sides (pillarbox)
    displayedHeight = rect.height
    displayedWidth = displayedHeight * videoRatio
    offsetX = (rect.width - displayedWidth) / 2
    offsetY = 0
  } else {
    // Letterboxed on top/bottom (letterbox)
    displayedWidth = rect.width
    displayedHeight = displayedWidth / videoRatio
    offsetX = 0
    offsetY = (rect.height - displayedHeight) / 2
  }
  
  // Click coordinates relative to the video content
  const clickX = event.clientX - rect.left - offsetX
  const clickY = event.clientY - rect.top - offsetY
  
  // Ignore clicks on black bars
  if (clickX < 0 || clickX > displayedWidth || clickY < 0 || clickY > displayedHeight) {
    return
  }
  
  const timestamp = video.currentTime
  const frameWidth = video.videoWidth // Intrinsic width
  const frameHeight = video.videoHeight // Intrinsic height
  
  const scaleX = frameWidth / displayedWidth
  const scaleY = frameHeight / displayedHeight
  
  const actualX = clickX * scaleX
  const actualY = clickY * scaleY

  loading.value = true
  reset()

  try {
    const formData = new FormData()
    formData.append('video_path', serverVideoPath.value) // Use absolute path on server
    formData.append('x', actualX)
    formData.append('y', actualY)
    formData.append('timestamp', timestamp)
    formData.append('frame_width', frameWidth)
    formData.append('frame_height', frameHeight)
    if (apiKey.value) {
      formData.append('api_key', apiKey.value)
    }
    if (baseUrl.value) {
      formData.append('base_url', baseUrl.value)
    }
    formData.append('qwen_model', qwenModel.value)
    formData.append('sam2_model', sam2Model.value)

    // Direct request to backend bypassing Vite proxy for debugging
    const response = await axios.post('http://127.0.0.1:8000/predict', formData)
    
    transcription.value = response.data.transcription
    encyclopedia.value = response.data.encyclopedia
    
    drawMask(response.data.mask)
  } catch (err) {
    console.error(err)
    alert("Analysis failed: " + (err.response?.data?.detail || err.message))
  } finally {
    loading.value = false
  }
}

const drawMask = (maskDataUrl) => {
  const canvas = maskCanvas.value
  const wrapper = videoWrapper.value
  const video = videoElement.value
  
  if (!canvas || !wrapper || !video) return

  const ctx = canvas.getContext('2d')
  
  // Set canvas size to match wrapper size
  canvas.width = wrapper.clientWidth
  canvas.height = wrapper.clientHeight
  
  const img = new Image()
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Calculate video content position relative to wrapper
    // 1. Video Element position relative to Wrapper
    const vLeft = video.offsetLeft
    const vTop = video.offsetTop
    const vWidth = video.clientWidth
    const vHeight = video.clientHeight
    
    // 2. Video Content position relative to Video Element (Object Fit)
    const videoRatio = video.videoWidth / video.videoHeight
    const elementRatio = vWidth / vHeight
    
    let displayedWidth, displayedHeight, offsetX, offsetY
    
    if (elementRatio > videoRatio) {
      displayedHeight = vHeight
      displayedWidth = displayedHeight * videoRatio
      offsetX = (vWidth - displayedWidth) / 2
      offsetY = 0
    } else {
      displayedWidth = vWidth
      displayedHeight = displayedWidth / videoRatio
      offsetX = 0
      offsetY = (vHeight - displayedHeight) / 2
    }
    
    // 3. Final Draw Coordinates
    const destX = vLeft + offsetX
    const destY = vTop + offsetY
    
    // Draw mask with screen blend mode (makes black transparent)
    ctx.globalCompositeOperation = 'screen'
    ctx.globalAlpha = 0.6
    ctx.drawImage(img, destX, destY, displayedWidth, displayedHeight)
    
    // Reset context
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1.0
  }
  img.src = maskDataUrl
}

const reset = () => {
  encyclopedia.value = null
  transcription.value = null
  const canvas = maskCanvas.value
  if (canvas) {
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }
}
</script>

<style scoped>
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.upload-section {
  max-width: 600px;
  margin: 100px auto;
  text-align: center;
}
.video-workspace {
  margin-top: 20px;
}
.video-wrapper {
  position: relative;
  width: 100%;
  background: #000;
  display: flex;
  justify-content: center;
  align-items: center;
}
.video-player {
  max-width: 100%;
  max-height: 600px;
  display: block;
}
.mask-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Let clicks pass through to video */
}
.transcription-text {
  font-style: italic;
  color: #666;
}
.encyclopedia-text {
  line-height: 1.6;
}
</style>
