<template>
  <div class="container">
    <el-card class="upload-section" v-if="!videoSource">
      <template #header>
        <div class="card-header">
          <span>Upload Video for Prototype Demo</span>
          <el-button type="primary" size="small" @click="launchHUD" style="margin-left: auto;">
            <el-icon><Monitor /></el-icon> Launch HUD
          </el-button>
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
                crossorigin="anonymous"
                @play="onVideoPlay"
                @pause="onVideoPause"
                @seeked="onVideoSeeked"
                @loadedmetadata="videoDuration = $event.target.duration"
                @timeupdate="currentTime = $event.target.currentTime"
              ></video>
              
              <!-- Mask Overlay -->
              <canvas 
                ref="maskCanvas" 
                class="mask-overlay"
                :style="{ 
                  pointerEvents: interactionMode === 'view' ? 'none' : 'auto', 
                  cursor: interactionMode === 'view' ? 'default' : 'crosshair',
                  opacity: 1
                }"
                @mousedown="handleCanvasMouseDown"
                @mousemove="handleCanvasMouseMove"
                @mouseup="handleCanvasMouseUp"
                @mouseleave="handleCanvasMouseLeave"
              ></canvas>
            </div>

            <!-- Timeline UI -->
            <div class="timeline-container">
              <div style="display: flex; justify-content: space-between; color: #fff; font-size: 12px; margin-bottom: 5px;">
                  <span>Start: {{ formatTime(segmentRange[0]) }}</span>
                  <span>End: {{ formatTime(segmentRange[1]) }}</span>
              </div>
              <el-slider 
                  v-model="segmentRange" 
                  range 
                  :max="videoDuration" 
                  :step="0.1"
                  :format-tooltip="formatTime"
                  @change="onTimelineChange"
              />
              <div style="text-align: right; margin-top: 5px;">
                   <el-button type="warning" size="small" @click="trimVideo">
                      <el-icon><Scissor /></el-icon> Trim & Segment
                   </el-button>
              </div>
            </div>
            
            <!-- Tools Toolbar -->
            <div style="margin-top: 15px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
               <el-radio-group v-model="interactionMode" size="default">
                 <el-radio-button label="view">
                    <el-icon><VideoPlay /></el-icon> View
                 </el-radio-button>
                 <el-radio-button label="positive">
                    <el-icon><CirclePlus /></el-icon> Point (+)
                 </el-radio-button>
                 <el-radio-button label="negative">
                    <el-icon><Remove /></el-icon> Point (-)
                 </el-radio-button>
                 <el-radio-button label="scribble">
                    <el-icon><EditPen /></el-icon> Scribble
                 </el-radio-button>
               </el-radio-group>
               
               <el-divider direction="vertical" />
               <el-switch v-model="showPoints" active-text="Show Points" />
               <el-divider direction="vertical" />
               
               <el-button type="warning" @click="undoLastPoint" :disabled="accumulatedPoints.length === 0">Undo</el-button>
               <el-button type="danger" @click="clearAllPoints" :disabled="accumulatedPoints.length === 0">Clear</el-button>
               <el-button type="primary" @click="runAnalysis" :loading="loading" :disabled="accumulatedPoints.length === 0">Analyze</el-button>
            </div>

            <div style="margin-top: 10px; text-align: center;">
              <el-text type="info" v-if="interactionMode === 'view'">Switch mode to interact with the video.</el-text>
              <el-text type="info" v-else>Click or drag on the video to add {{ interactionMode }} prompts.</el-text>
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
import { ref, onMounted, nextTick, watch } from 'vue'
import { UploadFilled, VideoPlay, CirclePlus, Remove, EditPen, Monitor, Scissor } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const videoSource = ref(null)

// Timeline State
const videoDuration = ref(0)
const currentTime = ref(0)
const segmentRange = ref([0, 10]) // Default range [start, end]
const apiKey = ref('')
const baseUrl = ref('')
const qwenModel = ref('Qwen/Qwen2-VL-7B-Instruct')
const sam2Model = ref('facebook/sam2-hiera-tiny')

const formatTime = (val) => {
  const min = Math.floor(val / 60)
  const sec = Math.floor(val % 60)
  return `${min}:${sec < 10 ? '0' + sec : sec}`
}

const onTimelineChange = (val) => {
  // Seek video to start point when range changes
  if (videoElement.value) {
    videoElement.value.currentTime = val[0]
  }
}

const trimVideo = () => {
  ElMessage.info(`Trimming video from ${formatTime(segmentRange.value[0])} to ${formatTime(segmentRange.value[1])}`)
  // TODO: Send backend API to trim video
}

const launchHUD = async () => {
  try {
    const response = await fetch('http://localhost:8000/system/launch-hud', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: apiKey.value,
        base_url: baseUrl.value,
        qwen_model: qwenModel.value,
        sam2_model: sam2Model.value
      })
    })
    const data = await response.json()
    if (data.status === 'success') {
      ElMessage.success('HUD Launched! Settings synced.')
    } else {
      ElMessage.error('Failed to launch HUD: ' + data.message)
    }
  } catch (e) {
    ElMessage.error('Error launching HUD: ' + e.message)
  }
}
const serverVideoPath = ref(null)
const uploadProgress = ref(0)
const activeCollapse = ref(['1'])
const videoElement = ref(null)
const videoWrapper = ref(null)
const maskCanvas = ref(null)
const loading = ref(false)
const encyclopedia = ref(null)
const transcription = ref(null)

// Interaction State
const interactionMode = ref('view') // view, positive, negative, scribble
const accumulatedPoints = ref([]) // Array of {x, y, label}
const isDrawing = ref(false)
const currentMaskUrl = ref(null)
const isPlaying = ref(false)
const showPoints = ref(true)

// Persist settings
onMounted(() => {
  const savedBaseUrl = localStorage.getItem('baseUrl')
  if (savedBaseUrl) baseUrl.value = savedBaseUrl
  
  const savedApiKey = localStorage.getItem('apiKey')
  if (savedApiKey) apiKey.value = savedApiKey
  
  const savedQwenModel = localStorage.getItem('qwenModel')
  if (savedQwenModel) qwenModel.value = savedQwenModel
  
  const savedSam2Model = localStorage.getItem('sam2Model')
  if (savedSam2Model) sam2Model.value = savedSam2Model
})

watch(baseUrl, (newVal) => localStorage.setItem('baseUrl', newVal))
watch(apiKey, (newVal) => localStorage.setItem('apiKey', newVal))
watch(qwenModel, (newVal) => localStorage.setItem('qwenModel', newVal))
watch(sam2Model, (newVal) => localStorage.setItem('sam2Model', newVal))

const onVideoPlay = () => {
  isPlaying.value = true
  redrawCanvas()
}

const onVideoPause = () => {
  isPlaying.value = false
  redrawCanvas()
}

const onVideoSeeked = () => {
  isPlaying.value = false
  redrawCanvas()
}

// Watch showPoints change to redraw
watch(showPoints, () => {
  redrawCanvas()
})

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

// Coordinate Helper
const getVideoCoordinates = (event, video, rect) => {
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
    return null
  }
  
  const scaleX = video.videoWidth / displayedWidth
  const scaleY = video.videoHeight / displayedHeight
  
  return {
    x: clickX * scaleX,
    y: clickY * scaleY,
    displayX: clickX + offsetX,
    displayY: clickY + offsetY
  }
}

const handleCanvasMouseDown = (event) => {
  if (interactionMode.value === 'view') return
  
  isDrawing.value = true
  addPoint(event)
}

const handleCanvasMouseMove = (event) => {
  if (!isDrawing.value) return
  if (interactionMode.value === 'scribble') {
    // Throttle or just add? Let's add every move for smooth scribble
    addPoint(event)
  }
}

const handleCanvasMouseUp = () => {
  isDrawing.value = false
  // Analysis is now triggered manually via the Analyze button
}

const handleCanvasMouseLeave = () => {
  isDrawing.value = false
}

const addPoint = (event) => {
  const video = videoElement.value
  if (!video) return
  
  // Clear previous mask when adding new points (stale result)
  if (currentMaskUrl.value) {
    currentMaskUrl.value = null
  }

  const rect = video.getBoundingClientRect()
  const coords = getVideoCoordinates(event, video, rect)
  
  if (coords) {
    let label = 1
    if (interactionMode.value === 'negative') label = 0
    if (interactionMode.value === 'scribble') label = 1 // Default scribble as positive? Or make it configurable? Usually scribble is positive unless 'eraser'.
    
    // For scribble, we might want negative scribble too?
    // Let's assume scribble follows the mode? But we have 'scribble' mode.
    // If user selected 'scribble', let's assume positive. 
    // If user wants negative region, maybe they should use 'negative' points or we add 'negative scribble'.
    // For now, 'scribble' is positive.
    
    accumulatedPoints.value.push({
      x: coords.x,
      y: coords.y,
      label: label,
      displayX: coords.displayX, // Store for drawing on canvas (relative to video rect)
      displayY: coords.displayY
    })
    
    redrawCanvas()
  }
}

const undoLastPoint = () => {
  accumulatedPoints.value.pop()
  if (currentMaskUrl.value) {
      currentMaskUrl.value = null
  }
  redrawCanvas()
}

const clearAllPoints = () => {
  accumulatedPoints.value = []
  currentMaskUrl.value = null
  redrawCanvas()
  encyclopedia.value = null
  transcription.value = null
}

const redrawCanvas = () => {
  const canvas = maskCanvas.value
  const video = videoElement.value
  if (!canvas || !video) return
  
  const ctx = canvas.getContext('2d')
  canvas.width = video.clientWidth
  canvas.height = video.clientHeight
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // 1. Draw Mask if exists (Always draw mask regardless of playing state, unless user wants it hidden too? 
  // User said "Mask should be there". So we draw it.)
  if (currentMaskUrl.value) {
      const img = new Image()
      img.onload = () => {
          ctx.globalCompositeOperation = 'screen'
          ctx.globalAlpha = 0.6
          
          // Re-calculate video rect
          const vLeft = 0 // Relative to canvas
          const vTop = 0
          const vWidth = canvas.width
          const vHeight = canvas.height
          
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
          
          ctx.drawImage(img, offsetX, offsetY, displayedWidth, displayedHeight)
          
          ctx.globalCompositeOperation = 'source-over'
          ctx.globalAlpha = 1.0
          
          // Only draw points if not playing AND showPoints is enabled
          if (!isPlaying.value && showPoints.value) {
             drawPointsOverlay(ctx)
          }
      }
      img.src = currentMaskUrl.value
  } else {
      // If no mask, still draw points if conditions met
      if (!isPlaying.value && showPoints.value) {
          drawPointsOverlay(ctx)
      }
  }
}

const drawPointsOverlay = (ctx) => {
    // Draw points
    accumulatedPoints.value.forEach(p => {
        ctx.beginPath()
        ctx.arc(p.displayX, p.displayY, 4, 0, 2 * Math.PI)
        ctx.fillStyle = p.label === 1 ? 'green' : 'red'
        ctx.fill()
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 1
        ctx.stroke()
    })
}

const runAnalysis = async () => {
  if (!serverVideoPath.value) {
    ElMessage.error("Please upload a video first")
    return
  }
  
  if (accumulatedPoints.value.length === 0 && interactionMode.value !== 'view') {
    ElMessage.warning("Please add some points or scribbles first")
    return
  }
  
  loading.value = true
  
  try {
    // Send points to backend
    // Format: [[x, y], [x, y]], labels: [1, 0]
    const points = accumulatedPoints.value.map(p => [p.x, p.y])
    const labels = accumulatedPoints.value.map(p => p.label)
    
    const formData = new FormData()
    formData.append('video_path', serverVideoPath.value)
    formData.append('points_json', JSON.stringify(points))
    formData.append('labels_json', JSON.stringify(labels))
    formData.append('timestamp', videoElement.value.currentTime)
    formData.append('frame_width', videoElement.value.videoWidth)
    formData.append('frame_height', videoElement.value.videoHeight)
    
    // Add config
    formData.append('api_key', apiKey.value)
    formData.append('base_url', baseUrl.value)
    formData.append('qwen_model', qwenModel.value)
    formData.append('sam2_model', sam2Model.value)

    // Add Time Range
    formData.append('start_time', segmentRange.value[0])
    formData.append('end_time', segmentRange.value[1])
    
    const response = await fetch('http://localhost:8000/process', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const result = await response.json()
    console.log("Analysis Result:", result)
    
    encyclopedia.value = result.encyclopedia
    transcription.value = result.transcription
    
    if (result.video_url) {
        currentMaskUrl.value = null // Clear static mask
        // Replace video source with processed video
        // videoSource.value = `http://localhost:8000${result.video_url}`
        // Ideally we should overlay the result video or show side-by-side.
        // For prototype, let's open in new tab or replace.
        // Let's replace.
        videoSource.value = `http://localhost:8000${result.video_url}`
        ElMessage.success("Analysis Complete! Video updated with segmentation.")
    } else {
        // Only show static mask if NO video result (fallback)
        currentMaskUrl.value = result.mask
    }
    
    redrawCanvas() // Trigger redraw with new mask state
    
  } catch (err) {
    console.error(err)
    ElMessage.error("Analysis failed: " + (err.message || "Unknown error"))
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
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
  flex-direction: column;
}
.video-player {
  max-width: 100%;
  max-height: 600px;
  display: block;
}
.timeline-container {
  width: 100%;
  padding: 10px;
  background: #1a1a1a;
  margin-top: 10px;
}
.timeline-slider {
  width: 100%;
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
