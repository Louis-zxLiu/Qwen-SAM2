# 多模态视频分析原型 (Multimodal Video Analysis Prototype)

这是一个全栈原型，展示了端到端的智能视频分析流程：
**本地视频上传 -> 点击目标对象 -> SAM2 实时分割 -> Whisper 语音转录 -> Qwen VL 百科生成 -> 前端可视化展示**

## 技术栈
- **后端**: FastAPI, Python 3.10+, PyTorch
  - 模型: `transformers.Sam2Model` (SAM2), `transformers.WhisperForConditionalGeneration` (Whisper), `dashscope` (Qwen VL)
  - 音频处理: `moviepy` (无需系统级 FFmpeg)
- **前端**: Vue 3, Vite, Element Plus

## 前置要求
- Python 3.10+
- Node.js & npm
- NVIDIA GPU (推荐用于加速推理，非必须)
- [Dashscope API Key](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key) 用于 Qwen VL (可在前端界面动态输入)

## 快速启动 (Windows)

1. **环境准备**: 确保已安装 Python 和 Node.js。
2. **启动应用**: 在项目根目录下运行：
   ```bash
   python start.py
   ```
   *该脚本将自动检测后端虚拟环境并同时启动前端与后端服务。*

## 手动安装与运行

### 1. 后端设置

```bash
cd backend
# 创建虚拟环境
python -m venv venv

# Windows 激活虚拟环境
.\venv\Scripts\activate
# Linux/Mac 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 前端设置

```bash
cd frontend
# 安装依赖
npm install
# 启动开发服务器
npm run dev
```

### 3. 使用说明

1. 打开浏览器访问前端地址 (通常为 `http://localhost:5173`)。
2. **API Key**: 在上传界面输入您的 Dashscope API Key（可选，不输入将显示模拟数据）。
3. **上传**: 选择并上传一个本地视频文件 (MP4)。
4. **交互**: 播放视频并在感兴趣的画面暂停，点击视频中的任意目标物体。
5. **结果**:
   - **SAM2** 将实时分割出目标轮廓并叠加显示。
   - **Whisper** 将提取并转录点击点附近的语音内容。
   - **Qwen VL** 将结合视觉与语音信息生成百科卡片。

## 注意事项

- **首次运行**: 后端会自动从 Hugging Face 下载 SAM2 和 Whisper 模型，这可能需要一些时间。
- **音频处理**: 现已切换至 `moviepy`，不再强制要求在系统中手动安装 FFmpeg，它会自动处理所需的二进制文件。
- **API 密钥**: 为了安全起见，API Key 仅在前端输入并随请求发送给后端，后端不会永久存储。
