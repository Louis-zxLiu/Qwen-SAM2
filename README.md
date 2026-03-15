# Qwen-SAM2 多模态视频分析与编辑平台

这是一个集成了最先进 AI 模型的全栈多模态视频分析原型系统。它结合了 **SAM 2 (Segment Anything Model 2)** 的强大视频分割能力、**Whisper** 的精准语音转录能力以及 **Qwen VL (Qwen2-VL)** 的视觉理解能力，提供了一个端到端的智能视频交互、分析与编辑平台。

用户可以通过简单的点击或涂鸦与视频进行交互，系统将实时分割目标对象、提取相关语音内容，并生成关于目标的详细百科介绍，最终生成带有动态掩膜和原始音频的分割视频。

---

## 🌟 核心功能

1.  **全知视界 HUD (Omni-Context HUD)** 🔥 
    *   **屏幕感知**: 无需上传文件，一键截取当前屏幕，AI 实时分析。
    *   **哪里不懂点哪里**: 点击屏幕任意位置，SAM2 毫秒级分割目标，Qwen-VL 自动识别并解释。
    *   **无缝集成**: 独立的 Electron 客户端，通过快捷键 `Alt+X` 随时唤醒。

2.  **交互式视频分割 (SAM 2)**
    *   **多点提示**: 支持添加多个**正点**（目标区域）和**负点**（背景区域）来精确控制分割范围。
    *   **涂鸦输入 (Scribble)**: 支持通过画笔涂鸦来标记目标区域，提供更直观的交互方式。
    *   **全视频传播**: 基于单帧的提示，利用 SAM 2 的视频传播机制，自动分割整个视频序列中的目标。
    *   **实时反馈**: 分割结果以动态掩膜形式叠加在视频上，支持播放/暂停查看。

2.  **智能语音转录 (Whisper)**
    *   **自动提取**: 自动提取视频中目标出现时间段附近的语音内容。
    *   **幻觉过滤**: 内置后处理逻辑，有效过滤 Whisper 模型常见的幻觉输出（如重复的 "You" 等）。

3.  **多模态百科生成 (Qwen VL)**
    *   **视觉理解**: 利用 Qwen2-VL 模型对分割出的目标图像进行深度理解。
    *   **上下文融合**: 结合视觉信息和语音转录文本，生成关于目标对象的详细百科介绍。

4.  **完整的视听体验**
    *   **Web 兼容视频**: 生成的分割视频采用 H.264 编码，确保在所有现代浏览器中流畅播放。
    *   **音频保留**: 分割后的视频完整保留了原视频的音频轨道，提供沉浸式的观看体验。

5.  **人性化前端设计**
    *   **配置持久化**: API Key、模型选择等配置信息自动保存到本地，无需重复输入。
    *   **智能交互**: 掩膜在添加新点时自动清除，避免干扰；播放视频时自动隐藏静态掩膜，切换为动态视频展示。

---

## 🛠️ 技术栈架构

*   **后端 (Backend)**
    *   **框架**: FastAPI (高性能异步 Web 框架)
    *   **核心模型**:
        *   `facebook/sam2-hiera-tiny/small/large` (视频分割)
        *   `openai/whisper-tiny` (语音转录)
        *   `Qwen/Qwen2-VL-7B-Instruct` (多模态理解，通过 OpenAI 兼容接口调用)
    *   **视频处理**: `moviepy` (基于 FFmpeg 的视频编辑库)
    *   **依赖管理**: PyTorch, Transformers

*   **前端 (Frontend)**
    *   **框架**: Vue 3 + Vite
    *   **UI 组件库**: Element Plus
    *   **图标库**: Element Plus Icons

---

## 📋 环境要求

*   **操作系统**: Windows / Linux / macOS
*   **Python**: 3.10 或更高版本
*   **Node.js**: 16.0 或更高版本
*   **显卡 (GPU)**: 推荐使用 NVIDIA GPU 以获得流畅的推理体验（SAM 2 和 Whisper 需要一定的显存）。
*   **API Key**: 需要兼容 OpenAI 格式的 VLM 服务 API Key（推荐 [SiliconFlow](https://siliconflow.cn/) 或 [阿里云 Dashscope](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)）。

---

## 🚀 快速启动 (推荐)

项目提供了一键启动脚本，自动处理环境依赖检查和多服务启动。

1.  **克隆项目**
    ```bash
    git clone <repository_url>
    cd Qwen-SAM2
    ```

2.  **运行启动脚本**
    在项目根目录下运行：
    ```bash
    python start.py
    ```
    *脚本功能：*
    *   自动检查 `transformers` 和 `tokenizers` 版本兼容性。
    *   提供 **HF-Mirror (国内镜像)** 选项，加速模型下载。
    *   同时启动后端 API 服务 (Port 8000) 和前端开发服务器 (Port 5173)。

3.  **访问应用**
    打开浏览器访问：`http://localhost:5173`

---

## 📦 手动安装与运行

如果您更喜欢手动管理环境，请按照以下步骤操作：

### 1. 后端设置

```bash
cd backend

# 创建虚拟环境 (可选但推荐)
python -m venv venv
# Windows 激活
.\venv\Scripts\activate
# Linux/Mac 激活
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 强制重装特定版本依赖以解决兼容性问题 (重要!)
pip install "tokenizers>=0.22.0,<=0.23.0" --force-reinstall

# 启动后端服务
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

### 3. 全知视界 HUD 设置

```bash
cd electron-hud

# 安装依赖
npm install

# 启动 HUD 客户端
npm start
```
*   **快捷键**: 按下 `Alt+X` 截取当前屏幕并进行 AI 交互。
*   **交互**: 点击屏幕上的任意位置，SAM2 会自动分割目标，Qwen-VL 会解释该物体。
*   **退出**: 按 `Esc` 键隐藏 HUD。

---

## 📖 使用指南

1.  **配置 API (首次运行)**
    *   页面加载后，展开左上角的 **"API & Model Configuration"** 面板。
    *   **Base URL**: 输入 VLM 服务的 Base URL (例如 `https://api.siliconflow.cn/v1`)。
    *   **API Key**: 输入您的 API Key。
    *   **Model**: 选择或输入想要使用的 Qwen VL 模型名称。
    *   *注：配置会自动保存，下次无需重新输入。*

2.  **上传视频**
    *   点击上传区域或拖拽一个 MP4 视频文件。
    *   等待上传完成。

3.  **交互与提示**
    *   **View 模式**: 仅观看视频，不进行标记。
    *   **Point (+)**: 点击视频中感兴趣的目标物体（添加正点）。
    *   **Point (-)**: 点击不希望包含的区域（添加负点）。
    *   **Scribble**: 在目标物体上按住鼠标画线（涂鸦提示）。
    *   *支持组合使用多种提示方式。如果不满意，可以使用 "Undo" 撤销上一步，或 "Clear" 清空所有点。*

4.  **开始分析**
    *   点击工具栏右侧的 **"Analyze"** 按钮。
    *   系统将开始处理：SAM 2 分割视频 -> Whisper 转录音频 -> Qwen VL 生成百科。
    *   *处理时间取决于视频长度和 GPU 性能。*

5.  **查看结果**
    *   **视频**: 自动播放带有绿色掩膜的分割视频，并包含原始音频。
    *   **百科**: 右侧面板显示 Qwen VL 生成的目标百科介绍。
    *   **转录**: 显示提取到的语音文本。

---

## ❓ 常见问题 (Troubleshooting)

### Q1: 分析失败，提示 "Input boxes must be a nested list with 3 levels"
**A**: 这是旧版代码的一个已知问题，已在最新版修复。请确保您使用的是最新的 `backend/utils.py` 代码。

### Q2: 视频无法播放或黑屏
**A**: 浏览器对视频编码格式有严格要求。本项目已升级为使用 `moviepy` 生成 `H.264 (libx264)` 编码的视频，确保网页兼容性。如果仍有问题，请尝试清除浏览器缓存或更换 Chrome/Edge 浏览器。

### Q3: 报错 "ImportError: cannot import name 'Sam2Model' from 'transformers'"
**A**: 这通常是因为 `transformers` 版本过低。请运行 `pip install --upgrade transformers`。SAM 2 需要 `transformers>=4.45.0`。

### Q4: 显存不足 (OOM)
**A**: SAM 2 处理长视频或高分辨率视频时显存占用较大。
*   尝试在配置中选择更小的模型，如 `facebook/sam2-hiera-tiny`。
*   上传较短或分辨率较低的视频进行测试。

### Q5: 前端图标显示不正常
**A**: 请确保已安装图表库依赖。如果遇到依赖冲突，尝试运行 `npm install @element-plus/icons-vue --legacy-peer-deps`。

---

## 📁 项目结构

```
Qwen-SAM2/
├── backend/                 # 后端代码
│   ├── main.py              # FastAPI 主程序，API 路由
│   ├── utils.py             # 核心逻辑：SAM2 推理、视频处理、Whisper 调用
│   ├── requirements.txt     # Python 依赖列表
│   └── temp/                # 临时文件存储 (上传的视频、生成的视频)
├── frontend/                # 前端代码
│   ├── src/
│   │   ├── App.vue          # 主页面逻辑 (Vue 3)
│   │   └── main.js          # 入口文件
│   ├── package.json         # npm 依赖配置
│   └── vite.config.js       # Vite 配置
├── electron-hud/            # 全知视界 HUD 客户端
│   ├── main.js              # Electron 主进程
│   ├── renderer.js          # 渲染进程与交互逻辑
│   ├── index.html           # HUD 界面
│   └── package.json         # Electron 依赖
├── start.py                 # 一键启动脚本
└── README.md                # 项目说明书
```
