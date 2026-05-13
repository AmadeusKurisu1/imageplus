# ImagePlus Real-ESRGAN Studio

一个基于 Real-ESRGAN 的本地图像超分 Web UI。项目保留原始 Real-ESRGAN 推理能力，新增 Vue 前端和 FastAPI 后端，并提供一键启动脚本。

默认启动的是新版 Vue 界面；旧 Gradio 界面仍然保留，可以作为备用入口。

## 功能

- 本地图像超分：支持照片、动漫插画、轻量通用模型。
- 模型自动下载：首次使用模型时自动下载权重到 `Real-ESRGAN/weights/`。
- Vue 产品界面：工作台、模型管理、任务记录三页。
- 原图/结果预览：支持对比、单看原图、单看结果。
- 大图查看：点击图片后弹窗查看，支持按钮缩放和滚轮缩放。
- 任务记录：本次会话内保留最近结果，方便回看和下载。
- 一键启动：自动使用 `.venv`，自动构建前端，端口占用时自动换端口。
- 旧版 Gradio fallback：需要时可通过 `--gradio` 启动。

## 环境要求

- Python 3.10+
- Node.js 18+ 和 npm
- macOS / Windows / Linux

推理设备会自动选择：

- CUDA：NVIDIA GPU
- MPS：Apple Silicon GPU
- CPU：无可用 GPU 时 fallback

## 安装

建议在项目根目录创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Real-ESRGAN/requirements.txt
```

安装前端依赖：

```bash
npm install --prefix frontend
```

## 启动

macOS / Linux：

```bash
python 启动超分界面.py
```

Windows：

```bat
启动超分界面.bat
```

启动后访问：

```text
http://127.0.0.1:7860
```

脚本会自动检查 `frontend/dist` 是否需要重新构建。前端源码有变化时，会自动执行 `npm run build`。

## 常用启动参数

```bash
python 启动超分界面.py --no-browser
```

只启动服务，不自动打开浏览器。

```bash
python 启动超分界面.py --port 7870
```

指定首选端口。默认是 `7860`，被占用时会自动尝试后续端口。

```bash
python 启动超分界面.py --strict-port
```

端口被占用时直接失败，不自动换端口。

```bash
python 启动超分界面.py --rebuild-frontend
```

强制重新构建 Vue 前端。

```bash
python 启动超分界面.py --gradio
```

启动旧版 Gradio UI。

## 前端开发

前端目录是 `frontend/`，使用 Vue + Vite。

```bash
cd frontend
npm install
npm run dev
```

生产构建：

```bash
npm run build
```

构建产物位于 `frontend/dist/`，默认不提交到 Git。

## 后端接口

新版 UI 使用 `Real-ESRGAN/vue_server.py` 提供 FastAPI 接口：

- `GET /api/health`：运行环境和推理设备
- `GET /api/models`：模型列表和权重状态
- `POST /api/enhance`：上传图片并执行超分
- `/results/...`：结果图片静态访问

核心推理逻辑仍复用 `Real-ESRGAN/app.py` 中的模型配置、权重下载、upsampler 缓存和结果保存逻辑。

## 项目结构

```text
.
├── 启动超分界面.py          # macOS/Linux 一键启动入口
├── 启动超分界面.bat         # Windows 一键启动入口
├── frontend/                # Vue + Vite 前端
│   ├── src/App.vue
│   └── src/styles.css
└── Real-ESRGAN/
    ├── app.py               # Gradio UI 和共享推理逻辑
    ├── vue_server.py        # FastAPI 后端
    ├── requirements.txt
    ├── weights/             # 自动下载的模型权重，默认不提交
    └── results/             # 生成结果，默认不提交
```

## 结果和模型文件

- 模型权重保存到 `Real-ESRGAN/weights/`
- UI 结果保存到 `Real-ESRGAN/results/ui/`
- `.pth`、`.pt`、`node_modules/`、`dist/` 等大文件或构建产物默认被 `.gitignore` 忽略

## 常见问题

### 7860 端口被占用

默认脚本会自动尝试 `7861`、`7862` 等后续端口，并在终端打印实际访问地址。

如果要强制使用 7860：

```bash
python 启动超分界面.py --strict-port
```

### 首次启动很慢

首次运行可能需要：

- 安装 npm 依赖
- 构建 Vue 前端
- 下载 Real-ESRGAN 模型权重

这些文件会缓存在本地，后续启动会快很多。

### 显存不足

把“分块大小”改为 `256 px` 或 `128 px` 后重试。分块越小越省显存，但速度会变慢。

### Vue 页面提示未构建

手动构建前端：

```bash
npm install --prefix frontend
npm run build --prefix frontend
```

### 只想用旧版 Gradio

```bash
python 启动超分界面.py --gradio
```

## 上游项目

核心模型和推理实现来自 Real-ESRGAN。上游文档保留在：

- `Real-ESRGAN/README.md`
- `Real-ESRGAN/README_CN.md`
