"""
Real-ESRGAN 图像超分辨率 - Web 界面
运行: python app.py
"""
import os
import time
import cv2
import torch
import gradio as gr
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "通用 4x 放大 (RealESRGAN_x4plus)": {
        "path": "weights/RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4, "arch": "RRDBNet", "num_block": 23,
        "desc": "推荐通用模型，适合照片、风景等真实场景",
        "color": "#667eea",
    },
    "通用 2x 放大 (RealESRGAN_x2plus)": {
        "path": "weights/RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2, "arch": "RRDBNet", "num_block": 23,
        "desc": "2倍放大，速度更快，适合小幅增强",
        "color": "#f093fb",
    },
    "轻量通用 4x (realesr-general-x4v3)": {
        "path": "weights/realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "scale": 4, "arch": "SRVGG", "num_conv": 32, "wdn": True,
        "desc": "轻量模型，推理更快，支持降噪调节",
        "color": "#4facfe",
    },
    "动漫 4x 放大 (Anime 6B)": {
        "path": "weights/RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4, "arch": "RRDBNet", "num_block": 6,
        "desc": "专为动漫/二次元图片优化的模型",
        "color": "#fa709a",
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def build_upsampler(model_key):
    cfg = MODEL_CONFIGS[model_key]
    model_path = os.path.join(ROOT, cfg["path"])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if cfg["arch"] == "RRDBNet":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"])
    else:
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=cfg["num_conv"], upscale=cfg["scale"], act_type="prelu")

    if cfg.get("wdn"):
        wdn_path = model_path.replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
        wdn_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
        if not os.path.exists(wdn_path):
            load_file_from_url(wdn_url, model_dir=os.path.join(ROOT, "weights"))
        model_path = [model_path, wdn_path]

    return RealESRGANer(scale=cfg["scale"], model_path=model_path, model=model, half=device.type != "cpu", device=device)

DEFAULT_MODEL = list(MODEL_CONFIGS.keys())[0]
upsamplers = {DEFAULT_MODEL: build_upsampler(DEFAULT_MODEL)}

# ── CSS 样式 ─────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }

/* 页面背景 */
body {
    background: linear-gradient(135deg, #f5f3ff 0%, #fdf2f8 25%, #f0f9ff 50%, #ecfdf5 75%, #fef3c7 100%) !important;
    background-attachment: fixed !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: transparent !important;
}

/* ======== 头部 ======== */
.main-header {
    text-align: center;
    padding: 36px 24px 20px 24px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4338ca 60%, #6d28d9 100%);
    border-radius: 24px;
    margin-bottom: 20px;
    color: #fff;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(99, 102, 241, 0.35);
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(244, 114, 182, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(96, 165, 250, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 50% 80%, rgba(167, 139, 250, 0.12) 0%, transparent 50%);
    animation: headerShimmer 8s ease-in-out infinite;
}
@keyframes headerShimmer {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(2%, 1%); }
}
.main-header > * { position: relative; z-index: 1; }
.main-header h1 {
    font-size: 2.3em;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #e0e7ff 0%, #c4b5fd 40%, #f9a8d4 70%, #fde68a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.main-header .subtitle {
    font-size: 1em;
    color: #a5b4fc;
    font-weight: 300;
    letter-spacing: 1px;
}
.badge-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 16px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 22px;
    padding: 6px 16px;
    font-size: 0.82em;
    color: #e0e7ff;
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
}
.badge .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-green  { background: #4ade80; box-shadow: 0 0 8px #4ade80; animation: dotPulse 2s ease-in-out infinite; }
.dot-blue   { background: #60a5fa; box-shadow: 0 0 8px #60a5fa; animation: dotPulse 2s ease-in-out 0.3s infinite; }
.dot-amber  { background: #fbbf24; box-shadow: 0 0 8px #fbbf24; animation: dotPulse 2s ease-in-out 0.6s infinite; }
.dot-purple { background: #c084fc; box-shadow: 0 0 8px #c084fc; animation: dotPulse 2s ease-in-out 0.9s infinite; }
@keyframes dotPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ======== 卡片通用 ======== */
.card {
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06);
    padding: 24px;
    margin-bottom: 16px;
    border: 1px solid #f1f0ff;
    transition: box-shadow 0.3s;
}
.card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 12px 32px rgba(0,0,0,0.08);
}
.card-title {
    font-size: 1.08em;
    font-weight: 700;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding-bottom: 12px;
    border-bottom: 2px solid #f1f0ff;
}
.card-title .icon {
    width: 32px; height: 32px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1em;
}

/* 上传卡片 */
.card-upload .card-title { color: #6366f1; border-bottom-color: #e0e7ff; }
.card-upload .card-title .icon { background: #eef2ff; }
.card-upload {
    border-top: 4px solid #6366f1;
}

/* 结果卡片 */
.card-result .card-title { color: #a855f7; border-bottom-color: #f3e8ff; }
.card-result .card-title .icon { background: #faf5ff; }
.card-result {
    border-top: 4px solid #a855f7;
}

/* 控制卡片 */
.card-controls .card-title { color: #f59e0b; border-bottom-color: #fef3c7; }
.card-controls .card-title .icon { background: #fffbeb; }
.card-controls {
    border-top: 4px solid #f59e0b;
}

/* ======== 图片上传区美化 ======== */
.wrap.svelte-1pyp92z {
    border: 2px dashed #c7d2fe !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%) !important;
    transition: all 0.3s !important;
}
.wrap.svelte-1pyp92z:hover {
    border-color: #818cf8 !important;
    background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%) !important;
}

/* ======== 按钮 ======== */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #8b5cf6 50%, #d946ef 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    padding: 14px 36px !important;
    border-radius: 14px !important;
    letter-spacing: 1px !important;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s !important;
    position: relative;
    overflow: hidden;
}
.primary-btn::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.55) !important;
}
.primary-btn:hover::after { opacity: 1; }

/* 清空按钮 */
button.secondary {
    border: 2px solid #e0e7ff !important;
    color: #6366f1 !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    transition: all 0.3s !important;
}
button.secondary:hover {
    background: #eef2ff !important;
    border-color: #818cf8 !important;
}

/* ======== 状态栏 ======== */
.status-box textarea {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%) !important;
    border: 1px solid #a7f3d0 !important;
    border-radius: 14px !important;
    color: #065f46 !important;
    font-weight: 500 !important;
    font-size: 0.95em !important;
    text-align: center !important;
    padding: 12px !important;
}

/* ======== 底部 ======== */
.footer {
    text-align: center;
    padding: 28px 20px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 20px;
    margin-top: 8px;
    color: #a5b4fc;
    font-size: 0.85em;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ======== 下拉框美化 ======== */
select, .svelte-1hfxr1f {
    border-radius: 12px !important;
    border: 2px solid #e0e7ff !important;
    transition: all 0.3s !important;
}
select:focus, .svelte-1hfxr1f:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
}

/* ======== 滑块美化 ======== */
input[type="range"] {
    accent-color: #8b5cf6 !important;
}

/* ======== 复选框美化 ======== */
input[type="checkbox"] {
    accent-color: #d946ef !important;
}

/* ======== 标签文字颜色 ======== */
label span.svelte-1gfkn6j {
    color: #4338ca !important;
    font-weight: 600 !important;
}

/* ======== 信息提示文字 ======== */
div.svelte-19vraru {
    color: #8b5cf6 !important;
    font-size: 0.82em !important;
}

/* ======== 响应式 ======== */
@media (max-width: 768px) {
    .main-header h1 { font-size: 1.5em; }
}
"""

# ── 推理函数 ────────────────────────────────────────
def enhance(image, model_key, denoise_strength, face_enhance):
    if image is None:
        return None, "请先上传一张图片"

    start = time.time()

    if model_key not in upsamplers:
        upsamplers[model_key] = build_upsampler(model_key)
    upsampler = upsamplers[model_key]
    scale = MODEL_CONFIGS[model_key]["scale"]

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=scale, arch="clean", channel_multiplier=2, bg_upsampler=upsampler)
        _, _, output = face_enhancer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        output, _ = upsampler.enhance(img_bgr, outscale=scale)

    elapsed = time.time() - start
    result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    in_h, in_w = image.shape[:2]
    out_h, out_w = output.shape[:2]
    info = f"✅ 处理完成 | {in_w}×{in_h} → {out_w}×{out_h} | 耗时 {elapsed:.1f}s | 放大 {scale}x"

    return result, info


# ── 构建界面 ────────────────────────────────────────
with gr.Blocks(title="Real-ESRGAN 图像超分辨率") as demo:

    # 头部
    gr.HTML("""<div class="main-header">
        <h1>Real-ESRGAN 图像超分辨率</h1>
        <div class="subtitle">AI 驱动的智能图像放大与增强 &nbsp;·&nbsp; 4x / 2x 无损超分</div>
        <div class="badge-row">
            <span class="badge"><span class="dot dot-green"></span> """ + gpu_name + """</span>
            <span class="badge"><span class="dot dot-blue"></span> PyTorch """ + torch.__version__ + """</span>
            <span class="badge"><span class="dot dot-amber"></span> CUDA """ + str(torch.version.cuda) + """</span>
            <span class="badge"><span class="dot dot-purple"></span> Gradio 6</span>
        </div>
    </div>""")

    # 上传 & 结果区
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["card", "card-upload"]):
                gr.HTML('<div class="card-title"><span class="icon">📷</span> 上传图片</div>')
                input_img = gr.Image(label=None, type="numpy", height=380)

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["card", "card-result"]):
                gr.HTML('<div class="card-title"><span class="icon">✨</span> 超分结果</div>')
                output_img = gr.Image(label=None, type="numpy", height=380)

    # 状态栏
    status_text = gr.Textbox(
        label=None,
        value="🎨 就绪 — 上传图片后点击「开始超分」",
        interactive=False,
        container=True,
        elem_classes=["status-box"],
    )

    # 控制面板
    with gr.Group(elem_classes=["card", "card-controls"]):
        gr.HTML('<div class="card-title"><span class="icon">⚙️</span> 处理设置</div>')
        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=DEFAULT_MODEL,
                label="🤖 AI 模型",
                info=MODEL_CONFIGS[DEFAULT_MODEL]["desc"],
                scale=5,
            )
            denoise_slider = gr.Slider(
                0, 1, value=0.3, step=0.05,
                label="🎚️ 降噪强度",
                info="仅对「轻量通用」模型生效",
                scale=2,
            )
            face_check = gr.Checkbox(
                label="🧑 人脸增强",
                value=False,
                info="调用 GFPGAN 修复人脸",
                scale=1,
            )

        with gr.Row():
            run_btn = gr.Button("✨ 开始超分", variant="primary", size="lg", scale=2, elem_classes=["primary-btn"])
            clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg", scale=1)

    # 底部
    gr.HTML("""<div class="footer">
        ⚡ Real-ESRGAN &nbsp;—&nbsp; Practical Algorithms for General Image Restoration<br>
        <span style="font-size:0.85em;color:#818cf8;">支持 4x / 2x 放大 &nbsp;·&nbsp; 通用 / 动漫模型 &nbsp;·&nbsp; 人脸增强 &nbsp;·&nbsp; GPU 加速</span>
    </div>""")

    def update_model_desc(key):
        return gr.update(info=MODEL_CONFIGS[key]["desc"])

    model_choice.change(fn=update_model_desc, inputs=[model_choice], outputs=[model_choice])

    run_btn.click(
        fn=enhance,
        inputs=[input_img, model_choice, denoise_slider, face_check],
        outputs=[output_img, status_text],
    )

    clear_btn.click(
        fn=lambda: (None, None, "🎨 已清空 — 请上传新图片"),
        outputs=[input_img, output_img, status_text],
    )


if __name__ == "__main__":
    print("Real-ESRGAN Web UI 启动中...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)