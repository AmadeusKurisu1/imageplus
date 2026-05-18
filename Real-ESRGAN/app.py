"""
Real-ESRGAN image super-resolution Web UI.

Run from this directory:
    python app.py
"""
import time
import traceback

import gradio as gr
import torch

import core


ROOT = core.ROOT
WEIGHTS_DIR = core.WEIGHTS_DIR
RESULTS_DIR = core.RESULTS_DIR
MODEL_CONFIGS = core.MODEL_CONFIGS
DEFAULT_MODEL = core.DEFAULT_MODEL
MODEL_CHOICES = core.MODEL_CHOICES
DEVICE = core.DEVICE

UI_THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate")


# ---- Gradio-specific UI helpers ----

def update_model_controls(model_key):
    cfg = MODEL_CONFIGS[model_key]
    denoise_visible = bool(cfg.get("supports_denoise"))
    return (
        core.model_info_markdown(model_key),
        gr.update(value=cfg["scale"]),
        gr.update(visible=denoise_visible, interactive=denoise_visible),
    )


def clear_all():
    return None, None, None, "已清空。上传图片后即可开始超分。"


def enhance(
    image,
    model_key,
    outscale,
    denoise_strength,
    tile_size,
    face_enhance,
    use_fp32,
    alpha_upsampler,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        return None, None, "请先上传一张图片。"

    try:
        progress(0.05, desc="准备模型")
        progress(0.45, desc="执行超分")
        result = core.run_enhance(
            image, model_key, outscale, denoise_strength, tile_size,
            face_enhance, use_fp32, alpha_upsampler,
        )
        progress(0.9, desc="保存结果")

        output = result["output"]
        output_path = result["output_path"]
        status = result["status"]
        rgb = core.from_cv_image(output)
        progress(1.0, desc="完成")
        return rgb, output_path, status

    except RuntimeError as error:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        message = str(error)
        if "out of memory" in message.lower():
            message = "显存不足。请把分块大小设为 256 或 128 后重试。"
        return None, None, message
    except Exception as error:
        traceback.print_exc()
        return None, None, "处理失败: {}".format(error)


def example_rows():
    candidates = [
        ("inputs/0014.jpg", DEFAULT_MODEL),
        ("inputs/OST_009.png", DEFAULT_MODEL),
        ("inputs/children-alpha.png", "realesr-general-x4v3"),
        ("inputs/00003.png", "RealESRGAN_x4plus_anime_6B"),
    ]
    rows = []
    for rel_path, model_key in candidates:
        path = ROOT / rel_path
        if path.exists():
            cfg = MODEL_CONFIGS[model_key]
            rows.append([str(path), model_key, cfg["scale"], 0.5, 0, False, False, "realesrgan"])
    return rows


CUSTOM_CSS = """
:root,
.dark {
    --bg: #eef3f8;
    --bg-strong: #dfe8f2;
    --surface: #ffffff;
    --surface-soft: #f7fafc;
    --surface-strong: #edf4f8;
    --text: #142133;
    --muted: #617084;
    --muted-strong: #3d4a5c;
    --line: #d3dee9;
    --line-strong: #b8c6d4;
    --blue: #1f63e9;
    --blue-deep: #153d8a;
    --teal: #0f766e;
    --amber: #b7791f;
    --red: #b42318;
    --shadow: 0 18px 42px rgba(31, 55, 85, 0.10);
    --shadow-soft: 0 8px 24px rgba(31, 55, 85, 0.08);
    --radius: 8px;
    --body-background-fill: var(--bg);
    --body-text-color: var(--text);
    --block-background-fill: var(--surface);
    --block-border-color: var(--line);
    --input-background-fill: #ffffff;
    --input-border-color: var(--line);
    --button-primary-background-fill: var(--blue);
    --button-primary-text-color: #ffffff;
}

html,
body {
    background:
        linear-gradient(180deg, rgba(223, 232, 242, 0.92) 0%, rgba(238, 243, 248, 0.76) 36%, #f7fafc 100%) !important;
    color: var(--text) !important;
    color-scheme: light !important;
}

gradio-app,
main,
.main,
.app,
.wrap,
.gradio-container {
    background: transparent !important;
    color: var(--text) !important;
    color-scheme: light !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 1360px !important;
    padding: 24px !important;
    margin: 0 auto !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}

.gradio-container *,
.gradio-container *::before,
.gradio-container *::after {
    box-sizing: border-box;
}

.gradio-container > .gap {
    gap: 16px !important;
}

.topbar {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 18px;
    min-height: 104px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.66);
    border-radius: var(--radius);
    margin-bottom: 18px;
    color: #ffffff;
    background:
        linear-gradient(135deg, rgba(16, 35, 60, 0.96) 0%, rgba(31, 99, 233, 0.90) 72%, rgba(15, 118, 110, 0.86) 100%);
    box-shadow: var(--shadow);
    overflow: hidden;
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}

.brand-mark {
    display: grid;
    place-items: center;
    flex: 0 0 54px;
    width: 54px;
    height: 54px;
    border: 1px solid rgba(255, 255, 255, 0.34);
    border-radius: var(--radius);
    background: rgba(255, 255, 255, 0.12);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.22);
}

.brand-mark svg {
    width: 30px;
    height: 30px;
    color: #ffffff;
}

.brand-block h1 {
    margin: 0 0 8px;
    font-size: clamp(26px, 3vw, 40px);
    line-height: 1.05;
    letter-spacing: 0;
    color: #ffffff;
    overflow-wrap: anywhere;
}

.brand-block {
    min-width: 0;
}

.brand-block p {
    margin: 0;
    color: rgba(255, 255, 255, 0.78);
    font-size: 14px;
    line-height: 1.6;
}

.runtime-badges {
    display: grid;
    align-content: center;
    grid-template-columns: repeat(3, minmax(104px, 1fr));
    gap: 8px;
    min-width: 360px;
}

.runtime-badges span {
    border: 1px solid rgba(255, 255, 255, 0.24);
    background: rgba(255, 255, 255, 0.13);
    color: #ffffff;
    border-radius: var(--radius);
    padding: 10px 11px;
    font-size: 12px;
    line-height: 1.2;
    white-space: nowrap;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    backdrop-filter: blur(12px);
}

.workspace-grid,
.support-grid,
.control-grid,
.button-row,
.advanced-grid {
    gap: 16px !important;
}

.panel {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(211, 222, 233, 0.96);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow-soft);
    min-width: 0;
}

.media-panel {
    min-height: 448px;
}

.control-panel {
    margin-top: 0;
}

.gradio-container .block,
.gradio-container .wrap,
.gradio-container .row,
.gradio-container .column,
.gradio-container .form,
.gradio-container .prose,
.gradio-container .table-wrap,
.gradio-container .upload-container,
.gradio-container .image-container {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
"""


# ---- Build Gradio UI ----

def build_demo():
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=UI_THEME,
        title="基于CNN的图像超分系统",
    ) as demo:
        # ---- topbar ----
        gr.HTML("""
        <div class="topbar">
          <div class="brand-lockup">
            <div class="brand-mark">
              <svg viewBox="0 0 32 32" fill="none">
                <path d="M7 9.5C7 8.1 8.1 7 9.5 7h13C23.9 7 25 8.1 25 9.5v13c0 1.4-1.1 2.5-2.5 2.5h-13A2.5 2.5 0 0 1 7 22.5v-13Z" stroke="currentColor" stroke-width="2"/>
                <path d="M10.5 21.5 15 16l3.2 3.2 2.1-2.7 3.2 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20.7 11.8h.1" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
              </svg>
            </div>
            <div class="brand-block">
              <h1>基于CNN的图像超分系统</h1>
              <p>上传图片，一键提升分辨率和画质。</p>
            </div>
          </div>
          <div class="runtime-badges" id="runtime-badges">
            <span id="device-badge">{}</span>
            <span>PyTorch {}</span>
            <span>CUDA {}</span>
          </div>
        </div>
        """.format(core.device_label(), torch.__version__, core.cuda_label()))

        refresh_inventory = gr.Button("刷新模型清单", visible=False)

        with gr.Row(elem_classes=["workspace-grid"], equal_height=False):
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["panel", "media-panel"]):
                    input_img = gr.Image(
                        label="上传图片",
                        type="numpy",
                        elem_id="input-image",
                    )
                with gr.Row(elem_classes=["button-row"]):
                    run_btn = gr.Button("开始超分", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)
                with gr.Group(elem_classes=["panel"]):
                    output_img = gr.Image(
                        label="增强结果",
                        type="numpy",
                        elem_id="output-image",
                    )
                output_file = gr.File(label="结果文件", visible=False)
                status_text = gr.Textbox(
                    label="状态",
                    value="就绪。上传图片后设置参数并开始超分。",
                    interactive=False,
                    elem_id="status-text",
                )

            with gr.Column(scale=1, elem_classes=["control-panel"]):
                with gr.Group(elem_classes=["panel"]):
                    model_choice = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL,
                        label="模型",
                        interactive=True,
                    )
                    model_note = gr.Markdown(core.model_info_markdown(DEFAULT_MODEL))
                    output_scale = gr.Slider(
                        minimum=1, maximum=4,
                        value=MODEL_CONFIGS[DEFAULT_MODEL]["scale"],
                        step=0.5, label="输出倍率",
                    )
                    tile_size = gr.Dropdown(
                        choices=[("关闭", 0), ("128 px", 128), ("256 px", 256), ("512 px", 512)],
                        value=0, label="分块大小",
                    )
                    denoise_slider = gr.Slider(
                        minimum=0, maximum=1, value=0.5, step=0.05,
                        label="降噪强度", visible=False, interactive=False,
                    )
                    alpha_choice = gr.Dropdown(
                        choices=[("Real-ESRGAN", "realesrgan"), ("Bicubic", "bicubic")],
                        value="realesrgan", label="透明通道处理",
                    )
                    with gr.Row(elem_classes=["advanced-grid"]):
                        face_check = gr.Checkbox(label="人脸增强", value=False)
                        fp32_check = gr.Checkbox(label="FP32 精度", value=False)
                with gr.Group(elem_classes=["panel", "inventory"]):
                    gr.HTML("<strong>模型权重清单</strong>")
                    inventory_text = gr.Markdown(core.model_inventory_markdown(), elem_id="inventory")

        # ---- events ----
        model_choice.change(
            fn=update_model_controls,
            inputs=[model_choice],
            outputs=[model_note, output_scale, denoise_slider],
        )
        refresh_inventory.click(fn=core.model_inventory_markdown, outputs=[inventory_text])
        run_btn.click(
            fn=enhance,
            inputs=[
                input_img, model_choice, output_scale, denoise_slider,
                tile_size, face_check, fp32_check, alpha_choice,
            ],
            outputs=[output_img, output_file, status_text],
            cache_examples=False,
        )
        clear_btn.click(fn=clear_all, outputs=[input_img, output_img, output_file, status_text])

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=UI_THEME,
        css=CUSTOM_CSS,
        footer_links=[],
    )