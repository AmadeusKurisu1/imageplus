"""
Real-ESRGAN image super-resolution Web UI.

Run from this directory:
    python app.py
"""
import re
import time
import traceback
from html import escape
from pathlib import Path

import cv2
import gradio as gr
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results" / "ui"
GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"


MODEL_CONFIGS = {
    "RealESRGAN_x4plus": {
        "title": "通用照片 4x",
        "description": "适合照片、风景、建筑和多数真实场景。",
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "arch": "RRDBNet",
        "num_block": 23,
        "supports_denoise": False,
    },
    "RealESRGAN_x2plus": {
        "title": "通用照片 2x",
        "description": "适合小幅放大，速度和显存占用更友好。",
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "arch": "RRDBNet",
        "num_block": 23,
        "supports_denoise": False,
    },
    "realesr-general-x4v3": {
        "title": "轻量通用 4x",
        "description": "推理更快，支持弱降噪到强降噪的连续调节。",
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "denoise_filename": "realesr-general-wdn-x4v3.pth",
        "denoise_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "scale": 4,
        "arch": "SRVGG",
        "num_conv": 32,
        "supports_denoise": True,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "title": "动漫插画 4x",
        "description": "适合动漫、插画、线稿和二次元图片。",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "arch": "RRDBNet",
        "num_block": 6,
        "supports_denoise": False,
    },
}

DEFAULT_MODEL = "RealESRGAN_x4plus"
MODEL_CHOICES = [(cfg["title"], key) for key, cfg in MODEL_CONFIGS.items()]
UPSAMPLERS = {}
FACE_ENHANCERS = {}
UI_THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


def device_label():
    if DEVICE.type == "cuda":
        return torch.cuda.get_device_name(0)
    if DEVICE.type == "mps":
        return "Apple Silicon GPU"
    return "CPU"


def cuda_label():
    return torch.version.cuda if torch.version.cuda else "N/A"


def create_model(cfg):
    if cfg["arch"] == "RRDBNet":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=cfg["num_block"],
            num_grow_ch=32,
            scale=cfg["scale"],
        )

    return SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=cfg["num_conv"],
        upscale=cfg["scale"],
        act_type="prelu",
    )


def ensure_weight(filename, url):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = WEIGHTS_DIR / filename
    if local_path.exists():
        return str(local_path)

    downloaded_path = load_file_from_url(
        url=url,
        model_dir=str(WEIGHTS_DIR),
        progress=True,
        file_name=filename,
    )
    if downloaded_path and Path(downloaded_path).exists():
        return str(downloaded_path)
    if local_path.exists():
        return str(local_path)

    raise FileNotFoundError("模型权重下载失败: {}".format(filename))


def normalize_tile(tile_size):
    if tile_size is None:
        return 0
    return int(tile_size)


def normalize_denoise(model_key, denoise_strength):
    cfg = MODEL_CONFIGS[model_key]
    if not cfg.get("supports_denoise"):
        return None
    return round(max(0.0, min(1.0, float(denoise_strength))), 2)


def build_upsampler(model_key, denoise_strength, tile_size, use_fp32):
    cfg = MODEL_CONFIGS[model_key]
    model = create_model(cfg)
    model_path = ensure_weight(cfg["filename"], cfg["url"])
    dni_weight = None

    if cfg.get("supports_denoise"):
        denoise = normalize_denoise(model_key, denoise_strength)
        if denoise is not None and denoise < 1:
            weak_model_path = ensure_weight(cfg["denoise_filename"], cfg["denoise_url"])
            model_path = [model_path, weak_model_path]
            dni_weight = [denoise, 1 - denoise]

    return RealESRGANer(
        scale=cfg["scale"],
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=normalize_tile(tile_size),
        tile_pad=10,
        pre_pad=10,
        half=(DEVICE.type == "cuda" and not use_fp32),
        device=DEVICE,
    )


def get_upsampler(model_key, denoise_strength, tile_size, use_fp32):
    denoise_key = normalize_denoise(model_key, denoise_strength)
    cache_key = (
        model_key,
        denoise_key,
        normalize_tile(tile_size),
        bool(use_fp32),
        DEVICE.type,
    )
    if cache_key not in UPSAMPLERS:
        UPSAMPLERS[cache_key] = build_upsampler(model_key, denoise_strength, tile_size, use_fp32)
    return UPSAMPLERS[cache_key], cache_key


def get_face_enhancer(upsampler, cache_key, outscale):
    face_key = (cache_key, round(float(outscale), 2))
    if face_key not in FACE_ENHANCERS:
        from gfpgan import GFPGANer

        FACE_ENHANCERS[face_key] = GFPGANer(
            model_path=GFPGAN_MODEL_URL,
            upscale=float(outscale),
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
    return FACE_ENHANCERS[face_key]


def to_cv_image(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def from_cv_image(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def safe_slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value or "image"


def save_result(output, model_key):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / "{}-{}.png".format(stamp, safe_slug(model_key))
    if not cv2.imwrite(str(path), output):
        raise RuntimeError("结果图片保存失败")
    return str(path)


def model_info_markdown(model_key):
    cfg = MODEL_CONFIGS[model_key]
    denoise = "支持降噪调节" if cfg.get("supports_denoise") else "固定模型权重"
    return (
        "**{}**  \n"
        "{}  \n"
        "网络: `{}` | 标称倍率: `{}x` | {}"
    ).format(cfg["title"], cfg["description"], cfg["arch"], cfg["scale"], denoise)


def model_inventory_markdown():
    rows = []
    for cfg in MODEL_CONFIGS.values():
        paths = [WEIGHTS_DIR / cfg["filename"]]
        if cfg.get("denoise_filename"):
            paths.append(WEIGHTS_DIR / cfg["denoise_filename"])
        state = "已就绪" if all(path.exists() for path in paths) else "首次运行时下载"
        rows.append("- {}: {}".format(cfg["title"], state))
    return "\n".join(rows)


def update_model_controls(model_key):
    cfg = MODEL_CONFIGS[model_key]
    denoise_visible = bool(cfg.get("supports_denoise"))
    return (
        model_info_markdown(model_key),
        gr.update(value=cfg["scale"]),
        gr.update(visible=denoise_visible, interactive=denoise_visible),
    )


def clear_all():
    return None, None, None, "已清空。上传图片后即可开始超分。"


def format_success(model_key, in_w, in_h, out_w, out_h, elapsed, outscale, tile_size):
    cfg = MODEL_CONFIGS[model_key]
    tile_text = "关闭" if normalize_tile(tile_size) == 0 else "{} px".format(normalize_tile(tile_size))
    return (
        "处理完成 | 模型: {} | {}x{} -> {}x{} | 输出倍率: {}x | 分块: {} | 耗时: {:.1f}s"
    ).format(cfg["title"], in_w, in_h, out_w, out_h, outscale, tile_text, elapsed)


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

    started_at = time.time()
    cfg = MODEL_CONFIGS[model_key]
    outscale = float(outscale or cfg["scale"])
    in_h, in_w = image.shape[:2]

    try:
        progress(0.05, desc="准备模型")
        upsampler, cache_key = get_upsampler(model_key, denoise_strength, tile_size, use_fp32)
        cv_image = to_cv_image(image)

        progress(0.45, desc="执行超分")
        if face_enhance:
            face_enhancer = get_face_enhancer(upsampler, cache_key, outscale)
            _, _, output = face_enhancer.enhance(
                cv_image,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
        else:
            output, _ = upsampler.enhance(
                cv_image,
                outscale=outscale,
                alpha_upsampler=alpha_upsampler,
            )

        progress(0.9, desc="保存结果")
        output_path = save_result(output, model_key)
        result = from_cv_image(output)
        out_h, out_w = output.shape[:2]
        status = format_success(model_key, in_w, in_h, out_w, out_h, time.time() - started_at, outscale, tile_size)
        progress(1.0, desc="完成")
        return result, output_path, status

    except RuntimeError as error:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        message = str(error)
        if "out of memory" in message.lower():
            message = "显存不足。请把分块大小设为 256 或 128 后重试。"
        else:
            message = "处理失败: {}".format(message)
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

.gradio-container .image-container,
.gradio-container .upload-container {
    border-radius: var(--radius) !important;
}

.image-drop .image-container,
.image-drop .upload-container {
    background:
        linear-gradient(45deg, #f4f7fb 25%, transparent 25%),
        linear-gradient(-45deg, #f4f7fb 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #f4f7fb 75%),
        linear-gradient(-45deg, transparent 75%, #f4f7fb 75%),
        #ffffff !important;
    background-size: 22px 22px !important;
    background-position: 0 0, 0 11px, 11px -11px, -11px 0 !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(31, 99, 233, 0.12) !important;
}

.panel-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
}

.panel-heading-title {
    display: flex;
    align-items: center;
    gap: 9px;
    min-width: 0;
}

.step-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius);
    background: var(--surface-strong);
    color: var(--blue-deep);
    font-size: 12px;
    font-weight: 800;
    line-height: 1;
}

.panel-heading h2 {
    margin: 0;
    font-size: 15px;
    font-weight: 800;
    line-height: 1.25;
    color: var(--text);
}

.panel-heading span {
    color: var(--muted);
    font-size: 12px;
    white-space: nowrap;
}

.model-note {
    background: linear-gradient(180deg, #f8fbfd 0%, #f1f6fa 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 13px 14px;
    min-height: 78px;
}

.model-note p {
    margin: 0;
    color: var(--muted-strong);
    font-size: 13px;
    line-height: 1.65;
}

.model-note strong {
    color: var(--text);
}

.model-note code {
    background: #e8eef5 !important;
    border: 1px solid var(--line) !important;
    border-radius: 5px !important;
    color: var(--blue-deep) !important;
    padding: 2px 5px !important;
}

.status-row {
    margin: 2px 0 !important;
}

.status-row textarea {
    min-height: 46px !important;
    border-radius: var(--radius) !important;
    border: 1px solid #a8d5c9 !important;
    background: linear-gradient(180deg, #f4fbf8 0%, #edf8f4 100%) !important;
    color: #0f513f !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

.inventory {
    color: var(--muted);
    font-size: 13px;
}

.inventory ul {
    margin-top: 8px;
    padding-left: 18px;
}

.inventory li {
    margin: 6px 0;
}

.download-action,
.download-action button,
#run-btn,
#run-btn button,
#clear-btn,
#clear-btn button,
#refresh-btn,
#refresh-btn button {
    border-radius: var(--radius) !important;
    font-weight: 800 !important;
}

#run-btn,
#run-btn button {
    background: linear-gradient(135deg, var(--blue) 0%, #174fbd 100%) !important;
    color: #fff !important;
    border: 1px solid #174fbd !important;
    box-shadow: 0 12px 24px rgba(31, 99, 233, 0.22) !important;
}

#run-btn button:hover {
    filter: brightness(1.04);
}

#clear-btn,
#clear-btn button,
#refresh-btn,
#refresh-btn button,
.download-action,
.download-action button {
    background: #f7fafc !important;
    border-color: var(--line-strong) !important;
    color: var(--text) !important;
}

.download-action button:disabled,
#clear-btn button:disabled,
#refresh-btn button:disabled {
    background: #e9eef5 !important;
    border-color: #cfd9e5 !important;
    color: var(--muted-strong) !important;
    opacity: 1 !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container .block label span,
.gradio-container .block span[class*="label"],
.gradio-container [data-testid="block-label"],
.gradio-container .block-title {
    color: var(--text) !important;
    opacity: 1 !important;
    font-size: 12px !important;
    font-weight: 750 !important;
}

.gradio-container .hide-label > label,
.gradio-container .hide-label .label-wrap {
    display: none !important;
}

.gradio-container .accordion {
    border-color: var(--line) !important;
    border-radius: var(--radius) !important;
    background: var(--surface-soft) !important;
}

.gradio-container table {
    font-size: 12px !important;
}

.gradio-container table,
.gradio-container thead,
.gradio-container tbody,
.gradio-container tr,
.gradio-container th,
.gradio-container td {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

.gradio-container th {
    color: var(--muted-strong) !important;
    font-weight: 800 !important;
}

.gradio-container tbody tr:hover td {
    background: #f5f9fc !important;
}

input[type="range"] {
    accent-color: var(--blue);
}

@media (max-width: 980px) {
    .topbar {
        flex-direction: column;
        min-height: 0;
    }

    .runtime-badges {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        min-width: 0;
    }
}

@media (max-width: 820px) {
    .gradio-container {
        padding: 14px !important;
    }

    .topbar {
        padding: 14px;
    }

    .brand-lockup {
        align-items: flex-start;
        width: 100%;
    }

    .brand-mark {
        width: 44px;
        height: 44px;
        flex-basis: 44px;
    }

    .brand-mark svg {
        width: 25px;
        height: 25px;
    }

    .brand-block h1 {
        font-size: 24px;
        line-height: 1.12;
    }

    .runtime-badges {
        grid-template-columns: 1fr;
    }

    .panel {
        padding: 13px;
    }

    .media-panel {
        min-height: 0;
    }

    .workspace-grid,
    .control-grid,
    .advanced-grid,
    .button-row,
    .support-grid {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }

    .workspace-grid > *,
    .control-grid > *,
    .advanced-grid > *,
    .button-row > *,
    .support-grid > * {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        flex: 1 1 auto !important;
    }

    .image-drop {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
}
"""


HEADER_HTML = """
<div class="topbar">
    <div class="brand-lockup">
        <div class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none" role="img">
                <path d="M7 9.5C7 8.1 8.1 7 9.5 7h13C23.9 7 25 8.1 25 9.5v13c0 1.4-1.1 2.5-2.5 2.5h-13A2.5 2.5 0 0 1 7 22.5v-13Z" stroke="currentColor" stroke-width="2"/>
                <path d="M10.5 21.5 15 16l3.2 3.2 2.1-2.7 3.2 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20.7 11.8h.1" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="brand-block">
            <h1>Real-ESRGAN 图像超分</h1>
            <p>本地推理工作台 · 高清增强 · 结果导出</p>
        </div>
    </div>
    <div class="runtime-badges">
        <span>Device: {device}</span>
        <span>PyTorch: {torch_version}</span>
        <span>CUDA: {cuda}</span>
    </div>
</div>
""".format(
    device=escape(device_label()),
    torch_version=escape(torch.__version__),
    cuda=escape(cuda_label()),
)


with gr.Blocks(title="Real-ESRGAN 图像超分") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=True, elem_classes=["workspace-grid"]):
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel", "media-panel"]):
                gr.HTML(
                    '<div class="panel-heading"><div class="panel-heading-title">'
                    '<strong class="step-index">01</strong><h2>输入源</h2></div>'
                    '<span>PNG / JPG / WebP</span></div>'
                )
                input_img = gr.Image(
                    label="输入图片",
                    show_label=False,
                    type="numpy",
                    height=360,
                    buttons=["fullscreen"],
                    elem_classes=["image-drop"],
                )

        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel", "media-panel"]):
                gr.HTML(
                    '<div class="panel-heading"><div class="panel-heading-title">'
                    '<strong class="step-index">02</strong><h2>增强结果</h2></div>'
                    '<span>预览 / 下载</span></div>'
                )
                output_img = gr.Image(
                    label="超分结果",
                    show_label=False,
                    type="numpy",
                    height=360,
                    interactive=False,
                    buttons=["download", "fullscreen"],
                    elem_classes=["image-drop"],
                )
                output_file = gr.DownloadButton(
                    "下载结果",
                    value=None,
                    variant="secondary",
                    size="md",
                    elem_classes=["download-action"],
                )

    status_text = gr.Textbox(
        label=None,
        show_label=False,
        value="就绪。上传图片后点击开始超分。",
        interactive=False,
        elem_classes=["status-row"],
    )

    with gr.Group(elem_classes=["panel", "control-panel"]):
        gr.HTML(
            '<div class="panel-heading"><div class="panel-heading-title">'
            '<strong class="step-index">03</strong><h2>处理参数</h2></div>'
            '<span>模型缓存 / 输出倍率 / 显存控制</span></div>'
        )
        with gr.Row(elem_classes=["control-grid"]):
            model_choice = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="模型",
                scale=3,
            )
            output_scale = gr.Slider(
                minimum=1,
                maximum=4,
                value=MODEL_CONFIGS[DEFAULT_MODEL]["scale"],
                step=0.5,
                label="输出倍率",
                scale=2,
            )
            tile_size = gr.Dropdown(
                choices=[("关闭", 0), ("128 px", 128), ("256 px", 256), ("512 px", 512)],
                value=0,
                label="分块大小",
                scale=2,
            )

        model_note = gr.Markdown(
            value=model_info_markdown(DEFAULT_MODEL),
            elem_classes=["model-note"],
        )

        with gr.Accordion("高级参数", open=False):
            with gr.Row(elem_classes=["advanced-grid"]):
                denoise_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.05,
                    label="降噪强度",
                    visible=False,
                    interactive=False,
                )
                alpha_choice = gr.Dropdown(
                    choices=[("Real-ESRGAN", "realesrgan"), ("Bicubic", "bicubic")],
                    value="realesrgan",
                    label="透明通道处理",
                )
                face_check = gr.Checkbox(label="人脸增强", value=False)
                fp32_check = gr.Checkbox(label="FP32 精度", value=False)

        with gr.Row(elem_classes=["button-row"]):
            run_btn = gr.Button("开始超分", variant="primary", size="lg", scale=3, elem_id="run-btn")
            clear_btn = gr.Button("清空", variant="secondary", size="lg", scale=1, elem_id="clear-btn")

    with gr.Row(elem_classes=["support-grid"]):
        with gr.Column(scale=2):
            examples = example_rows()
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[
                        input_img,
                        model_choice,
                        output_scale,
                        denoise_slider,
                        tile_size,
                        face_check,
                        fp32_check,
                        alpha_choice,
                    ],
                    outputs=[output_img, output_file, status_text],
                    fn=enhance,
                    cache_examples=False,
                )
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel", "inventory"]):
                gr.HTML(
                    '<div class="panel-heading"><div class="panel-heading-title">'
                    '<strong class="step-index">04</strong><h2>模型权重</h2></div>'
                    '<span>本地状态</span></div>'
                )
                inventory_text = gr.Markdown(value=model_inventory_markdown())
                refresh_inventory = gr.Button("刷新状态", size="sm", elem_id="refresh-btn")

    model_choice.change(
        fn=update_model_controls,
        inputs=[model_choice],
        outputs=[model_note, output_scale, denoise_slider],
    )
    refresh_inventory.click(fn=model_inventory_markdown, outputs=[inventory_text])
    run_btn.click(
        fn=enhance,
        inputs=[
            input_img,
            model_choice,
            output_scale,
            denoise_slider,
            tile_size,
            face_check,
            fp32_check,
            alpha_choice,
        ],
        outputs=[output_img, output_file, status_text],
    )
    clear_btn.click(fn=clear_all, outputs=[input_img, output_img, output_file, status_text])

demo.queue(max_size=8)


if __name__ == "__main__":
    print("Real-ESRGAN Web UI is starting...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=UI_THEME,
        css=CUSTOM_CSS,
        footer_links=[],
    )
