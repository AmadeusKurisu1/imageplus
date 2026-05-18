"""Shared core for Real-ESRGAN — no Gradio dependency."""
import re
import time
import traceback
from pathlib import Path

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.swinir_arch import SwinIR
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
    "realesr-animevideov3": {
        "title": "动漫视频 4x",
        "description": "超轻量动漫模型，专为动画视频优化，推理极快。",
        "filename": "realesr-animevideov3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4,
        "arch": "SRVGG",
        "num_conv": 16,
        "supports_denoise": False,
    },
    "RealESRNet_x4plus": {
        "title": "忠实还原 4x",
        "description": "MSE 损失训练，输出更忠实原图，不会生成额外纹理细节。",
        "filename": "RealESRNet_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "scale": 4,
        "arch": "RRDBNet",
        "num_block": 23,
        "supports_denoise": False,
    },
    "SwinIR-M": {
        "title": "SwinIR 中等 4x",
        "description": "Swin Transformer 架构，真实世界超分，细节还原力强。",
        "filename": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "scale": 4,
        "arch": "SwinIR",
        "swin_params": {"img_size": 64, "embed_dim": 180, "depths": [6, 6, 6, 6, 6, 6], "num_heads": [6, 6, 6, 6, 6, 6], "window_size": 8, "upsampler": "nearest+conv", "resi_connection": "1conv"},
        "supports_denoise": False,
    },
    "SwinIR-L": {
        "title": "SwinIR 大型 4x",
        "description": "Swin Transformer 大型变体，顶级的真实世界超分画质。",
        "filename": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "scale": 4,
        "arch": "SwinIR",
        "swin_params": {"img_size": 64, "embed_dim": 240, "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6], "num_heads": [8, 8, 8, 8, 8, 8, 8, 8, 8], "window_size": 8, "upsampler": "nearest+conv", "resi_connection": "3conv"},
        "supports_denoise": False,
    },
}

DEFAULT_MODEL = "RealESRGAN_x4plus"
MODEL_CHOICES = [(cfg["title"], key) for key, cfg in MODEL_CONFIGS.items()]
UPSAMPLERS = {}
FACE_ENHANCERS = {}


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
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=cfg["num_block"], num_grow_ch=32, scale=cfg["scale"],
        )
    if cfg["arch"] == "SRVGG":
        return SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=cfg["num_conv"], upscale=cfg["scale"], act_type="prelu",
        )
    if cfg["arch"] == "SwinIR":
        p = cfg["swin_params"]
        return SwinIR(
            upscale=cfg["scale"], img_size=p["img_size"],
            embed_dim=p["embed_dim"], depths=p["depths"],
            num_heads=p["num_heads"], window_size=p["window_size"],
            mlp_ratio=2, upsampler=p.get("upsampler", "pixelshuffle"),
            img_range=1.0, resi_connection=p.get("resi_connection", "1conv"),
        )
    raise ValueError("未知模型架构: {}".format(cfg["arch"]))


def ensure_weight(filename, url):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = WEIGHTS_DIR / filename
    if local_path.exists():
        return str(local_path)
    downloaded_path = load_file_from_url(
        url=url, model_dir=str(WEIGHTS_DIR), progress=True, file_name=filename,
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


class SwinIRUpsampler:
    """Lightweight tile-based inference wrapper for SwinIR, mimic RealESRGANer."""

    def __init__(self, model, scale, tile_size=0, tile_pad=10, pre_pad=10, device=None):
        self.model = model
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler="realesrgan"):
        h_input, w_input = img.shape[0:2]
        img = img.astype(np.float32)
        max_range = 255 if np.max(img) <= 256 else 65535
        img = img / max_range

        if len(img.shape) == 2:
            img_mode = "L"
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img_mode = "RGBA"
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = "RGB"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output = self._process(img)

        output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

        if img_mode == "RGBA":
            if alpha_upsampler == "realesrgan":
                output_alpha = self._process(alpha)
                output_alpha = output_alpha.squeeze(0).float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[0], :, :], (1, 2, 0))
                if output_alpha.shape[2] == 3:
                    output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_RGB2GRAY)
                output_alpha = output_alpha.reshape(output_alpha.shape[0], output_alpha.shape[1])
            else:
                h, w = img_mode == "RGBA" and alpha.shape[:2] or (h_input, w_input)
                output_alpha = cv2.resize(alpha if img_mode == "RGBA" else img,
                                          (w * self.scale, h * self.scale),
                                          interpolation=cv2.INTER_LINEAR)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
            output[:, :, 3] = output_alpha
        elif img_mode == "L":
            output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

        output = (output * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (int(w_input * outscale), int(h_input * outscale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return output, img_mode

    def _process(self, img_np):
        tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().unsqueeze(0)
        _, _, h_orig, w_orig = tensor.shape
        ws = 8  # SwinIR window size

        # Pad input to window_size multiple so tiles and features align
        mod_h = (ws - h_orig % ws) % ws
        mod_w = (ws - w_orig % ws) % ws
        if mod_h or mod_w:
            tensor = F.pad(tensor, (0, mod_w, 0, mod_h), "reflect")

        if self.tile_size > 0:
            tile_aligned = max(ws, self.tile_size - (self.tile_size % ws))
            self._effective_tile = tile_aligned
            output = self._tile_process(tensor)
        else:
            output = self.model(tensor.to(self.device))

        # Trim model's internal check_image_size padding then our mod padding
        out_h = (h_orig + mod_h) * self.scale
        out_w = (w_orig + mod_w) * self.scale
        output = output[:, :, :out_h, :out_w]
        if mod_h or mod_w:
            output = output[:, :, :h_orig * self.scale, :w_orig * self.scale]
        return output

    def _tile_process(self, tensor):
        _, _, h, w = tensor.shape
        ts = self._effective_tile
        out_h = h * self.scale
        out_w = w * self.scale
        output = torch.zeros(1, 3, out_h, out_w, device=self.device, dtype=tensor.dtype)
        tiles_x = math.ceil(w / ts)
        tiles_y = math.ceil(h / ts)

        for y in range(tiles_y):
            for x in range(tiles_x):
                sx = x * ts
                sy = y * ts
                ex = min(sx + ts, w)
                ey = min(sy + ts, h)
                sx_p = max(sx - self.tile_pad, 0)
                sy_p = max(sy - self.tile_pad, 0)
                ex_p = min(ex + self.tile_pad, w)
                ey_p = min(ey + self.tile_pad, h)

                tile = tensor[:, :, sy_p:ey_p, sx_p:ex_p].to(self.device)
                tile_out = self.model(tile)

                osx = sx * self.scale
                osy = sy * self.scale
                oex = ex * self.scale
                oey = ey * self.scale
                osx_t = (sx - sx_p) * self.scale
                oex_t = osx_t + (ex - sx) * self.scale
                osy_t = (sy - sy_p) * self.scale
                oey_t = osy_t + (ey - sy) * self.scale

                # Trim model's extra padding from tile output
                tile_h = (ey_p - sy_p) * self.scale
                tile_w = (ex_p - sx_p) * self.scale
                tile_out = tile_out[:, :, :tile_h, :tile_w]

                output[:, :, osy:oey, osx:oex] = tile_out[:, :, osy_t:oey_t, osx_t:oex_t].to(output.device)
        return output

def build_upsampler(model_key, denoise_strength, tile_size, use_fp32):
    cfg = MODEL_CONFIGS[model_key]
    arch = cfg["arch"]
    model_path = ensure_weight(cfg["filename"], cfg["url"])
    tile = normalize_tile(tile_size)

    if arch == "SwinIR":
        model = create_model(cfg)
        loadnet = torch.load(model_path, map_location="cpu")
        if "params_ema" in loadnet:
            model.load_state_dict(loadnet["params_ema"], strict=True)
        elif "params" in loadnet:
            model.load_state_dict(loadnet["params"], strict=True)
        else:
            model.load_state_dict(loadnet, strict=True)
        return SwinIRUpsampler(
            model=model, scale=cfg["scale"], tile_size=tile,
            tile_pad=10, pre_pad=0, device=DEVICE,
        )

    model = create_model(cfg)
    dni_weight = None
    if cfg.get("supports_denoise"):
        denoise = normalize_denoise(model_key, denoise_strength)
        if denoise is not None and denoise < 1:
            weak_model_path = ensure_weight(cfg["denoise_filename"], cfg["denoise_url"])
            model_path = [model_path, weak_model_path]
            dni_weight = [denoise, 1 - denoise]
    return RealESRGANer(
        scale=cfg["scale"], model_path=model_path, dni_weight=dni_weight,
        model=model, tile=tile, tile_pad=10, pre_pad=10,
        half=(DEVICE.type == "cuda" and not use_fp32), device=DEVICE,
    )


def get_upsampler(model_key, denoise_strength, tile_size, use_fp32):
    denoise_key = normalize_denoise(model_key, denoise_strength)
    cache_key = (model_key, denoise_key, normalize_tile(tile_size), bool(use_fp32), DEVICE.type)
    if cache_key not in UPSAMPLERS:
        UPSAMPLERS[cache_key] = build_upsampler(model_key, denoise_strength, tile_size, use_fp32)
    return UPSAMPLERS[cache_key], cache_key


def get_face_enhancer(upsampler, cache_key, outscale):
    face_key = (cache_key, round(float(outscale), 2))
    if face_key not in FACE_ENHANCERS:
        from gfpgan import GFPGANer
        FACE_ENHANCERS[face_key] = GFPGANer(
            model_path=GFPGAN_MODEL_URL, upscale=float(outscale),
            arch="clean", channel_multiplier=2, bg_upsampler=upsampler,
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
    return ("**{}**  \n{}\n网络: `{}` | 标称倍率: `{}x` | {}").format(
        cfg["title"], cfg["description"], cfg["arch"], cfg["scale"], denoise,
    )


def model_inventory_markdown():
    rows = []
    for cfg in MODEL_CONFIGS.values():
        paths = [WEIGHTS_DIR / cfg["filename"]]
        if cfg.get("denoise_filename"):
            paths.append(WEIGHTS_DIR / cfg["denoise_filename"])
        state = "已就绪" if all(path.exists() for path in paths) else "首次运行时下载"
        rows.append("- {}: {}".format(cfg["title"], state))
    return "\n".join(rows)


def format_success(model_key, in_w, in_h, out_w, out_h, elapsed, outscale, tile_size):
    cfg = MODEL_CONFIGS[model_key]
    tile_text = "关闭" if normalize_tile(tile_size) == 0 else "{} px".format(normalize_tile(tile_size))
    return ("处理完成 | 模型: {} | {}x{} -> {}x{} | 输出倍率: {}x | 分块: {} | 耗时: {:.1f}s").format(
        cfg["title"], in_w, in_h, out_w, out_h, outscale, tile_text, elapsed,
    )


def run_enhance(image, model_key, outscale, denoise_strength, tile_size,
                face_enhance, use_fp32, alpha_upsampler):
    if image is None:
        raise ValueError("请先上传一张图片。")
    if model_key not in MODEL_CONFIGS:
        raise ValueError("未知模型: {}".format(model_key))

    started_at = time.time()
    cfg = MODEL_CONFIGS[model_key]
    outscale = float(outscale or cfg["scale"])
    in_h, in_w = image.shape[:2]

    try:
        upsampler, cache_key = get_upsampler(model_key, denoise_strength, tile_size, use_fp32)
        cv_image = to_cv_image(image)

        if face_enhance:
            face_enhancer = get_face_enhancer(upsampler, cache_key, outscale)
            _, _, output = face_enhancer.enhance(
                cv_image, has_aligned=False, only_center_face=False, paste_back=True,
            )
        else:
            output, _ = upsampler.enhance(cv_image, outscale=outscale, alpha_upsampler=alpha_upsampler)

        output_path = save_result(output, model_key)
        elapsed = time.time() - started_at
        out_h, out_w = output.shape[:2]
        status = format_success(model_key, in_w, in_h, out_w, out_h, elapsed, outscale, tile_size)

        return {
            "output": output,
            "output_path": output_path,
            "elapsed": elapsed,
            "input_size": {"width": in_w, "height": in_h},
            "output_size": {"width": out_w, "height": out_h},
            "status": status,
            "model_key": model_key,
        }
    except RuntimeError as error:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        message = str(error)
        if "out of memory" in message.lower():
            message = "显存不足。请把分块大小设为 256 或 128 后重试。"
        elif "PytorchStreamReader" in message or "failed reading zip" in message.lower():
            message = "模型权重文件已损坏，请删除 weights 目录下的 .pth 文件后重新启动。"
        raise RuntimeError(message)
    except Exception as error:
        traceback.print_exc()
        message = str(error)
        if "PytorchStreamReader" in message or "failed reading zip" in message.lower():
            message = "模型权重文件已损坏，请删除 weights 目录下的 .pth 文件后重新启动。"
        raise RuntimeError("处理失败: {}".format(message))