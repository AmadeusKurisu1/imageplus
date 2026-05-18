"""FastAPI backend for the CNN image super-resolution Studio UI."""
import asyncio
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import core
import torch


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

api = FastAPI(title="基于CNN的图像超分系统", version="0.1.0")


def decode_upload(data):
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:
        raise HTTPException(status_code=400, detail="无法读取上传图片: {}".format(error))

    if image.mode in ("RGBA", "RGB", "L"):
        normalized = image
    elif "A" in image.mode:
        normalized = image.convert("RGBA")
    else:
        normalized = image.convert("RGB")
    return np.array(normalized)


def model_payload(key, cfg):
    paths = [core.WEIGHTS_DIR / cfg["filename"]]
    if cfg.get("denoise_filename"):
        paths.append(core.WEIGHTS_DIR / cfg["denoise_filename"])
    return {
        "key": key,
        "title": cfg["title"],
        "description": cfg["description"],
        "scale": cfg["scale"],
        "arch": cfg["arch"],
        "supports_denoise": bool(cfg.get("supports_denoise")),
        "ready": all(path.exists() for path in paths),
    }


def run_inference(image, model_key, outscale, denoise_strength, tile_size,
                  face_enhance, use_fp32, alpha_upsampler):
    try:
        result = core.run_enhance(
            image, model_key, outscale, denoise_strength, tile_size,
            face_enhance, use_fp32, alpha_upsampler,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    output_path = Path(result["output_path"])
    relative = output_path.relative_to(ROOT)
    return {
        "id": output_path.stem,
        "url": "/{}?t={}".format(relative.as_posix(), int(output_path.stat().st_mtime)),
        "status": result["status"],
        "elapsed": result["elapsed"],
        "input": result["input_size"],
        "output": result["output_size"],
        "model": model_payload(model_key, core.MODEL_CONFIGS[model_key]),
    }


@api.get("/api/health")
def health():
    return {
        "device": {
            "type": core.DEVICE.type,
            "label": core.device_label(),
        },
        "torch": torch.__version__,
        "cuda": core.cuda_label(),
    }


@api.get("/api/models")
def models():
    return [model_payload(key, cfg) for key, cfg in core.MODEL_CONFIGS.items()]


@api.post("/api/enhance")
async def enhance(
    image: UploadFile = File(...),
    model_key: str = Form(core.DEFAULT_MODEL),
    outscale: float = Form(4),
    denoise_strength: float = Form(0.5),
    tile_size: int = Form(0),
    face_enhance: bool = Form(False),
    use_fp32: bool = Form(False),
    alpha_upsampler: str = Form("realesrgan"),
):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="请先上传一张图片。")
    decoded = decode_upload(data)
    return await asyncio.to_thread(
        run_inference, decoded, model_key, outscale, denoise_strength,
        tile_size, face_enhance, use_fp32, alpha_upsampler,
    )


api.mount("/results", StaticFiles(directory=ROOT / "results"), name="results")

if FRONTEND_DIST.exists():
    api.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @api.get("/", response_class=HTMLResponse)
    def frontend_missing():
        return """
        <!doctype html>
        <meta charset="utf-8">
        <title>基于CNN的图像超分系统</title>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 32px;">
          <h1>Vue 前端尚未构建</h1>
          <p>请在项目根目录运行 <code>npm install --prefix frontend</code> 和 <code>npm run build --prefix frontend</code>。</p>
        </body>
        """