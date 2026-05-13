"""FastAPI backend for the Vue Real-ESRGAN Studio UI."""
import asyncio
import io
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import app as gradio_app


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

api = FastAPI(title="Real-ESRGAN Studio", version="0.1.0")


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


def image_dimensions(image):
    height, width = image.shape[:2]
    return {"width": width, "height": height}


def model_payload(key, cfg):
    paths = [gradio_app.WEIGHTS_DIR / cfg["filename"]]
    if cfg.get("denoise_filename"):
        paths.append(gradio_app.WEIGHTS_DIR / cfg["denoise_filename"])
    return {
        "key": key,
        "title": cfg["title"],
        "description": cfg["description"],
        "scale": cfg["scale"],
        "arch": cfg["arch"],
        "supports_denoise": bool(cfg.get("supports_denoise")),
        "ready": all(path.exists() for path in paths),
    }


def run_inference(
    image,
    model_key,
    outscale,
    denoise_strength,
    tile_size,
    face_enhance,
    use_fp32,
    alpha_upsampler,
):
    if model_key not in gradio_app.MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="未知模型: {}".format(model_key))

    started_at = time.time()
    cfg = gradio_app.MODEL_CONFIGS[model_key]
    outscale = float(outscale or cfg["scale"])
    input_size = image_dimensions(image)

    try:
        upsampler, cache_key = gradio_app.get_upsampler(model_key, denoise_strength, tile_size, use_fp32)
        cv_image = gradio_app.to_cv_image(image)

        if face_enhance:
            face_enhancer = gradio_app.get_face_enhancer(upsampler, cache_key, outscale)
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

        output_path = Path(gradio_app.save_result(output, model_key))
        output_size = {"width": output.shape[1], "height": output.shape[0]}
        elapsed = time.time() - started_at
        tile_text = "关闭" if gradio_app.normalize_tile(tile_size) == 0 else "{} px".format(
            gradio_app.normalize_tile(tile_size)
        )
        status = (
            "处理完成 | 模型: {} | {}x{} -> {}x{} | 输出倍率: {}x | 分块: {} | 耗时: {:.1f}s"
        ).format(
            cfg["title"],
            input_size["width"],
            input_size["height"],
            output_size["width"],
            output_size["height"],
            outscale,
            tile_text,
            elapsed,
        )

        relative = output_path.relative_to(ROOT)
        return {
            "id": output_path.stem,
            "url": "/{}?t={}".format(relative.as_posix(), int(output_path.stat().st_mtime)),
            "status": status,
            "elapsed": elapsed,
            "input": input_size,
            "output": output_size,
            "model": model_payload(model_key, cfg),
        }

    except HTTPException:
        raise
    except RuntimeError as error:
        if gradio_app.DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        message = str(error)
        if "out of memory" in message.lower():
            message = "显存不足。请把分块大小设为 256 或 128 后重试。"
        raise HTTPException(status_code=500, detail=message)
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="处理失败: {}".format(error))


@api.get("/api/health")
def health():
    return {
        "device": {
            "type": gradio_app.DEVICE.type,
            "label": gradio_app.device_label(),
        },
        "torch": torch.__version__,
        "cuda": gradio_app.cuda_label(),
    }


@api.get("/api/models")
def models():
    return [model_payload(key, cfg) for key, cfg in gradio_app.MODEL_CONFIGS.items()]


@api.post("/api/enhance")
async def enhance(
    image: UploadFile = File(...),
    model_key: str = Form(gradio_app.DEFAULT_MODEL),
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
        run_inference,
        decoded,
        model_key,
        outscale,
        denoise_strength,
        tile_size,
        face_enhance,
        use_fp32,
        alpha_upsampler,
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
        <title>Real-ESRGAN Studio</title>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 32px;">
          <h1>Vue 前端尚未构建</h1>
          <p>请在项目根目录运行 <code>npm install --prefix frontend</code> 和 <code>npm run build --prefix frontend</code>。</p>
        </body>
        """
