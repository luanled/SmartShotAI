import io
import json
import math
import pathlib
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE = pathlib.Path(__file__).parent

MODELS = {
    "standard": BASE.parent / "tflite" / "original_20MB" / "emov2_scorer_full.onnx",
    "optimized": BASE.parent / "tflite" / "optimized_7MB" / "emov2_scorer_optimized_full.onnx",
    "siglip": BASE.parent / "tflite" / "siglip_356MB" / "siglip_aesthetic_preprocessed.onnx",
}

# Models that require a fixed input size and uint8 dtype
FIXED_INPUT_MODELS = {
    "siglip": (224, 224),
}

LOG_PATH = BASE / "logs" / "results.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

app = FastAPI(title="SmartShot Aesthetic Scorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {
    key: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for key, path in MODELS.items()
}
input_names = {key: s.get_inputs()[0].name for key, s in sessions.items()}


def to_percent(raw: float) -> tuple[int, int]:
    """Returns (curved_score, linear_score) both 0-100.

    Blend strategy: below 30 → no boost (bad images stay bad).
    30-60 → gradual ramp. Above 60 → full sqrt curve.
    """
    linear = ((raw - 1.0) / 9.0) * 100
    blend = max(0.0, min(1.0, (linear - 30) / 30))  # 0 at ≤30, 1 at ≥60
    curved = (1 - blend) * linear + blend * math.sqrt(linear + 20) * 10
    return round(max(0, min(100, curved))), round(max(0, min(100, linear)))


def log_result(model: str, source: str, score: int, score_linear: int, raw_score: float, server_ms: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source": source,
        "score_curved": score,
        "score_linear": score_linear,
        "raw_score": raw_score,
        "server_ms": round(server_ms, 1),
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
async def health():
    return {"status": "ok", "models": list(MODELS.keys())}


@app.post("/score")
async def score(
    file: UploadFile = File(...),
    model: str = Form("optimized"),
    source: str = Form("live"),
):
    if model not in sessions:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Use: {list(MODELS.keys())}")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()

    # Start timer after network upload is complete; covers decode + preprocess + inference
    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    if model in FIXED_INPUT_MODELS:
        w, h = FIXED_INPUT_MODELS[model]
        img = img.resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)[np.newaxis]   # [1, H, W, 3], uint8 0-255
    else:
        arr = np.array(img, dtype=np.float32)[np.newaxis]  # [1, H, W, 3], float32 0-255
    outputs = sessions[model].run(None, {input_names[model]: arr})
    server_ms = (time.perf_counter() - t0) * 1000

    raw_score = float(outputs[0].flat[0])
    percent, linear = to_percent(raw_score)

    log_result(model, source, percent, linear, round(raw_score, 4), server_ms)
    print(f"[{source:7}] model={model:9} score={percent:3d} (linear={linear:3d})  raw={raw_score:.4f}  server={server_ms:.1f}ms")

    return {"score": percent, "score_linear": linear, "raw_score": round(raw_score, 4), "server_ms": round(server_ms, 1)}
