import base64
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

BASE = Path(__file__).parent
CFG = json.loads((BASE / "config.json").read_text())
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Servis açılırken model BİR KEZ yüklenir
    sess = ort.InferenceSession(str(BASE / CFG["model_path"]),
                                providers=["CPUExecutionProvider"])
    state["sess"] = sess
    state["input"] = sess.get_inputs()[0].name
    yield
    state.clear()


app = FastAPI(title="Visual Inspection Service", lifespan=lifespan)


def hazirla(data: bytes) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    n = CFG["input_size"]
    im = cv2.resize(im, (n, n)).astype(np.float32) / 255.0
    return im.transpose(2, 0, 1)


def isi_haritasi_png(harita: np.ndarray, rgb: np.ndarray) -> str:
    h = (harita - harita.min()) / (harita.max() - harita.min() + 1e-8)
    renk = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    zemin = cv2.cvtColor((rgb.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    kaplama = cv2.addWeighted(zemin, 0.5, renk, 0.5, 0)
    ok, buf = cv2.imencode(".png", kaplama)
    return base64.b64encode(buf.tobytes()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "sess" in state,
            "threshold": CFG["threshold"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  threshold: float | None = None,
                  include_heatmap: bool = False):
    x = hazirla(await file.read())
    t0 = time.perf_counter()
    # Dışa aktarılan model batch=2 bekliyor; aynı görüntüyü iki kez veriyoruz
    out = state["sess"].run(None, {state["input"]: np.stack([x, x])})
    ms = (time.perf_counter() - t0) * 1000

    score = float(out[0][0])
    esik = CFG["threshold"] if threshold is None else threshold
    yanit = {
        "score": round(score, 4),
        "threshold": esik,
        "decision": "NOK" if score > esik else "OK",
        "inference_ms": round(ms, 1),
    }
    if include_heatmap:
        yanit["heatmap_png_base64"] = isi_haritasi_png(out[2][0, 0], x)
    return yanit