# api.py
from __future__ import annotations

import io
import os
import hashlib
from pathlib import Path
from typing import Tuple

import torch
import timm
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Optional (recommended for deployment): download model from Hugging Face Hub
# pip install huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

import hashlib
from pathlib import Path

def sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "AI Image Detector"
STATIC_DIR = "static"

# If model.pt is not present, we will download it from HF_REPO_ID (if set).
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.pt")

# Example: HF_REPO_ID="AIzahS/ai-image-detector-model"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "model.pt")

# Patch voting: higher = more stable but slower
VOTES = int(os.environ.get("VOTES", "7"))

# Upload size guard
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(12 * 1024 * 1024)))  # 12MB


# -----------------------------
# Device
# -----------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Hash helper (to verify deployed weights)
# -----------------------------
def sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# -----------------------------
# Model file loader
# -----------------------------
def ensure_model_file() -> Path:
    """
    Ensure model file exists locally.
    - If model.pt exists: use it
    - Else if HF_REPO_ID is set and huggingface_hub is available: download it
    """
    model_path = Path(MODEL_FILENAME)
    if model_path.exists():
        return model_path

    if HF_REPO_ID and hf_hub_download is not None:
        downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        downloaded_path = Path(downloaded_path)
        downloaded_path.replace(model_path)  # move into working dir as MODEL_FILENAME
        return model_path

    raise FileNotFoundError(
        f"Missing {MODEL_FILENAME}. "
        f"Either place it beside api.py OR set HF_REPO_ID (and install huggingface_hub)."
    )


# -----------------------------
# Robust image loading
# -----------------------------
def load_image_safe(file_bytes: bytes) -> Image.Image:
    if len(file_bytes) > MAX_BYTES:
        raise ValueError("File too large.")

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)  # fix rotation
        img = img.convert("RGB")
        img.load()  # force decode now
        return img
    except UnidentifiedImageError as e:
        raise ValueError("Unsupported image format.") from e
    except OSError as e:
        raise ValueError("Corrupted or truncated image.") from e


# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = ensure_model_file()
MODEL_SHA = sha256_file(str(MODEL_PATH))

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
arch = ckpt.get("arch", "efficientnet_b0")
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = timm.create_model(arch, pretrained=False, num_classes=1)
model.load_state_dict(ckpt["model_state"])
model.eval().to(DEVICE)

print(f"[startup] device={DEVICE} arch={arch} model={MODEL_PATH} sha={MODEL_SHA} classes={class_to_idx}")


# -----------------------------
# Transforms (patch voting)
# -----------------------------
base_resize = transforms.Resize(512)

crop_tfm = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

center_tfm = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def predict_image(img: Image.Image, votes: int = VOTES) -> Tuple[str, float, float]:
    """
    Returns: (label, confidence, prob_ai_generated)
    Uses patch voting for more stable predictions.
    """
    resized = base_resize(img)

    probs_y1 = []
    with torch.no_grad():
        # Center crop baseline
        x0 = center_tfm(resized).unsqueeze(0).to(DEVICE)
        logit0 = model(x0).squeeze(1)[0]
        probs_y1.append(torch.sigmoid(logit0).item())

        # Random crop votes
        for _ in range(max(1, votes) - 1):
            x = crop_tfm(resized).unsqueeze(0).to(DEVICE)
            logit = model(x).squeeze(1)[0]
            probs_y1.append(torch.sigmoid(logit).item())

    prob_y1 = sum(probs_y1) / len(probs_y1)

    # Map y==1 to actual class name ("ai" or "real")
    class_for_one = idx_to_class[1]
    if class_for_one == "ai":
        prob_ai = prob_y1
    else:
        prob_ai = 1.0 - prob_y1

    label = "AI-generated" if prob_ai >= 0.5 else "Real"
    confidence = prob_ai if prob_ai >= 0.5 else (1.0 - prob_ai)
    return label, float(confidence), float(prob_ai)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title=APP_TITLE)

# Needed so Vercel frontend can call Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend if you want (optional)
if Path(STATIC_DIR).exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def home():
        index_path = Path(STATIC_DIR) / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"status": "ok", "message": "static/index.html not found"}
else:
    @app.get("/")
    def home():
        return {"status": "ok", "message": "API is running. Visit /docs."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "arch": arch,
        "classes": class_to_idx,
        "votes": VOTES,
        "model_file": str(MODEL_PATH),
        "model_sha": sha256_file(str(MODEL_PATH)),
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = load_image_safe(contents)
    except Exception:
        return JSONResponse(
            {"error": "Cannot analyze this image. Try a different PNG/JPG or a smaller file."},
            status_code=400,
        )

    label, confidence, prob_ai = predict_image(img, votes=VOTES)
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_ai_generated": round(prob_ai, 4),
        "votes": VOTES,
    }
