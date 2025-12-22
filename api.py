from io import BytesIO

import torch
import timm
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Load model checkpoint
# -----------------------------
ckpt = torch.load("model.pt", map_location=DEVICE)
arch = ckpt.get("arch", "efficientnet_b0")
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = timm.create_model(arch, pretrained=False, num_classes=1)
model.load_state_dict(ckpt["model_state"])
model.eval().to(DEVICE)

# -----------------------------
# Patch-voting transforms
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

# -----------------------------
# App + frontend
# -----------------------------
app = FastAPI(title="AI Image Detector")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

# -----------------------------
# Inference (patch voting)
# -----------------------------
def predict_image(img: Image.Image, votes: int = 12):
    img = img.convert("RGB")
    resized = base_resize(img)

    probs_y1 = []
    with torch.no_grad():
        # Center crop vote
        x0 = center_tfm(resized).unsqueeze(0).to(DEVICE)
        logit0 = model(x0).squeeze(1)[0]
        probs_y1.append(torch.sigmoid(logit0).item())

        # Random crop votes
        for _ in range(votes - 1):
            x = crop_tfm(resized).unsqueeze(0).to(DEVICE)
            logit = model(x).squeeze(1)[0]
            probs_y1.append(torch.sigmoid(logit).item())

    prob_y1 = sum(probs_y1) / len(probs_y1)

    # prob_y1 = P(y==1). Need to map y==1 to actual class name.
    class_for_one = idx_to_class[1]  # either "ai" or "real"
    if class_for_one == "ai":
        prob_ai = prob_y1
    else:
        prob_ai = 1.0 - prob_y1

    label = "AI-generated" if prob_ai >= 0.5 else "Real"
    confidence = prob_ai if prob_ai >= 0.5 else (1.0 - prob_ai)

    return label, confidence, prob_ai

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(BytesIO(content))

    votes_used = 12
    label, confidence, prob_ai = predict_image(img, votes=votes_used)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob_ai_generated": round(prob_ai, 4),
        "votes": votes_used
    }
