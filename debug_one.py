from PIL import Image
import torch
import timm
from torchvision import transforms

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("model.pt", map_location=DEVICE)
arch = ckpt.get("arch", "efficientnet_b0")
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("class_to_idx:", class_to_idx)
print("idx_to_class:", idx_to_class)

model = timm.create_model(arch, pretrained=False, num_classes=1)
model.load_state_dict(ckpt["model_state"])
model.eval().to(DEVICE)

base_resize = transforms.Resize(512)
center_tfm = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

img_path = "data/test/real/00016.JPEG"
img = Image.open(img_path).convert("RGB")
img = base_resize(img)

x = center_tfm(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logit = model(x).squeeze()
    p1 = torch.sigmoid(logit).item()   # probability of "class index 1"

print("p(class_index_1) =", round(p1, 4), " -> class 1 =", idx_to_class[1])

# Convert to prob_ai in a way that cannot be wrong:
if idx_to_class[1] == "ai":
    prob_ai = p1
else:
    prob_ai = 1 - p1

print("prob_ai =", round(prob_ai, 4))
print("prediction =", "AI" if prob_ai >= 0.5 else "REAL")
