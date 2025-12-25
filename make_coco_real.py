from torchvision.datasets import CocoDetection
from pathlib import Path
import random
import shutil

OUT = Path("data/train/real")
OUT.mkdir(parents=True, exist_ok=True)

# This automatically downloads & caches correctly
dataset = CocoDetection(
    root="coco_images",
    annFile=None,
    download=True
)

print("Total COCO images:", len(dataset))

# Sample a reasonable amount
N = 6000
idxs = random.sample(range(len(dataset)), N)

for i, idx in enumerate(idxs):
    img, _ = dataset[idx]
    img.save(OUT / f"coco_{i}.jpg")

print("Saved", N, "real COCO images")
