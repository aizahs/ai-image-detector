from pathlib import Path
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
    return (preds == y).float().mean().item()

def main():
    ckpt = torch.load("model.pt", map_location=DEVICE)

    model = timm.create_model(
        ckpt.get("arch", "efficientnet_b0"),
        pretrained=False,
        num_classes=1
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    test_dir = Path("data/test")
    test_ds = datasets.ImageFolder(str(test_dir), transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).long()

            correct += (preds == y).sum().item()
            total += y.size(0)

    print("Test accuracy:", round(correct / total, 4))
    print("Classes:", test_ds.class_to_idx)

if __name__ == "__main__":
    main()
