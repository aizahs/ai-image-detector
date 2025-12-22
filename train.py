from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
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
    # Expecting: data/train/{ai,real} and data/val/{ai,real}
    train_dir = Path("data/train")
    val_dir = Path("data/val")

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Missing data/train or data/val. Run make_val_split.py first.")

    train_ds = datasets.ImageFolder(str(train_dir), transform=build_transforms(train=True))
    val_ds   = datasets.ImageFolder(str(val_dir), transform=build_transforms(train=False))

    print("Classes:", train_ds.class_to_idx)
    print("Train samples:", len(train_ds), "Val samples:", len(val_ds))
    print("Device:", DEVICE)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)


    # Binary classifier (single logit)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val_acc = 0.0
    epochs = 2  # MVP: keep it short. You can increase later.

    for epoch in range(1, epochs + 1):
       # ---- Train ----
        model.train()
        train_acc_sum = 0.0
        train_batches = 0

        import time
        t0 = time.time()

        for i, (x, y) in enumerate(train_loader):
            if i == 10:
               dt = time.time() - t0
               est_min = (dt / 10) * len(train_loader) / 60
               print(f"10 batches took {dt:.1f}s → approx epoch time {est_min:.1f} min")

            x = x.to(DEVICE)
            y = y.float().to(DEVICE)

            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc_sum += accuracy_from_logits(logits.detach(), y.detach().long())
            train_batches += 1

       
        train_acc = train_acc_sum / max(1, train_batches)
        # ---- Validate ----
        model.eval()
        val_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.float().to(DEVICE)

                logits = model(x).squeeze(1)
                val_acc_sum += accuracy_from_logits(logits, y.long())
                val_batches += 1

        val_acc = val_acc_sum / max(1, val_batches)

        print(f"Epoch {epoch}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "arch": "efficientnet_b0",
                },
                "model.pt",
            )
            print("✅ Saved model.pt (best so far)")

    print("Done. Best val_acc:", round(best_val_acc, 4))

if __name__ == "__main__":
    main()