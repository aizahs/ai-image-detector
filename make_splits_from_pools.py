import random
import shutil
from pathlib import Path

# Folder that contains: gan_pool/, mj_pool/, real_pool/, sd_pool/
SRC = Path("genimage_subset")

# Where we will create the training structure your code expects
OUT = Path("data")

REAL_DIR = SRC / "real_pool"
AI_DIRS = [SRC / "sd_pool", SRC / "mj_pool", SRC / "gan_pool"]

# Keep it laptop-friendly (you can increase later)
MAX_REAL = 6000
MAX_AI = 6000

# Split ratios
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def copy_files(files, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in files:
        # Copy and keep filename only
        shutil.copy2(p, dst / p.name)


def split_list(items):
    n = len(items)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def main():
    random.seed(SEED)

    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {REAL_DIR}")
    for d in AI_DIRS:
        if not d.exists():
            raise FileNotFoundError(f"Missing folder: {d}")

    real_imgs = list_images(REAL_DIR)
    ai_imgs = []
    for d in AI_DIRS:
        ai_imgs.extend(list_images(d))

    if len(real_imgs) == 0:
        raise RuntimeError(f"No images found in {REAL_DIR}")
    if len(ai_imgs) == 0:
        raise RuntimeError(f"No images found in AI dirs: {AI_DIRS}")

    random.shuffle(real_imgs)
    random.shuffle(ai_imgs)

    real_imgs = real_imgs[:MAX_REAL]
    ai_imgs = ai_imgs[:MAX_AI]

    real_train, real_val, real_test = split_list(real_imgs)
    ai_train, ai_val, ai_test = split_list(ai_imgs)

    # Remove old data folder so you don't accidentally mix datasets
    if OUT.exists():
        shutil.rmtree(OUT)

    # Copy into the structure torchvision ImageFolder expects
    copy_files(real_train, OUT / "train" / "real")
    copy_files(ai_train,   OUT / "train" / "ai")

    copy_files(real_val, OUT / "val" / "real")
    copy_files(ai_val,   OUT / "val" / "ai")

    copy_files(real_test, OUT / "test" / "real")
    copy_files(ai_test,   OUT / "test" / "ai")

    print("✅ Done creating splits!")
    print(f"Train: real={len(real_train)} ai={len(ai_train)}")
    print(f"Val:   real={len(real_val)} ai={len(ai_val)}")
    print(f"Test:  real={len(real_test)} ai={len(ai_test)}")
    print("Output folder:", OUT.resolve())


if __name__ == "__main__":
    main()
