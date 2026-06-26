import random
import shutil
from pathlib import Path

SRC = Path("genimage_subset")

OUT = Path("data")

REAL_DIR = SRC / "real_pool"
AI_DIRS = [SRC / "sd_pool", SRC / "mj_pool", SRC / "gan_pool"]

# MJ and GAN pools are small (500 each) — use all of them
# SD pool is large (5000) — use more to increase total data, but cap to avoid domination
MAX_MJ  = 500
MAX_GAN = 500
MAX_SD  = 2000
MAX_REAL = MAX_MJ + MAX_GAN + MAX_SD  # 3000 — matches total AI for balanced classes

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
        # Prefix with source pool name to avoid collisions across pools
        fname = f"{p.parent.name}_{p.name}"
        shutil.copy2(p, dst / fname)


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
    if len(real_imgs) == 0:
        raise RuntimeError(f"No images found in {REAL_DIR}")

    caps = {"sd_pool": MAX_SD, "mj_pool": MAX_MJ, "gan_pool": MAX_GAN}

    ai_imgs = []
    for d in AI_DIRS:
        pool = list_images(d)
        if len(pool) == 0:
            raise RuntimeError(f"No images found in AI dir: {d}")
        cap = caps.get(d.name, MAX_MJ)
        random.shuffle(pool)
        ai_imgs.extend(pool[:cap])
        print(f"  {d.name}: using {min(len(pool), cap)} of {len(pool)} images")

    random.shuffle(real_imgs)
    random.shuffle(ai_imgs)

    real_imgs = real_imgs[:MAX_REAL]

    real_train, real_val, real_test = split_list(real_imgs)
    ai_train, ai_val, ai_test = split_list(ai_imgs)

    if OUT.exists():
        shutil.rmtree(OUT)

    copy_files(real_train, OUT / "train" / "real")
    copy_files(ai_train,   OUT / "train" / "ai")

    copy_files(real_val, OUT / "val" / "real")
    copy_files(ai_val,   OUT / "val" / "ai")

    copy_files(real_test, OUT / "test" / "real")
    copy_files(ai_test,   OUT / "test" / "ai")

    print(" Done creating splits!")
    print(f"Train: real={len(real_train)} ai={len(ai_train)}")
    print(f"Val:   real={len(real_val)} ai={len(ai_val)}")
    print(f"Test:  real={len(real_test)} ai={len(ai_test)}")
    print("Output folder:", OUT.resolve())


if __name__ == "__main__":
    main()
