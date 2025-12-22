import random
import shutil
from pathlib import Path

# Settings
VAL_RATIO = 0.15
SEED = 42

# Paths (relative to project root)
SRC_ARCHIVE = Path("archive")   # where CIFAKE is right now
DST_DATA = Path("data")         # where we want train/val/test

random.seed(SEED)

def ensure_dir(p: Path):              # ensure directory exists
    p.mkdir(parents=True, exist_ok=True) 

def copy_all(src_dir: Path, dst_dir: Path):   # copy all images into dst_dir
    ensure_dir(dst_dir)
    for img in src_dir.glob("*"):
        if img.is_file():
            shutil.copy2(img, dst_dir / img.name)

def move_some(src_dir: Path, dst_dir: Path, count: int):   # move 'count' images from src_dir to dst_dir
    ensure_dir(dst_dir)
    files = [p for p in src_dir.glob("*") if p.is_file()]
    random.shuffle(files)
    chosen = files[:count]
    for p in chosen:
        shutil.move(str(p), dst_dir / p.name)
    return len(chosen)

def main():      
    # Expected CIFAKE layout after renaming:
    # archive/train/real, archive/train/ai, archive/test/real, archive/test/ai
    for cls in ["real", "ai"]:
        src_train = SRC_ARCHIVE / "train" / cls
        src_test = SRC_ARCHIVE / "test" / cls

        if not src_train.exists():
            raise FileNotFoundError(f"Missing: {src_train} (did you rename REAL->real and FAKE->ai?)")
        if not src_test.exists():
            raise FileNotFoundError(f"Missing: {src_test} (did you rename REAL->real and FAKE->ai?)")

        # 1) Copy ALL archive/train -> data/train
        dst_train = DST_DATA / "train" / cls
        copy_all(src_train, dst_train)

        # 2) Create validation by MOVING 15% from data/train -> data/val
        train_files = [p for p in dst_train.glob("*") if p.is_file()]
        n_val = int(len(train_files) * VAL_RATIO)

        dst_val = DST_DATA / "val" / cls
        moved = move_some(dst_train, dst_val, n_val)

        # 3) Copy ALL archive/test -> data/test
        dst_test = DST_DATA / "test" / cls
        copy_all(src_test, dst_test)

        print(f"{cls}: copied train={len(train_files)} -> moved to val={moved} -> copied test={len(list((dst_test).glob('*')))}")

    print("\nDone! You now have: data/train, data/val, data/test")

if __name__ == "__main__":
    main()
