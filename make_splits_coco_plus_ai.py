import os, zipfile, urllib.request
from pathlib import Path

def dl(url, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print("Already have", out_path.name)
        return
    print("Downloading", url)
    urllib.request.urlretrieve(url, out_path)

def unzip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Unzipping", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main():
    # COCO 2017 train/val images (real photographs)
    urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    }

    downloads = Path("downloads")
    coco_dir = downloads / "coco2017"
    coco_dir.mkdir(parents=True, exist_ok=True)

    for name, url in urls.items():
        zp = downloads / name
        dl(url, zp)
        unzip(zp, coco_dir)

    print("\nCOCO images are in:")
    print((coco_dir / "train2017").resolve())
    print((coco_dir / "val2017").resolve())

if __name__ == "__main__":
    main()
