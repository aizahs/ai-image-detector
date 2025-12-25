import csv
import os
import requests
from pathlib import Path
from time import sleep

CSV_PATH = "photos.csv"
OUT_DIR = Path("unsplash_real")
MAX_IMAGES = 3000        # start small
SLEEP_SEC = 0.3          # be polite

OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    count = 0
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count >= MAX_IMAGES:
                break

            url = row.get("photo_image_url") or row.get("photo_url")
            if not url:
                continue

            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                img_path = OUT_DIR / f"unsplash_{count}.jpg"
                with open(img_path, "wb") as out:
                    out.write(r.content)
                count += 1
                print(f"Downloaded {count}")
                sleep(SLEEP_SEC)
            except Exception as e:
                print("skip:", e)

    print("Done.")

if __name__ == "__main__":
    main()
