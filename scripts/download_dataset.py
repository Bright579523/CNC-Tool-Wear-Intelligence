import os
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DATA_DIR / "3_Milling.zip"
MAT_PATH = DATA_DIR / "mill.mat"

URL = "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"

def download_dataset():
    if not MAT_PATH.exists():
        if not ZIP_PATH.exists():
            print(f"Downloading from {URL} to {ZIP_PATH}...")
            req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
                total_length = response.getheader('content-length')
                if total_length:
                    print(f"File size: {int(total_length)/(1024*1024):.2f} MB")
                out_file.write(response.read())
            print("Download completed.")
        
        print(f"Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction completed. Files in data directory:")
        for p in DATA_DIR.iterdir():
            print(f" - {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"Dataset already exists at {MAT_PATH} ({MAT_PATH.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    download_dataset()
