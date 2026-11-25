"""
download_fives.py - Download FIVES dataset parquet files from Hugging Face.
Dataset: tyluan/FIVES (800 2048x2048 fundus images with vessel masks).
"""

import os
import sys
import time
import urllib.request

DATA_DIR = r"D:\SFHformer-FreqLab\data\fives"
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
os.makedirs(PARQUET_DIR, exist_ok=True)

FILES = [
    "data/test-00000-of-00001.parquet",
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
]

BASE_URL = "https://huggingface.co/datasets/tyluan/FIVES/resolve/main"

def download_file(rel_path):
    filename = os.path.basename(rel_path)
    out_path = os.path.join(PARQUET_DIR, filename)
    url = f"{BASE_URL}/{rel_path}"
    
    print(f"\n[FIVES Downloader] Fetching {filename}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1 MB chunks
        t0 = time.time()
        
        # Check if already downloaded
        if os.path.exists(out_path) and os.path.getsize(out_path) == total_size:
            print(f" -> Already exists and verified ({total_size / (1024*1024):.1f} MB). Skipping.")
            return out_path
            
        with open(out_path, 'wb') as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - t0
                speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.01)
                percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"\r [{percent:5.1f}%] {downloaded/(1024*1024):6.1f} / {total_size/(1024*1024):6.1f} MB @ {speed_mb:5.2f} MB/s", end="", flush=True)
                
    print(f"\n -> Finished in {time.time() - t0:.1f}s.")
    return out_path

if __name__ == '__main__':
    t_start = time.time()
    for f in FILES:
        download_file(f)
    print(f"\n[FIVES Downloader] All 4 parquet files downloaded in {time.time() - t_start:.1f}s.")
