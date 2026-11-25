"""
scripts/prepare_milestone3_test_restoration.py
Restores:
1. 200 FIVES test images (2048x2048) -> data/fives/restored_test/images/
2. 45 HRF images (3504x2336) -> data/hrf/restored/images/
using the frozen SFHFormer-Retina checkpoint.
Enforces hardware safeguards: 70% VRAM cap, BF16 AMP, chunked inference.
Verifies SHA-256 before and after.
"""

import os
import sys
import time
import json
import glob
import hashlib
import numpy as np
from PIL import Image
import torch

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from freqlab.sfhformer_gated import SFHformer_m_gated

# Hardware safeguards
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.70)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_PATH = os.path.join(REPO_DIR, "checkpoints", "fives_retina_finetune", "best.pth")
EXPECTED_SHA = "d74e267c6ac664686528d6285564fcefcfddc97db1c1f0f676cc00ab5a49563c"

def sha256_file(fname):
    h = hashlib.sha256()
    with open(fname, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_hamming_window(tile_size=512):
    w1d = np.hamming(tile_size).astype(np.float32)
    w2d = np.outer(w1d, w1d)
    w2d = np.maximum(w2d, 1e-4)
    return torch.from_numpy(w2d).to(DEVICE)

def load_model():
    curr_sha = sha256_file(CKPT_PATH)
    print(f"[Provenance] Checking SFHFormer checkpoint hash...")
    print(f" -> Found:    {curr_sha}")
    print(f" -> Expected: {EXPECTED_SHA}")
    assert curr_sha == EXPECTED_SHA, f"Checkpoint mismatch! Expected {EXPECTED_SHA}, got {curr_sha}"
    print(" -> Checkpoint verified 100% untampered.")

    model = SFHformer_m_gated(use_gate=False).to(DEVICE)
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    st = ck['model'] if 'model' in ck else ck
    model.load_state_dict(st)
    model.eval()
    return model

def restore_canvas_sliding_window(model, img_arr, coords, tile_size=512, chunk_sz=5, window_2d=None):
    H, W = img_arr.shape[:2]
    accum_rest = torch.zeros((H, W, 3), dtype=torch.float32, device=DEVICE)
    accum_weight = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
    
    tile_tensors = []
    for c in coords:
        t, l = c["top"], c["left"]
        tile = img_arr[t:t+tile_size, l:l+tile_size]
        tile_tensors.append(torch.from_numpy(tile).permute(2, 0, 1))
        
    x_tiles = torch.stack(tile_tensors, dim=0).to(DEVICE)
    
    out_chunks = []
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for ch in range(0, len(x_tiles), chunk_sz):
                pred_chunk = model(x_tiles[ch:ch+chunk_sz])
                out_chunks.append(torch.clamp(pred_chunk, 0.0, 1.0).float())
                
    out_tiles = torch.cat(out_chunks, dim=0).permute(0, 2, 3, 1)
    
    w_tile = window_2d.unsqueeze(-1)
    for k, c in enumerate(coords):
        t, l = c["top"], c["left"]
        accum_rest[t:t+tile_size, l:l+tile_size] += out_tiles[k] * w_tile
        accum_weight[t:t+tile_size, l:l+tile_size] += window_2d
        
    full_restored = accum_rest / accum_weight.unsqueeze(-1)
    full_np = (full_restored.cpu().numpy() * 255.0).round().astype(np.uint8)
    return full_np

def restore_fives_test(model):
    print("\n" + "=" * 70)
    print(" [Part 1/2] Restoring 200 FIVES Test Images (2048x2048) ")
    print("=" * 70)
    out_dir = os.path.join(REPO_DIR, "data", "fives", "restored_test", "images")
    os.makedirs(out_dir, exist_ok=True)
    
    # 25 tiles grid for 2048x2048
    coords = []
    offsets = [0, 384, 768, 1152, 1536]
    for top in offsets:
        for left in offsets:
            coords.append({"top": top, "left": left})
            
    window_2d = get_hamming_window(512)
    
    img_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "images")
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert len(files) == 200, f"Expected 200 test images, found {len(files)}"
    
    t0 = time.time()
    for idx, f in enumerate(files):
        base = os.path.basename(f)
        out_p = os.path.join(out_dir, base)
        if os.path.exists(out_p):
            continue
        im = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
        rest_np = restore_canvas_sliding_window(model, im, coords, 512, 5, window_2d)
        Image.fromarray(rest_np).save(out_p, compress_level=1)
        if (idx + 1) % 25 == 0 or (idx + 1) == 200:
            print(f" -> [{idx+1}/200] FIVES test images restored ({(time.time()-t0)/(idx+1):.2f}s/img)")
            
    print(f" -> Completed FIVES Test restoration in {time.time()-t0:.1f}s.")

def restore_hrf(model):
    print("\n" + "=" * 70)
    print(" [Part 2/2] Restoring 45 HRF Images (3504x2336) ")
    print("=" * 70)
    out_dir = os.path.join(REPO_DIR, "data", "hrf", "restored", "images")
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_p = os.path.join(REPO_DIR, "data", "hrf", "hrf_tile_manifest_512.json")
    with open(manifest_p, 'r') as f:
        records = json.load(f)
        
    coords_by_id = {r['image_id']: r['tile_coordinates'] for r in records}
    window_2d = get_hamming_window(512)
    
    img_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "images")
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert len(files) == 45, f"Expected 45 HRF images, found {len(files)}"
    
    t0 = time.time()
    for idx, f in enumerate(files):
        base = os.path.basename(f)
        img_id = os.path.splitext(base)[0]
        out_p = os.path.join(out_dir, base)
        if os.path.exists(out_p):
            continue
        coords = coords_by_id[img_id]
        im = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
        rest_np = restore_canvas_sliding_window(model, im, coords, 512, 5, window_2d)
        Image.fromarray(rest_np).save(out_p, compress_level=1)
        if (idx + 1) % 10 == 0 or (idx + 1) == 45:
            print(f" -> [{idx+1}/45] HRF images restored ({(time.time()-t0)/(idx+1):.2f}s/img)")
            
    print(f" -> Completed HRF restoration in {time.time()-t0:.1f}s.")

def main():
    t_start = time.time()
    model = load_model()
    restore_fives_test(model)
    restore_hrf(model)
    
    # Final check of integrity
    final_sha = sha256_file(CKPT_PATH)
    assert final_sha == EXPECTED_SHA, f"Checkpoint tampered! Expected {EXPECTED_SHA}, got {final_sha}"
    print(f"\n[Integrity] Checkpoint SHA-256 confirmed invariant: {final_sha}")
    print(f"[Done] Total restoration time: {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    main()
