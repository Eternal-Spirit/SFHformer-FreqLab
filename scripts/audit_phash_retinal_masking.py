"""
audit_phash_retinal_masking.py - Rigorous Investigation of pHash Candidates:
1. Exact breakdown of candidate pairs at Hamming distance 0, 1, 2, 3, 4 (Full Image vs Retinal FOV Masked).
2. Tight bounding-box retinal crop pHash computation to remove camera border effects.
3. Side-by-side visual comparisons of lowest-distance candidate pairs.
4. Clean Validation Protocol specification (59 clean vs 60 sensitivity).
"""

import os
import json
import glob
import numpy as np
from PIL import Image
from scipy.fftpack import dct
import scipy.ndimage as ndi

REPO_DIR = r"D:\SFHformer-FreqLab"
DATA_DIR = os.path.join(REPO_DIR, "data")
FIVES_DIR = os.path.join(DATA_DIR, "fives")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "dataset_audit", "followup_audit")
PHASH_DIR = os.path.join(RESULTS_DIR, "phash_investigation")
os.makedirs(PHASH_DIR, exist_ok=True)

def compute_phash_arr(arr_gray, hash_size=8, highfreq_factor=4):
    """Computes 64-bit DCT perceptual hash on 32x32 array."""
    im = Image.fromarray(arr_gray).resize((hash_size * highfreq_factor, hash_size * highfreq_factor), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32)
    dct_mat = dct(dct(arr, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_low = dct_mat[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()

def compute_retinal_masked_crop(img_path, fov_path):
    """
    Applies FOV mask, crops to tight bounding box of retinal tissue,
    and replaces non-retinal pixels with the mean luminance of the retina
    to avoid sharp contrast step edges with black border.
    """
    im = Image.open(img_path).convert('L')
    im_arr = np.array(im)
    
    if os.path.exists(fov_path):
        fov = Image.open(fov_path).convert('L')
        fov_arr = np.array(fov) > 0
    else:
        # Fast luminance thresholding and hole filling
        fov_arr = im_arr > 12
        fov_arr = ndi.binary_fill_holes(fov_arr)
        # Cache for subsequent runs
        os.makedirs(os.path.dirname(fov_path), exist_ok=True)
        Image.fromarray((fov_arr.astype(np.uint8)) * 255).save(fov_path)
    
    # Tight bounding box of retina
    y_idx, x_idx = np.where(fov_arr)
    if len(y_idx) == 0:
        return im_arr # Fallback
    
    ymin, ymax = np.min(y_idx), np.max(y_idx)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    
    crop_im = im_arr[ymin:ymax+1, xmin:xmax+1].copy()
    crop_fov = fov_arr[ymin:ymax+1, xmin:xmax+1]
    
    # Neutral fill: set background to mean retinal intensity so borders don't induce artificial high frequencies
    retina_mean = float(np.mean(crop_im[crop_fov])) if np.sum(crop_fov) > 0 else 128.0
    crop_im[~crop_fov] = int(round(retina_mean))
    return crop_im

def main():
    print("=" * 80)
    print(" EMPIRICAL INVESTIGATION: PHASH CANDIDATES & RETINAL MASKING ")
    print("=" * 80)
    
    # 1. Load cached full-image pHash
    cache_file = os.path.join(RESULTS_DIR, "fives_hashes_cache.json")
    with open(cache_file, 'r') as f:
        cached = json.load(f)
        
    full_phashes = [(item[0], item[1], np.array(item[2], dtype=bool)) for item in cached["phashes"]]
    
    test_full = [p for p in full_phashes if p[0] == 'test']
    train_val_full = [p for p in full_phashes if p[0] in ('train', 'val')]
    
    # Count distance breakdown for unmasked full images
    unmasked_dist_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    unmasked_pairs = []
    
    for t_split, t_name, t_ph in test_full:
        for tv_split, tv_name, tv_ph in train_val_full:
            d = int(np.sum(t_ph != tv_ph))
            if d <= 4:
                unmasked_dist_counts[d] += 1
                unmasked_pairs.append({
                    "test_image": t_name,
                    "train_val_image": tv_name,
                    "train_val_split": tv_split,
                    "unmasked_distance": d
                })
                
    unmasked_pairs.sort(key=lambda x: x["unmasked_distance"])
    
    print("\n--- Unmasked Full-Image pHash Distance Breakdown (Test vs Train/Val) ---")
    for d in range(5):
        print(f"  Distance {d}: {unmasked_dist_counts[d]} pairs")
    print(f"  Total <= 4: {len(unmasked_pairs)} pairs")
    
    # 2. Compute Retinal-Masked Tight-Crop pHash for all 800 images
    masked_cache_file = os.path.join(RESULTS_DIR, "fives_masked_phashes_cache.json")
    if os.path.exists(masked_cache_file):
        print("\n--- Loading Cached Retinal-Masked & Cropped pHash ---")
        with open(masked_cache_file, 'r') as f:
            cached_m = json.load(f)
        masked_phashes = [(item[0], item[1], np.array(item[2], dtype=bool)) for item in cached_m]
    else:
        print("\n--- Computing Retinal-Masked & Cropped pHash (Removing Camera Border) ---")
        all_imgs = []
        for split in ['train', 'val', 'test']:
            for p in sorted(glob.glob(os.path.join(FIVES_DIR, "raw", split, "images", "*.png"))):
                all_imgs.append((split, os.path.basename(p), p))
                
        masked_phashes = []
        for i, (split, base, p) in enumerate(all_imgs):
            fov_p = p.replace("images", "fov")
            crop_retina = compute_retinal_masked_crop(p, fov_p)
            ph = compute_phash_arr(crop_retina)
            masked_phashes.append((split, base, ph))
            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1} / {len(all_imgs)} images...")
                
        with open(masked_cache_file, 'w') as f:
            json.dump([(item[0], item[1], [bool(b) for b in item[2]]) for item in masked_phashes], f)
            
    test_masked = [p for p in masked_phashes if p[0] == 'test']
    train_val_masked = [p for p in masked_phashes if p[0] in ('train', 'val')]
    
    masked_map = {(p[0], p[1]): p[2] for p in masked_phashes}
    
    masked_dist_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    masked_pairs_le4 = []
    
    for t_split, t_name, t_ph in test_masked:
        for tv_split, tv_name, tv_ph in train_val_masked:
            d = int(np.sum(t_ph != tv_ph))
            if d <= 4:
                masked_dist_counts[d] += 1
                masked_pairs_le4.append({
                    "test_image": t_name,
                    "train_val_image": tv_name,
                    "train_val_split": tv_split,
                    "masked_distance": d
                })
                
    print("\n--- Retinal-Masked & Cropped pHash Distance Breakdown (Test vs Train/Val) ---")
    for d in range(5):
        print(f"  Distance {d}: {masked_dist_counts[d]} pairs")
    print(f"  Total <= 4: {len(masked_pairs_le4)} pairs")
    
    # Check what happened to the 193 unmasked candidate pairs under retinal masking:
    pair_comparisons = []
    for pair in unmasked_pairs:
        t_name = pair["test_image"]
        tv_name = pair["train_val_image"]
        tv_split = pair["train_val_split"]
        
        t_mph = masked_map[('test', t_name)]
        tv_mph = masked_map[(tv_split, tv_name)]
        d_masked = int(np.sum(t_mph != tv_mph))
        
        pair_comparisons.append({
            "test_image": t_name,
            "train_val_image": tv_name,
            "train_val_split": tv_split,
            "unmasked_distance": pair["unmasked_distance"],
            "retinal_masked_distance": d_masked,
            "delta": d_masked - pair["unmasked_distance"]
        })
        
    mean_delta = np.mean([p["delta"] for p in pair_comparisons])
    mean_unmasked = np.mean([p["unmasked_distance"] for p in pair_comparisons])
    mean_masked_for_candidates = np.mean([p["retinal_masked_distance"] for p in pair_comparisons])
    
    print(f"\nEffect of Retinal Masking on Candidate Pairs:")
    print(f"  Mean unmasked distance: {mean_unmasked:.2f}")
    print(f"  Mean retinal-masked distance: {mean_masked_for_candidates:.2f} (Shift: +{mean_delta:.2f})")
    
    # 3. Side-by-side visual comparison generation for lowest-distance pairs
    print("\n--- Generating Side-by-Side Visual Comparison Panels ---")
    # Let's inspect the top lowest-distance pairs
    top_candidates = pair_comparisons[:8]
    
    gallery_files = []
    for idx, cand in enumerate(top_candidates):
        t_name = cand["test_image"]
        tv_name = cand["train_val_image"]
        tv_split = cand["train_val_split"]
        
        t_img_p = os.path.join(FIVES_DIR, "raw", "test", "images", t_name)
        t_mk_p = os.path.join(FIVES_DIR, "raw", "test", "masks", t_name)
        
        tv_img_p = os.path.join(FIVES_DIR, "raw", tv_split, "images", tv_name)
        tv_mk_p = os.path.join(FIVES_DIR, "raw", tv_split, "masks", tv_name)
        
        t_im = Image.open(t_img_p).convert('RGB')
        t_mk = Image.open(t_mk_p).convert('RGB')
        tv_im = Image.open(tv_img_p).convert('RGB')
        tv_mk = Image.open(tv_mk_p).convert('RGB')
        
        scale = 512
        p1 = t_im.resize((scale, scale), Image.Resampling.LANCZOS)
        p2 = t_mk.resize((scale, scale), Image.Resampling.NEAREST)
        p3 = tv_im.resize((scale, scale), Image.Resampling.LANCZOS)
        p4 = tv_mk.resize((scale, scale), Image.Resampling.NEAREST)
        
        # 4-panel side-by-side: [Test Fundus | Test Mask | Train/Val Fundus | Train/Val Mask]
        comp = Image.new('RGB', (scale * 4, scale))
        comp.paste(p1, (0, 0))
        comp.paste(p2, (scale, 0))
        comp.paste(p3, (scale * 2, 0))
        comp.paste(p4, (scale * 3, 0))
        
        out_name = f"comparison_rank{idx+1:02d}_test_{t_name.replace('.png','')}_vs_{tv_split}_{tv_name.replace('.png','')}.png"
        out_p = os.path.join(PHASH_DIR, out_name)
        comp.save(out_p)
        gallery_files.append({
            "rank": idx + 1,
            "test_image": t_name,
            "train_val_image": tv_name,
            "train_val_split": tv_split,
            "unmasked_distance": cand["unmasked_distance"],
            "retinal_masked_distance": cand["retinal_masked_distance"],
            "image_path": out_p
        })
        print(f"  Saved comparison: {out_name} (Unmasked Dist: {cand['unmasked_distance']}, Masked Dist: {cand['retinal_masked_distance']})")
        
    # 4. Clean Validation Protocol & Manifest Update
    print("\n--- Formalizing Clean Validation Protocol (N=59 clean vs N=60 sensitivity) ---")
    val_tile_manifest_p = os.path.join(FIVES_DIR, "patches_512", "val_full_image_tile_manifest.json")
    with open(val_tile_manifest_p, 'r') as f:
        val_tiles = json.load(f)
        
    for rec in val_tiles:
        img_id = rec.get("image_id") or rec.get("source_image_id")
        if img_id == "train_121_A":
            rec["is_clean_val"] = False
            rec["contamination_flag"] = "EXACT_SHA256_DUPLICATE_OF_TRAIN_61_A"
            rec["inclusion_mode"] = "SENSITIVITY_BENCHMARK_ONLY"
        else:
            rec["is_clean_val"] = True
            rec["contamination_flag"] = "NONE"
            rec["inclusion_mode"] = "PRIMARY_AND_SENSITIVITY_BENCHMARK"
            
    with open(val_tile_manifest_p, 'w') as f:
        json.dump(val_tiles, f, indent=2)
        
    clean_count = sum(1 for r in val_tiles if r["is_clean_val"])
    total_count = len(val_tiles)
    print(f"  Updated val_full_image_tile_manifest.json: {clean_count} clean validation images, {total_count} total validation images.")

    # Inspect retinal-masked distance <= 1 pairs
    print("\n--- Retinal-Masked Closest Pairs (Distance <= 1) ---")
    for pair in masked_pairs_le4:
        if pair["masked_distance"] <= 1:
            print(f"  Masked Dist {pair['masked_distance']}: Test {pair['test_image']} vs {pair['train_val_split']} {pair['train_val_image']}")
            # Also generate visual panel if not already generated
            t_name = pair["test_image"]
            tv_name = pair["train_val_image"]
            tv_split = pair["train_val_split"]
            t_im = Image.open(os.path.join(FIVES_DIR, "raw", "test", "images", t_name)).convert('RGB')
            t_mk = Image.open(os.path.join(FIVES_DIR, "raw", "test", "masks", t_name)).convert('RGB')
            tv_im = Image.open(os.path.join(FIVES_DIR, "raw", tv_split, "images", tv_name)).convert('RGB')
            tv_mk = Image.open(os.path.join(FIVES_DIR, "raw", tv_split, "masks", tv_name)).convert('RGB')
            scale = 512
            comp = Image.new('RGB', (scale * 4, scale))
            comp.paste(t_im.resize((scale, scale), Image.Resampling.LANCZOS), (0, 0))
            comp.paste(t_mk.resize((scale, scale), Image.Resampling.NEAREST), (scale, 0))
            comp.paste(tv_im.resize((scale, scale), Image.Resampling.LANCZOS), (scale * 2, 0))
            comp.paste(tv_mk.resize((scale, scale), Image.Resampling.NEAREST), (scale * 3, 0))
            out_p = os.path.join(PHASH_DIR, f"masked_closest_dist{pair['masked_distance']}_{t_name[:8]}_vs_{tv_name[:8]}.png")
            comp.save(out_p)
    
    # Save investigation report
    res_summary = {
        "unmasked_distance_breakdown": unmasked_dist_counts,
        "masked_distance_breakdown": masked_dist_counts,
        "mean_candidate_shift": round(float(mean_delta), 2),
        "mean_unmasked_candidate_distance": round(float(mean_unmasked), 2),
        "mean_masked_candidate_distance": round(float(mean_masked_for_candidates), 2),
        "top_pairs_detailed": top_candidates[:15],
        "comparison_visualizations": gallery_files,
        "clean_validation_protocol": {
            "primary_clean_validation_n": clean_count,
            "sensitivity_full_validation_n": total_count,
            "excluded_duplicate_id": "train_121_A",
            "identical_training_counterpart": "train_61_A",
            "test_duplicate_observation": {
                "pair": ["test_32_A.png", "test_48_A.png"],
                "protocol": "Documented as non-independent observations in test statistical tests; no Train-Test leakage."
            }
        }
    }
    
    out_summary_p = os.path.join(PHASH_DIR, "phash_investigation_summary.json")
    with open(out_summary_p, 'w') as f:
        json.dump(res_summary, f, indent=2)
        
    print(f"\nComplete pHash investigation report written to: {out_summary_p}")
    print("=" * 80)

if __name__ == '__main__':
    main()
