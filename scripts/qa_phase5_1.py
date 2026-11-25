"""
scripts/qa_phase5_1.py - Phase 5.1 Integration QA Verification Suite.
Verifies:
1. Fusion gate spatial alignment, gradient direction (blue=raw, amber=restored), and single-pass parity
2. Invalid API edge cases (bad thresholds, unknown models, corrupted images, cross-model fusion viz)
3. 20 consecutive inference requests (median & p95 latency, GPU VRAM stability)
4. 4 simultaneous concurrent API requests (thread safety)
5. Frozen Milestone 3 benchmark & checkpoint SHA-256 immutability
6. Generates qualitative multi-model comparison PNG for demonstration
"""

import os
import io
import json
import time
import base64
import hashlib
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch

import sys
REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if os.path.join(REPO_DIR, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))

from retina_engine import create_fusion_gate_overlay
from models.dual_stream_unet import get_dual_stream_unet

API_BASE = "http://127.0.0.1:5001"
ARTIFACT_DIR = r"C:\Users\ARAN\.gemini\antigravity\brain\73eae9fd-ef3d-4e16-bd31-bdb8624d8e09"


def http_json(path, payload=None, raw_body=None, timeout=30):
    url = f"{API_BASE}{path}"
    if raw_body is not None:
        req = urllib.request.Request(url, data=raw_body, headers={"Content-Type": "application/json"})
    elif payload is not None:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def wait_for_server(max_wait=20):
    for _ in range(max_wait):
        try:
            code, data = http_json("/health")
            if code == 200 and data.get("status") == "ok":
                return data
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server did not become healthy in time")


def b64_to_np(b64_url):
    _, b64 = b64_url.split(",", 1)
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(im)


def test_1_fusion_correctness():
    print("\n[QA 1/5] Fusion Gate Alignment, Gradient Direction & Single-Pass Parity...")
    # 1a. Verify exact gradient direction in create_fusion_gate_overlay
    blank = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    gate = np.zeros((64, 64), dtype=np.float32)
    gate[:, 32:] = 1.0  # Left half = 0.0 (Raw), Right half = 1.0 (Restored)
    ov = create_fusion_gate_overlay(blank, gate, alpha=1.0)
    left_rgb = ov[32, 16]
    right_rgb = ov[32, 48]
    print(f"  Gradient check (alpha=1.0): w_rest=0.0 -> RGB {left_rgb.tolist()} (expected Blue ~[37,99,235]), w_rest=1.0 -> RGB {right_rgb.tolist()} (expected Amber ~[245,158,11])")
    assert left_rgb[2] > left_rgb[0], "w_rest=0.0 must be blue-dominant!"
    assert right_rgb[0] > right_rgb[2], "w_rest=1.0 must be amber/red-dominant!"

    # 1b. Verify same-forward-pass equality on DualStreamUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_dual_stream_unet().to(device)
    ck = torch.load(os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream", "best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    x_r = torch.rand(1, 3, 256, 256, device=device)
    x_s = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out_no_gate = model(x_r, x_s, return_gates=False)
        out_with_gate, gates = model(x_r, x_s, return_gates=True)
    max_diff = float(torch.max(torch.abs(out_no_gate - out_with_gate)).item())
    assert max_diff == 0.0, f"Logits differed with return_gates=True: {max_diff}"
    assert gates["scale_0"].shape == (1, 1, 256, 256), f"Unexpected scale_0 shape: {gates['scale_0'].shape}"
    print(f"  Single-pass parity verified: max logit diff = {max_diff:.1f}, scale_0 shape = {tuple(gates['scale_0'].shape)}")

    # 1c. Spatial quadrant alignment check via live API with localized high-frequency checkerboard perturbation
    base_arr = np.ones((256, 256, 3), dtype=np.float32) * 0.35
    # Inject dense grid texture only in Top-Left quadrant [20:100, 20:100]
    yy, xx = np.mgrid[0:256, 0:256]
    checker = ((yy + xx) % 2).astype(np.float32) * 0.25
    pert_arr = base_arr.copy()
    pert_arr[20:100, 20:100, 1] += checker[20:100, 20:100]

    def enc_np(arr):
        buf = io.BytesIO()
        Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    c1, r1 = http_json("/api/retina/dual_segment", {"image": enc_np(base_arr), "resolution": 256, "include_fusion_viz": True})
    c2, r2 = http_json("/api/retina/dual_segment", {"image": enc_np(pert_arr), "resolution": 256, "include_fusion_viz": True})
    assert c1 == 200 and c2 == 200
    hm1 = b64_to_np(r1["overlay_fusion_gate"]).astype(np.float32)
    hm2 = b64_to_np(r2["overlay_fusion_gate"]).astype(np.float32)
    assert hm1.shape == (256, 256, 3) and hm2.shape == (256, 256, 3)
    # Verify spatial localization: Top-Left difference >> Bottom-Right difference
    tl_diff = float(np.mean(np.abs(hm2[20:100, 20:100] - hm1[20:100, 20:100])))
    br_diff = float(np.mean(np.abs(hm2[150:230, 150:230] - hm1[150:230, 150:230])))
    print(f"  Spatial alignment check: Top-Left perturbed delta={tl_diff:.2f} vs Bottom-Right unperturbed delta={br_diff:.2f}")
    assert tl_diff > br_diff * 3.0, "Heatmap change was not spatially localized to perturbed quadrant!"


def test_2_invalid_api_inputs():
    print("\n[QA 2/5] Testing Invalid API Inputs & Cross-Model Guards...")
    cases = [
        ("Threshold < 0", {"sample_id": "hrf_unexposed_14h", "threshold": -0.1}, 400),
        ("Threshold > 1", {"sample_id": "hrf_unexposed_14h", "threshold": 1.25}, 400),
        ("Non-numeric threshold", {"sample_id": "hrf_unexposed_14h", "threshold": "abc"}, 400),
        ("Unknown selected_model", {"sample_id": "hrf_unexposed_14h", "selected_model": "imaginary_net"}, 400),
        ("Unsupported resolution", {"sample_id": "hrf_unexposed_14h", "resolution": 1024}, 400),
        ("Unknown sample_id", {"sample_id": "nonexistent_scan_xyz"}, 400),
        ("Corrupted base64 image", {"image": "data:image/png;base64,NOT_VALID_BASE64!!!"}, 400),
        ("Empty image string", {"image": ""}, 400),
    ]
    for label, payload, expected in cases:
        code, resp = http_json("/api/retina/dual_segment", payload)
        print(f"  {label}: HTTP {code} -> {resp.get('error')}")
        assert code == expected, f"{label} expected {expected}, got {code}"

    # Malformed JSON
    code, resp = http_json("/api/retina/dual_segment", raw_body=b"{bad json")
    print(f"  Malformed JSON: HTTP {code} -> {resp.get('error')}")
    assert code == 400

    # Fusion viz requested while another model selected
    code, resp = http_json("/api/retina/dual_segment", {
        "sample_id": "hrf_unexposed_14h",
        "selected_model": "raw_unet",
        "threshold": 0.50,
        "include_fusion_viz": True
    })
    assert code == 200
    assert resp["overlay_fusion_gate"] is None, "Fusion heatmap must NOT be returned when raw_unet is selected!"
    assert resp["fusion_viz_note"] is not None
    print(f"  Cross-model fusion viz guard PASS: overlay_fusion_gate=None, note='{resp['fusion_viz_note']}'")


def test_3_consecutive_latency_and_vram():
    print("\n[QA 3/5] Running 20 Consecutive Inference Requests & VRAM Stability Check...")
    samples = ["hrf_unexposed_14h", "fives_glaucoma_37db", "fives_amd_37db", "fives_normal_37db", "fives_dr_35db"]
    h_before = http_json("/health")[1].get("gpu_memory", {})
    latencies = []
    stage_diffs = []
    for i in range(20):
        sid = samples[i % len(samples)]
        viz = (i % 4 == 0)
        code, resp = http_json("/api/retina/dual_segment", {
            "sample_id": sid,
            "threshold": 0.40,
            "selected_model": "dual_stream",
            "include_fusion_viz": viz
        })
        assert code == 200, f"Request {i} failed with {code}"
        t = resp["timing"]
        latencies.append(t["total_ms"])
        stage_sum = t["neural_inference_total_ms"] + t["postprocessing_and_io_total_ms"]
        stage_diffs.append(abs(t["total_ms"] - stage_sum))

    h_after = http_json("/health")[1].get("gpu_memory", {})
    med = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    print(f"  20 requests complete: Median={med:.1f} ms, P95={p95:.1f} ms, Min={min(latencies):.1f} ms, Max={max(latencies):.1f} ms")
    print(f"  Max timing reconciliation diff: {max(stage_diffs):.1f} ms")
    print(f"  GPU Memory Before: alloc={h_before.get('allocated_mb')} MB, reserved={h_before.get('reserved_mb')} MB")
    print(f"  GPU Memory After : alloc={h_after.get('allocated_mb')} MB, reserved={h_after.get('reserved_mb')} MB")
    alloc_delta = abs(h_after.get("allocated_mb", 0) - h_before.get("allocated_mb", 0))
    assert alloc_delta < 50.0, f"Potential VRAM leak detected: delta={alloc_delta} MB"


def test_4_concurrent_requests():
    print("\n[QA 4/5] Testing 4 Simultaneous Concurrent API Requests (Thread-Safety)...")
    payloads = [
        {"sample_id": "hrf_unexposed_14h", "threshold": 0.40, "include_fusion_viz": True},
        {"sample_id": "fives_glaucoma_37db", "threshold": 0.50, "include_fusion_viz": False},
        {"sample_id": "fives_amd_37db", "threshold": 0.40, "include_fusion_viz": True},
        {"sample_id": "fives_normal_37db", "threshold": 0.40, "include_fusion_viz": False},
    ]

    def worker(p):
        t_s = time.time()
        code, resp = http_json("/api/retina/dual_segment", p, timeout=60)
        return code, (time.time() - t_s) * 1000.0, resp.get("biomarkers", {}).get("dual_stream", {}).get("vessel_density_pct")

    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(worker, payloads))

    for idx, (code, wall_ms, dens) in enumerate(outs):
        print(f"  Worker {idx}: HTTP {code} | wall={wall_ms:.1f} ms | dual_density={dens}%")
        assert code == 200 and dens is not None


def test_5_frozen_benchmarks_and_qualitative_capture():
    print("\n[QA 5/5] Verifying Frozen Benchmark Immutability & Capturing Multi-Model Comparison...")
    code, bench = http_json("/api/retina/benchmark")
    assert code == 200 and bench["status"] == "frozen_milestone3"
    assert round(bench["fives_summary"]["per_system"]["raw_unet"]["dice"]["mean"], 4) == 0.8133
    assert round(bench["fives_summary"]["per_system"]["restored_unet"]["dice"]["mean"], 4) == 0.6927
    assert bench["dual_stream"]["fives_overall"]["dice"] == 0.8474
    assert bench["dual_stream"]["fives_overall"]["thin_skel_sens"] == 0.4352
    assert bench["dual_stream"]["hrf_overall"]["dice"] == 0.7419

    # Verify exact SHA-256 of frozen checkpoints & results files
    expected_hashes = {
        "checkpoints/unet_raw/best.pth": "97639a65b6cc3d09d0638036e04ea018c236f3e8df0bf556282de6beba6f9d50",
        "checkpoints/unet_restored/best.pth": "2cfb37371cc5c0038bba6f7c3dcde808a4571c246a3c5df7bddba460ff9011cb",
        "checkpoints/unet_dual_stream/best.pth": "9be2213e0cf0c79998f31e7f9c64f4ef82e676a4ae8d74c22328aa80bcc1ee37",
        "checkpoints/fives_retina_finetune/best.pth": "d74e267c6ac664686528d6285564fcefcfddc97db1c1f0f676cc00ab5a49563c",
    }
    for rel, exp in expected_hashes.items():
        curr = hashlib.sha256(open(os.path.join(REPO_DIR, rel), "rb").read()).hexdigest()
        assert curr == exp, f"Hash mismatch on {rel}"
        print(f"  Frozen SHA-256 OK: {rel} -> {curr[:16]}...")

    # Capture multi-model qualitative demo strip (Input | Raw U-Net | Restored U-Net | Dual-Stream | Gate Map)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demo_samples = [("hrf_unexposed_14h", "HRF Unexposed (14_h)"), ("fives_normal_37db", "FIVES Healthy Arcade")]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), facecolor="#0F172A")
    cols = ["Original Fundus", "Raw U-Net (tau=0.40)", "Restored U-Net (tau=0.40)", "Dual-Stream Hybrid (tau=0.40)", "Spatial Gate Map (w_rest)"]
    for row_idx, (sid, label) in enumerate(demo_samples):
        code, r = http_json("/api/retina/dual_segment", {
            "sample_id": sid, "threshold": 0.40, "selected_model": "dual_stream", "include_fusion_viz": True
        })
        assert code == 200
        imgs = [
            b64_to_np(r["raw_img"]),
            b64_to_np(r["overlay_raw"]),
            b64_to_np(r["overlay_restored"]),
            b64_to_np(r["overlay_dual_stream"]),
            b64_to_np(r["overlay_fusion_gate"]),
        ]
        bio = r["biomarkers"]
        subtitles = [
            f"{label}",
            f"Dens: {bio['raw']['vessel_density_pct']}% | CC: {bio['raw']['branch_fragment_count']}",
            f"Dens: {bio['restored']['vessel_density_pct']}% | CC: {bio['restored']['branch_fragment_count']}",
            f"Dens: {bio['dual_stream']['vessel_density_pct']}% | CC: {bio['dual_stream']['branch_fragment_count']}",
            f"Mean w_rest: {r['fusion_gate_stats']['mean_w_rest']} (0=Raw, 1=Rest)",
        ]
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx])
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"{cols[col_idx]}\n{subtitles[col_idx]}", color="#F8FAFC", fontsize=10, fontweight="bold", pad=8)
            else:
                ax.set_title(subtitles[col_idx], color="#CBD5E1", fontsize=9.5, pad=6)

    plt.suptitle("RetinaScan Studio (Research Prototype) — Multi-Model Qualitative Demonstration", color="#38BDF8", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(ARTIFACT_DIR, "phase5_1_qualitative_multimodel_demo.png")
    plt.savefig(out_png, dpi=200, facecolor="#0F172A")
    plt.close()
    print(f"  Saved qualitative demonstration figure: {out_png}")


if __name__ == "__main__":
    h = wait_for_server()
    print(f"[QA Connected] Server OK | GPU={h.get('gpu')} | VRAM={h.get('gpu_memory')}")
    test_1_fusion_correctness()
    test_2_invalid_api_inputs()
    test_3_consecutive_latency_and_vram()
    test_4_concurrent_requests()
    test_5_frozen_benchmarks_and_qualitative_capture()
    print("\nALL PHASE 5.1 INTEGRATION QA TESTS PASSED!")
