# SFHformer-FreqLab: Engineering & System Implementation Documentation

> **Target Audience**: ML Engineers, Infrastructure Maintainers & Technical Reviewers  
> **Snapshot Tag**: `v3.1-phase5.2-verified` (`42ac0565d1d1517ef5bad9fdc1f392608272db1b`)  
> **Scope Notice**: RetinaScan Studio is a **research prototype for retinal vessel segmentation**, not a certified clinical diagnostic product.


## 1. Repository Layout & Module Ownership

```
D:\SFHformer-FreqLab\
├── models/
│   ├── SFHformer.py               # 5-stage SFHformer_m (14.18M params)
│   ├── unet.py                    # 4-stage DoubleConv + Bilinear UNet (7.7M params)
│   └── dual_stream_unet.py        # DualStreamUNet + CrossStreamFusionBlock (16.41M params)
├── freqlab/
│   ├── gate.py                    # RadialBandGate (16 learnable radial spectral bands)
│   ├── sfhformer_gated.py         # GatedBackbone (R2 variant)
│   └── metrics.py / classical.py  # PSNR, SSIM, Notch/Gaussian/Median filters
├── utils/
│   └── losses.py                  # BCEDiceLoss, SoftSkeletonize(FP32), HybridTopologyLoss
├── scripts/
│   ├── train_fives_retina.py      # Online FIVES restoration fine-tuning
│   ├── train_unet.py              # Single-stream Raw & Restored U-Net training
│   ├── train_dual_stream_unet.py  # Two-stage Dual-Stream training + N=59 calibration
│   ├── evaluate_milestone3.py     # Frozen baseline Milestone 3 harness
│   ├── evaluate_dual_stream_milestone3.py  # Frozen Dual-Stream evaluation + Holm-Bonferroni
│   ├── serve_restore.py           # Flask REST API server (:5001) with GPU_LOCK
│   ├── retina_engine.py           # Overlays, create_fusion_gate_overlay, CCL & skeleton
│   └── qa_phase5_1.py             # Automated API, alignment, latency, concurrency QA
├── frontend/src/pages/RetinaStudio.jsx   # React 18 / Vite RetinaScan Studio UI (:3000)
├── checkpoints/                   # Frozen .pth weights + validation_calibration.json
├── results/milestone3/            # Immutable per-image JSON experimental outputs
├── demo_package/                  # Phase 5.3 diagrams, qualitative strips, manifest
└── final_documentation/           # Phase 6 reports, sources & DOCUMENTATION_MANIFEST.json
```



## 2. Model Loading, State-Dict Mapping & Checkpoint Provenance

### 2.1 Dual-Stream Warm-Start Contract (`models/dual_stream_unet.py`)
`DualStreamUNet.load_pretrained_streams(raw_ckpt_path, rest_ckpt_path, init_decoder_from="raw")` maps single-stream `UNet` keys cleanly onto dual encoders without shape ambiguity:
- `downs.0..3.*` & `bottleneck.*` from `unet_raw/best.pth` $\to$ `raw_downs.*`, `raw_bottleneck.*`
- `downs.0..3.*` & `bottleneck.*` from `unet_restored/best.pth` $\to$ `rest_downs.*`, `rest_bottleneck.*`
- `ups.*` & `final_conv.*` from `unet_raw/best.pth` $\to$ shared decoder `ups.*`, `final_conv.*`
- Verified `208` tensors loaded cleanly; `80` parameters/buffers across `fusions[0..3]` and `bottleneck_fusion` initialized fresh.

### 2.2 Checkpoint Inventory & Hash Reconciliation

| Alias / Identifier | File Path | Size | Format Contents | SHA-256 Prefix (`Full in SNAPSHOT_MANIFEST`) |
| :--- | :--- | :---: | :--- | :--- |
| `sfhformer` / `retina_fives` | `checkpoints/fives_retina_finetune/best.pth` | `67.80 MB` | Model + Optimizer | `d74e267c6ac664686528d628...` |
| `unet_raw` / `raw` | `checkpoints/unet_raw/best.pth` | `86.97 MB` | Model + Optimizer | `97639a65b6cc3d09d0638036...` |
| `unet_restored` / `restored` | `checkpoints/unet_restored/best.pth` | `86.97 MB` | Model + Optimizer | `2cfb37371cc5c0038bba6f7c...` |
| `unet_dual_stream` | `checkpoints/unet_dual_stream/best.pth` | `62.74 MB` | Model `state_dict` + Val Meta | `9be2213e0cf0c79998f31e7f...` |

> **Reconciliation Note**: In `scripts/evaluate_milestone3.py`, key `"sfhformer"` maps directly to `checkpoints/fives_retina_finetune/best.pth` (`d74e267c...`). The `62.74 MB` size of `unet_dual_stream/best.pth` reflects saving `model.state_dict()` (`16.41M` params) without optimizer momentum buffers.

![Architecture Overview of DualStreamUNet & CrossStreamFusionBlock Implementation|6.8|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_4_dual_stream_hybrid_unet.png)



## 3. Inference Pipeline, GPU Memory Management & API Specification

### 3.1 Hardware Safety & Thread Serialization
- **Target Device**: NVIDIA GeForce RTX 5060 Laptop GPU (`8,150.6 MB` total VRAM).
- **Training Allocator Cap**: Training scripts enforce `torch.cuda.set_per_process_memory_fraction(0.65)` (Dual-Stream peak Stage 2 VRAM = `2,948.8 MB`, `batch_size=2, accum=2`, BF16 AMP) or `0.70` (`5.57 GB`).
- **Serving Concurrency (`GPU_LOCK`)**: Inside `scripts/serve_restore.py`, Flask runs with `threaded=True`. All CUDA forward passes in `/api/retina/dual_segment` execute inside `with GPU_LOCK:` (`threading.Lock()`), serializing GPU memory allocation so `4` simultaneous requests execute cleanly with `0.0 MB` VRAM leak (`216.5 MB` allocated / `1,436.0 MB` reserved steady state).

### 3.2 REST API Endpoints (`http://127.0.0.1:5001`)

| Endpoint | Method | Purpose & Payload | Validation Guards |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | Device, loaded `MODELS` keys, `gpu_memory` (`allocated_mb`, `reserved_mb`, `free_mb`) | Always `200 OK` when alive |
| `/api/retina/samples` | `GET` | Returns 15 curated fundus/OCT sample definitions (`hrf_unexposed_14h`, FIVES, DRIVE, STARE, OCT) | Static + disk catalog |
| `/api/retina/benchmark` | `GET` | Frozen Milestone 3 records (`fives_summary`, `hrf_overall`, `hrf_unexposed`, `dual_stream`) | Cached read-only from JSONs |
| `/api/retina/restore` | `POST` | Runs synthetic ophthalmic corruption + single SFHformer restoration + Frangi ablation | `400` on missing image |
| `/api/retina/dual_segment` | `POST` | Runs SFHformer + Raw/Restored U-Net + `DualStreamUNet` + topology + overlays | Strict HTTP `400` guards (below) |

#### `/api/retina/dual_segment` Input Validation Matrix
- `threshold`: Finite float in `[0.0, 1.0]` (defaults to `0.40`). Out-of-range or non-numeric $\to$ `HTTP 400`.
- `selected_model`: Must be in `['dual_stream', 'raw_unet', 'restored_unet']`. Unknown string $\to$ `HTTP 400`.
- `resolution`: Must be integer `256` or `512`. Unsupported size $\to$ `HTTP 400`.
- `sample_id` / `image`: Unknown `sample_id`, empty string, corrupted Base64, or malformed JSON $\to$ `HTTP 400`.
- `include_fusion_viz`: Boolean (default `False`). When `True` alongside `selected_model != 'dual_stream'`, suppresses `overlay_fusion_gate: null` and returns explanatory `fusion_viz_note`.

### 3.3 6-Stage Reconciled Latency Instrumentation
Every response from `/api/retina/dual_segment` accounts for `100%` of request wall time across 6 sequential timers (discrepancy `<= 0.1 ms` across 20 consecutive QA runs):
1. `input_io_and_prep_ms`: JSON parsing, disk read / Base64 decode, `512x512` bilinear resize.
2. `sfhformer_restoration_ms`: `SFHformer-m` GPU forward pass (`[0, 1]` clamped).
3. `dual_unet_segmentation_ms` + `dual_stream_segmentation_ms`: BF16 AMP forward passes for `unet_raw`, `unet_restored`, and `unet_dual_stream` (single pass returning both `logits_dual` and `scale_0` gate when requested).
4. `topology_analysis_ms`: FOV mask (`clean.mean(axis=2) > 10/255`), binarization (`>= threshold`), `scipy.ndimage.label`, `skimage.morphology.skeletonize`, and endpoint convolution.
5. `overlay_blending_ms`: Alpha compositing for Raw, Restored, Differential Amber, Dual-Stream, and optional Spatial Gate overlays.
6. `fidelity_and_base64_encoding_ms`: PSNR/SSIM calculation and lossless PNG Base64 serialization.
- **Measured QA Performance (20 runs, RTX 5060)**: **Median = `1,300.8 ms`**, **P95 = `1,724.9 ms`**, Min = `1,019.7 ms`, Max = `1,740.3 ms`.



## 4. Spatial Fusion Visualization & Frontend Architecture

### 4.1 `create_fusion_gate_overlay()` Implementation (`scripts/retina_engine.py`)
When `include_fusion_viz=True` and `selected_model='dual_stream'`, `u_dual(x_raw, x_restored, return_gates=True)` extracts `gates['scale_0']` (`512x512` or `256x256`) from the **exact same forward pass** as segmentation (`max logit diff = 0.0`). Pure vectorized NumPy maps `w_rest` in `[0, 1]` across a 3-stop colormap (`0.0` Blue `[37,99,235]` $\to$ `0.5` Teal `[20,184,166]` $\to$ `1.0` Amber `[245,158,11]`) blended over fundus tissue (`alpha=0.58`).

![Combined End-to-End RetinaScan Studio System Architecture|6.8|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_5_combined_retinascan_system.png)

### 4.2 Frontend State & Safeguards (`RetinaStudio.jsx`)
- **Active Segmentation Model**: Defaults to `Dual-Stream Hybrid (Default)`; switching to `Raw U-Net` or `Restored U-Net` cleanly updates the view and disables the optional `w_rest` gate checkbox.
- **Operating Threshold (`tau`)**: Defaults to `0.40 (Calibrated)` with an explicit `0.50 (Baseline Match)` preset button. Adjusting the exploratory slider displays `Exploratory View Mode` without altering Tab 2 frozen benchmark data.
- **Prototype Notice**: Header explicitly states *"Research Prototype for Retinal Vessel Segmentation & Frequency Restoration Analysis (Not a Certified Diagnostic System)"*.



## 5. Automated Integration QA & Deployment Verification

### 5.1 Phase 5.1 Backend QA Results (`scripts/qa_phase5_1.py`)
- **Spatial Gate Alignment & Polarity**: `w_rest=0.0 -> RGB [37,99,235]`, `w_rest=1.0 -> RGB [245,158,11]`; checkerboard perturbation yielded `Top-Left delta = 12.27` vs `Bottom-Right = 2.45`.
- **Edge-Case Guards**: All `9` malformed/out-of-range API calls returned structured `HTTP 400` JSON errors; cross-model fusion guard passed.
- **Concurrency & Stability**: `4/4` concurrent threads returned `HTTP 200` (`1,028.2 ms` to `3,189.9 ms` wall); `20` sequential calls showed `0.0 MB` VRAM drift (`216.5 MB` alloc before & after).

### 5.2 Phase 5.2 Live Chromium QA (`Edg/157.0.4341.60` via `--remote-debugging-port=9222`)
All `11/11` interactive frontend assertions passed live against `http://localhost:3000/retina` + `http://127.0.0.1:5001`:
- Verified default `Dual-Stream Hybrid` + `0.40` calibrated notice, Raw/Restored switching, `0.50` preset, `0.65` exploratory slider isolation, hidden-by-default heatmap + `0.0 -> 1.0` legend, separate FIVES/HRF benchmark disclosures, `naturalWidth > 0` across all overlays, and `0` JS/React runtime errors.

![Live Chromium Tab 1 Verification Screenshot|6.8|4.1](D:/SFHformer-FreqLab/demo_package/qualitative_examples/phase5_2_ui_tab1_fusion.png)

### 5.3 Deployment Commands
```powershell
# Backend (Port 5001)
d:\SFHformer-FreqLab\.venv\python.exe -u scripts\serve_restore.py

# Frontend (Port 3000)
Set-Location "C:\Users\ARAN\.gemini\antigravity-ide\scratch\freqlab-ui"; npm run dev
```
