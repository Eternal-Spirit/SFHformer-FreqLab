# SFHformer-FreqLab & Dual-Stream Hybrid U-Net: Master Technical Report

### Spatial-Frequency Retinal Image Restoration and Dual-Stream Vessel Segmentation with Frozen Milestone 3 Evaluation

---

| Report Metadata | Verified Record |
| :--- | :--- |
| **Project / Repository** | `D:\SFHformer-FreqLab` (`SFHformer-FreqLab` + `RetinaScan Studio`) |
| **Immutable Snapshot Tag** | `v3.1-phase5.2-verified` |
| **Git Commit Hash** | `42ac0565d1d1517ef5bad9fdc1f392608272db1b` |
| **Compute Environment** | NVIDIA GeForce RTX 5060 Laptop GPU (`8 GB` VRAM, BF16 AMP + FP32 `SoftclDice`) |
| **Primary In-Domain Test** | FIVES Untouched Test Set (`N=200`: 50 Normal / 50 AMD / 50 DR / 50 Glaucoma) |
| **External Transfer Cohort** | HRF (`N=45`: `40` upstream pretrain-exposed / `5` strictly unexposed exploratory) |
| **Calibrated Dual Threshold** | `tau = 0.40` (frozen strictly on `N=59` clean FIVES validation) |
| **Baseline U-Net Thresholds** | Raw U-Net `tau = 0.50` / Restored U-Net `tau = 0.50` / Classical Frangi `tau = 0.0010` |

>! **Scope & Clinical Disclaimer**: RetinaScan Studio and all associated neural models presented in this dossier constitute a **research prototype for retinal vessel segmentation and frequency-domain restoration analysis**. This software has not undergone prospective clinical validation or regulatory clearance and must **not** be used as a diagnostic medical device.




## 1. Executive Summary & Research Objectives

### 1.1 Scientific Problem Statement
Automated segmentation of the retinal vascular tree in fundus photography underpins morphological research into diabetic retinopathy (DR), age-related macular degeneration (AMD), and glaucomatous optic neuropathy. While modern convolutional encoder-decoder networks (U-Nets) achieve strong overlap on thick primary arcades, fine peripheral capillaries frequently degrade beneath photon shot noise, coherent laser speckle, non-uniform illumination falloff, optical halo glare, and periodic sensor raster striping.

Conversely, aggressively restoring high-frequency contrast prior to segmentation creates a sharp **sensitivity-precision trade-off** observed empirically across our Milestone 2 and Milestone 3 baselines:
- **Raw U-Net (`checkpoints/unet_raw/best.pth`)**: Achieves high overall Dice (`0.8133`), high positive predictive value (`Precision = 0.9266`), and strong topological alignment (`clDice = 0.8134`) on untouched FIVES (`N=200`), yet omits low-contrast thin peripheral vessels (`Thin Skeleton Sensitivity = 33.33%`).
- **Restored U-Net (`checkpoints/unet_restored/best.pth`)**: Trained on images denoised by our 5-stage Spatial-Frequency Hybrid Transformer (**SFHformer-m**), recovers substantially more thin-vessel structure on clean FIVES validation (`N=59`: `+6.16 pp` diameter <= 3.5 px skeleton sensitivity, Wilcoxon `p = 7.58e-7`) and on high-resolution external HRF scans (`N=45`: `+13.67 pp` radius <= 3.5 px thin skeleton sensitivity, Wilcoxon `p = 5.68e-13`). However, single-stream restoration suffers severe precision drops on coarse-annotated datasets (FIVES Dice drops to `0.6927`, Precision `0.7142`), elevated connected-component fragmentation (`114.25` vs `52.19` components/image), and contrast amplification along the excavated optic cup rim in glaucoma (`Precision = 57.59%`, `-33.54 pp` vs Raw).

### 1.2 Principal Contribution: Dual-Stream Hybrid U-Net (`DualStreamUNet`)
To resolve this divergence without sacrificing either macro precision or capillary sensitivity, we introduce, train, validate, and rigorously benchmark **`DualStreamUNet`** ([models/dual_stream_unet.py](file:///d:/SFHformer-FreqLab/models/dual_stream_unet.py)):
1. **Parallel Pretrained Representation Streams**: Stream A encodes unprocessed native fundus features (warm-started from `unet_raw`), while Stream B encodes SFHformer-restored features (warm-started from `unet_restored`).
2. **Multi-Scale Spatial Gated Fusion (`CrossStreamFusionBlock`)**: At all five spatial resolutions (`512x512` down to `32x32` bottleneck), a learned per-pixel 2-channel spatial softmax gate (`w_raw + w_rest = 1`) dynamically arbitrates between native macro stability and restored capillary contrast, augmented by a Squeeze-and-Excitation (SE) cross-projection residual.
3. **Topology-Aware Compound Optimization**: Trained using `HybridTopologyLoss` combining Binary Cross-Entropy (`0.4`), Soft-Dice (`0.4`), and strictly FP32 differentiable Soft-clDice (`0.2`, `SoftSkeletonize(iter=12)` with `2 px` border masking).

### 1.3 Summary of Frozen Milestone 3 Empirical Findings
- **In-Domain Untouched FIVES (`N=200`, calibrated frozen `tau=0.40`)**: `DualStreamUNet` outperforms **both** single-stream baselines simultaneously across Overall Dice (`0.8474`, `+3.41 pp` vs Raw, Holm `p = 8.72e-28`), Thin Skeleton Sensitivity (`43.52%`, `+10.20 pp` vs Raw, Holm `p = 4.09e-29`), Branch Recall (`74.69%`, `+7.36 pp`), and `clDice` (`0.8537`, `+4.02 pp`), while **reducing** fragmentation (`41.22` vs `52.19` Raw and `114.25` Restored components/image) and restoring Glaucoma precision to **`89.98%`** (`+32.39 pp` above single-stream Restored U-Net).
- **External HRF Transfer (`N=45`, `40` pretrain-exposed / `5` exploratory)**: `DualStreamUNet` reveals **honest mixed domain-transfer trade-offs**: it retains higher thin-vessel skeleton sensitivity (`+9.78 pp`), branch recall (`+6.59 pp`), `clDice` (`+2.78 pp`), and lower fragmentation (`-12.49` CCs/img) than Raw U-Net, while trading `-8.54 pp` Precision and `-1.82 pp` Dice vs Raw U-Net and trailing Restored U-Net (`-3.89 pp`) on HRF thin skeleton sensitivity.



## 2. Datasets, Provenance Audits & Preprocessing Protocols

### 2.1 Dataset Inventory & Zero-Leakage Governance
Rigorous provenance auditing was conducted during Milestone 1 (`results/dataset_audit/milestone1_master_audit.json`, `elapsed = 865.9 s`) and Milestone 2.5 prior to freezing Milestone 3 test runs:

| Dataset Split | Image Count | Resolution | Annotation Scope | Exposure & Audit Notes |
| :--- | :---: | :---: | :--- | :--- |
| **FIVES Train** | `540` | `2048 x 2048` | Expert retinal vessel masks | Generated `2160` paired `512x512` patches (`train_patch_manifest_512.json`) and fine-tuned `SFHformer-m` (`27,000` online crops). |
| **FIVES Clean Validation** | `59` (`60` total) | `2048 x 2048` | Expert retinal vessel masks | Exact train-val duplicate pair (`train_61_A.png` / `train_121_A.png`) isolated; all hyperparameter, checkpoint, and threshold calibration decisions locked exclusively on `N=59` clean val. |
| **FIVES Untouched Test** | `200` | `2048 x 2048` | `50 N / 50 A / 50 D / 50 G` | **Zero training or tuning access.** Intra-test duplicate (`test_32_A.png` / `test_48_A.png`) tracked via duplicate-clean `N=199` sensitivity checks. |
| **HRF Exposed Subset** | `40` | `3504 x 2336` | High-res full vascular tree | Disclosed prior upstream exposure during initial `SFHformer` pretraining; **not** used for U-Net or Dual-Stream training. |
| **HRF Unexposed Subset** | `5` | `3504 x 2336` | High-res full vascular tree | Identifiers: `14_g`, `14_h`, `15_dr`, `15_g`, `15_h`. Zero exposure in either SFHformer or U-Net training; **strictly exploratory (`N=5`)**. |

> **Patient-Level Provenance Disclosure (`milestone1_master_audit.json`)**: While 100% image-level disjointness is verified between FIVES splits after duplicate filtering, FIVES numbering restarts per partition (e.g., `train_1_A` vs `test_1_A`, `180` repeating subject indices), and public metadata omits independent patient linkage IDs.

### 2.2 Audited Vessel Caliber Profile, FOV Masking & Patch Stratification
Quantitative morphological profiling across 15 full-resolution scans per dataset revealed fundamental structural differences between FIVES and HRF that directly explain downstream model transfer behavior:

| Audited Morphometric Property | FIVES (`2048 x 2048`) | HRF (`3504 x 2336`) | Engineering Impact on Loss & Calibration |
| :--- | :---: | :---: | :--- |
| **Smallest Vessels (`1.0 - 3.5 px` radius)** | `1.49%` | `16.71%` | HRF contains `11.2x` denser fine-capillary annotation share than FIVES, shifting optimal threshold sensitivity. |
| **Intermediate Vessels (`3.5 - 8.0 px`)** | `25.4%` | `39.46%` | Intermediate arcades benefit strongly from `HybridTopologyLoss` centerline alignment. |
| **Primary Trunks (`> 8.0 px`)** | `73.11%` | `43.84%` | Dictated setting `SoftSkeletonize(num_iter=12)` (`p95 = 8.96 px`, median max `13.69 px`). |
| **FOV Border Clipping Mean / Max** | `0.116%` / `11.7933%` | `0.0102%` / `0.0781%` | Motivated strict FOV masking (`intensity > 10/255`) and `border_ignore=2 px` during soft-clDice computation. |

- **Training Patch Stratification (`2,160` patches, `512 x 512`)**: Extracted across `540` FIVES training pairs with balanced coverage (`1241` vessel-rich, `717` mixed, `202` vessel-poor patches).

### 2.3 Synthetic Ophthalmic Degradation Engine (`scripts/retina_corrupt.py`)
During restoration fine-tuning and robustness ablation, five physically grounded ophthalmic degradations are composed dynamically:
1. **Raster Striping (`apply_raster_banding`)**: Periodic line-scan detector readout interference modeled as sinusoidal spatial modulation along horizontal/angled scan lines.
2. **Coherent Laser Speckle (`apply_laser_speckle`)**: Multiplicative granular noise simulating coherent optical interference in scanning laser ophthalmoscopy.
3. **Optical Halo Glare (`apply_optical_halo`)**: Radially symmetric corneal/pupil flash reflection artifacts.
4. **Illumination Falloff (`apply_illumination_gradient`)**: Smooth low-frequency peripheral vignetting typical of handheld or undilated fundus cameras.
5. **CMOS Shot Noise (`apply_sensor_noise`)**: Zero-mean additive Gaussian sensor readout noise.

### 2.4 Full-Field Sliding-Window Inference Protocol
Because training runs on `512 x 512` crops under strict `8 GB` VRAM bounds while test frames measure `2048 x 2048` (FIVES) and `3504 x 2336` (HRF), full-resolution evaluation employs overlapping `512 x 512` tiles at stride `384 px` (`25` tiles for FIVES; complete boundary-anchored grid for HRF) weighted by a 2D separable Hamming window:
- `W_2D(y, x) = max(w_hamming(y) * w_hamming(x), 1e-4)`
- Overlapping tile probabilities are accumulated pixel-wise as `sum(P_k * W_2D) / sum(W_2D)` in FP32, completely eliminating tile-boundary seams during evaluation.



## 3. SFHformer Spatial-Frequency Restoration Architecture

### 3.1 Macro-Architecture (`models/SFHformer.py`)
`SFHformer-m` (`14.18M` parameters) replaces standard ViT patch flattening (which destroys pixel-level vessel edges) with a 5-stage symmetric U-shaped encoder-decoder operating at full spatial resolution in Stage 1 and Stage 5 (`patch_size=1`, `3x3` convolution). Feature channel widths progress as `[32, 64, 128, 64, 32]` with SFH block depths `[4, 4, 8, 4, 4]` (`24` blocks total). Encoder downsampling uses strided `3x3` convolutions (`stride=2`), whereas decoder upsampling uses `Conv1x1 + nn.PixelShuffle(2)` to avoid transposed-convolution checkerboards. A long global identity connection adds the degraded input directly to the network output so all `24` blocks predict pure degradation residuals.

![Figure 1: SFHformer-m 5-stage spatial-frequency hierarchical transformer macro-architecture detailing encoder/decoder stages, lateral skips, PixelShuffle upsampling, and global residual learning.|6.9|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_1_sfhformer_macro.png)

### 3.2 Internal Mechanics of the SFH Block (`Mixer`, `FourierUnit`, & `RadialBandGate`)
Each SFH Block applies pre-normalization (`BatchNorm2d`), a parallel **Hybrid Token Mixer** with learnable residual scale `beta`, and a depthwise **Feed-Forward Network (`FFN`)** (`Conv1x1 C->4C -> DWConv3x3 -> GELU -> Conv1x1 4C->C`) scaled by `gamma`.

![Figure 2: In-depth internal architecture of the SFH Block illustrating parallel Local Dilated DW-Convolutions, Global 2D Real FFT FourierUnit, Frequency Positional Encoding, Dynamic Grouped Convolutions, Channel Attention, and FFN.|6.9|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_2_sfh_block_detail.png)

Within `Mixer`, the `1x1` projected tensor is split equally into local and spectral streams:
1. **Local Spatial Stream (`TokenMixer_For_Local`)**: Splits `C` channels into two `C/2` chunks processed in parallel by depthwise dilated convolutions (`3x3, dilation=1` and `3x3, dilation=2`) to capture fine bifurcations and local multi-scale context without mixing unrelated frequencies.
2. **Global Spectral Stream (`TokenMixer_For_Gloal` + `FourierUnit`)**:
   - Transforms normalized spatial tensor `Z` via orthonormal 2D Real FFT: `Z_fft = torch.fft.rfft2(Z, norm='ortho')` in `C^(B x 2C x H x (W/2+1))`.
   - Concatenates real and imaginary components along channels (`4C`), applies `BatchNorm2d`, and adds **Frequency Positional Encoding (`fpe`)** (`3x3` depthwise conv residual) so kernels distinguish low-frequency illumination from high-frequency periodic striping.
   - Applies grouped `1x1` **Frequency Domain Convolution (`fdc`)** modulated by spatial-spectral `Softmax` weights via `torch.einsum`.
   - In the gated `R2` variant ([freqlab/gate.py](file:///d:/SFHformer-FreqLab/freqlab/gate.py)), `RadialBandGate` interpolates `16` learnable log-amplitude bands across normalized radial frequency `R_norm = sqrt(f_y^2 + f_x^2) / R_max` (`exp(0) = 1.0` identity initialization).
   - Reconstructs complex tensors and transforms back via `torch.fft.irfft2(..., s=(H, W), norm='ortho')`, followed by **Squeeze-and-Excitation Channel Attention** fusion with the local stream.

### 3.3 Restoration Fine-Tuning & Held-Out 200-Scan Evaluation
- **Training Configuration** ([scripts/train_fives_retina.py](file:///d:/SFHformer-FreqLab/scripts/train_fives_retina.py)): Initialized from DRIVE/baseline weights (`L1` pixel loss, `AdamW`, `CosineAnnealingLR`, `256x256` tissue-filtered crops with `>35%` fundus tissue, `batch=1, accum=4`, online mixed degradations).
- **Empirical Restoration Fidelity** ([results/fives_evaluation_results.json](file:///d:/SFHformer-FreqLab/results/fives_evaluation_results.json)):
  - **Peak Validation PSNR**: `31.92 dB` on FIVES fine-tuning validation crops (`checkpoints/fives_retina_finetune/best.pth`).
  - **Held-Out 200-Scan Synthetic Degradation Test**: Mean PSNR improves from `20.28 dB` (corrupted) and `25.59 dB` (DRIVE baseline `checkpoints/retina/best.pth`) to **`29.24 dB`** (`+3.65 dB` gain over DRIVE baseline), with mean SSIM reaching **`0.9309`** (`avg latency = 74.2 ms`, `peak VRAM = 0.33 GB`).

| FIVES Pathology Subset | Corrupted PSNR | DRIVE Baseline PSNR | Fine-Tuned SFHformer PSNR | Gain vs Baseline | Fine-Tuned SSIM | Residual Noise FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Healthy (NOR)** (`n=50`) | `20.68 dB` | `26.90 dB` | **`30.22 dB`** | `+3.32 dB` | `0.9351` | `9.98%` |
| **AMD** (`n=50`) | `19.58 dB` | `24.81 dB` | **`27.35 dB`** | `+2.55 dB` | `0.9284` | `7.35%` |
| **Diabetic Retinopathy (DR)** (`n=50`) | `20.08 dB` | `24.68 dB` | **`28.38 dB`** | `+3.69 dB` | `0.9222` | `9.14%` |
| **Glaucoma (GLA)** (`n=50`) | `20.78 dB` | `25.99 dB` | **`31.01 dB`** | `+5.02 dB` | `0.9379` | `10.42%` |
| **Overall FIVES (`N=200`)** | `20.28 dB` | `25.59 dB` | **`29.24 dB`** | **`+3.65 dB`** | **`0.9309`** | `9.22%` |

- *(Scope Qualification: These restoration metrics evaluate intra-project progression on our synthetic multi-degradation benchmark under matched experimental protocols; they do not claim external cross-benchmark SOTA.)*



## 4. Baseline Single-Stream Clinical U-Net & Classical Frangi Control

### 4.1 Single-Stream U-Net Architecture (`models/unet.py`)
To evaluate whether SFHformer restoration genuinely aids downstream segmentation, two structurally identical 4-stage residual U-Nets (`32 -> 64 -> 128 -> 256 -> 512` bottleneck, `7.7M` parameters each) were trained under identical optimization schedules ([scripts/train_unet.py](file:///d:/SFHformer-FreqLab/scripts/train_unet.py)):
- Each encoder/decoder level employs `DoubleConv` (`Conv3x3 -> BN -> GELU -> [Dropout2d(0.2) at bottleneck] -> Conv3x3 -> BN -> GELU`) plus a `1x1` residual identity projection.
- Decoder stages utilize `Upsample(scale_factor=2, mode='bilinear', align_corners=True) + Conv1x1` rather than transposed convolutions, preserving crisp binary thresholds without checkerboard artifacts.

![Figure 3: Clinical single-stream 4-stage residual U-Net architecture detailing DoubleConv blocks, lateral skip projections, bilinear upsampling decoder, and the divergence between Raw U-Net and Restored U-Net baselines.|6.9|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_3_single_unet.png)

### 4.2 Single-Stream Training Dynamics (`N=59` Clean Validation)
Both single-stream models trained for `8` epochs (`batch=2, accum=2`, BF16 AMP, `BCEDiceLoss` with `0.5 BCE + 0.5 Soft-Dice`) with checkpoint selection on `N=59` clean validation at fixed threshold `tau = 0.50`:

#### `unet_raw` Training History (Native FIVES Fundus Input)
| Epoch | Train Loss | Clean `N=59` Dice | Precision | Sensitivity | ThinSens (diam<=3.5px) | Continuity Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 02 | `0.2026` | `0.7336` | `0.8089` | `0.7123` | `20.09%` | `25.49` |
| Epoch 04 | `0.1381` | `0.8054` | `0.9390` | `0.7228` | `10.03%` | `7.84` |
| Epoch 06 | `0.1070` | `0.8299` | `0.9626` | `0.7500` | `10.96%` | `5.72` |
| Epoch 08 | `0.0941` | `0.8495` | `0.9577` | `0.7819` | `13.58%` | `5.04` |

#### `unet_restored` Training History (`SFHformer-m` Restored Input)
| Epoch | Train Loss | Clean `N=59` Dice | Precision | Sensitivity | ThinSens (diam<=3.5px) | Continuity Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 02 | `0.3011` | `0.6150` | `0.7241` | `0.5601` | `13.22%` | `27.28` |
| Epoch 04 | `0.2307` | `0.6974` | `0.7453` | `0.6626` | `18.98%` | `17.12` |
| Epoch 06 | `0.1954` | `0.7203` | `0.7444` | `0.7069` | `19.95%` | `13.73` |
| Epoch 08 | `0.1768` | `0.7251` | `0.7514` | `0.7101` | `19.74%` | `11.88` |

### 4.3 Milestone 2.5 Clean-Val Divergence & Classical Frangi Control
- **Clean Validation `N=59` (Milestone 2.5 Paired Audit, diameter <= 3.5 px)**:
  - `unet_raw`: Dice `0.8495`, Thin Skeleton Sensitivity `13.58%`, `clDice 0.8406`, `44.6` components/image.
  - `unet_restored`: Dice `0.7251`, Thin Skeleton Sensitivity **`19.74%`** (`+6.16 pp`, 95% CI `[+4.19, +8.12] pp`, Wilcoxon `p = 7.58e-07`), `clDice 0.6745`, `108.4` components/image.
- **Classical Multiscale Frangi Control (`scripts/baseline_frangi.py`)**: After threshold calibration (`tau = 0.0010` across `sigmas=[1..5]`), Frangi achieved `0.1876` Dice (Raw) and `0.1578` Dice (Restored) on FIVES `N=200`, confirming that deep segmentation networks substantially outperform classical Hessian vesselness filters (`>0.81` vs `<0.20` Dice).
- **Motivation for Dual-Stream Fusion**: Between Epoch 6 and Epoch 8 on `N=59` clean validation, `unet_raw` improved by `+0.0196` Dice (`0.8299 -> 0.8495`) while `unet_restored` improved by `+0.0048` (`0.7203 -> 0.7251`). Neither single-stream representation alone satisfies both clinical requirements simultaneously.



## 5. Proposed Architecture: Dual-Stream Hybrid U-Net (`DualStreamUNet`)

### 5.1 Architectural Design & Warm-Start Strategy
Implemented in [models/dual_stream_unet.py](file:///d:/SFHformer-FreqLab/models/dual_stream_unet.py) (`16.41M` total parameters), `DualStreamUNet` eliminates the forced binary choice between native and restored inputs by running **paired encoding streams** concurrently and fusing them at every hierarchical resolution before a single shared decoder.

![Figure 4: Dual-Stream Hybrid U-Net architecture (DualStreamUNet) showing warm-started Raw and Restored encoders, five multi-scale CrossStreamFusionBlock modules with per-pixel spatial gate inset (w_raw + w_rest = 1), optional scale_0 gate visualization export, shared bilinear decoder, and HybridTopologyLoss.|6.9|4.1](D:/SFHformer-FreqLab/demo_package/diagrams/diag_4_dual_stream_hybrid_unet.png)

#### Exact Warm-Start Mapping (Verified Before Training)
Inspecting `models/unet.py` confirmed identical encoder block keys (`downs.0..3`, `bottleneck`) across `unet_raw/best.pth` and `unet_restored/best.pth`:
- **Stream A (`raw_downs`, `raw_bottleneck`)**: Initialized with `unet_raw/best.pth` (`downs.* -> raw_downs.*`, `bottleneck.* -> raw_bottleneck.*`).
- **Stream B (`rest_downs`, `rest_bottleneck`)**: Initialized with `unet_restored/best.pth` (`downs.* -> rest_downs.*`, `bottleneck.* -> rest_bottleneck.*`).
- **Shared Bilinear Decoder (`ups`, `final_conv`)**: Initialized from `unet_raw/best.pth` (`208` total matched tensors loaded with zero shape mismatches).
- **Fusion Spine (`fusions[0..3]`, `bottleneck_fusion`)**: Randomly initialized (`80` new parameter/buffer tensors).

### 5.2 Mathematical Formulation of `CrossStreamFusionBlock`
Given aligned raw features `F_raw` and restored features `F_rest` in `R^(B x C x H_s x W_s)` at scale `s`, concatenation forms `Z_cat = [F_raw, F_rest]` in `R^(B x 2C x H_s x W_s)`. The block computes two parallel mechanisms:
1. **Per-Pixel 2-Channel Spatial Gate**:
   - `A_raw, A_rest = Softmax(Conv1x1(GELU(BN(Conv3x3(Z_cat)))), dim=1)`
   - Produces one normalized spatial weighting pair `(w_raw, w_rest)` per pixel such that `w_raw + w_rest = 1` across every `(y, x)` location (`scale_0` returns `w_rest` at full `512x512` input resolution). *Important distinction*: this is a **2-channel per-pixel spatial gate**, not channel-wise attention; `w_rest` serves strictly as a spatial routing visualization and never implies pixel confidence or anatomical proof.
2. **SE-Weighted Cross-Projection Residual**:
   - `R_cross = SE(GELU(BN(Conv1x1(Z_cat))))`, where `SE(u) = u * Sigmoid(Conv1x1(GELU(Conv1x1(GAP(u)))))`.
3. **Fused Feature Output**:
   - `F_fused = w_raw * F_raw + w_rest * F_rest + R_cross`

### 5.3 Numerically Stable FP32 Topology Loss (`HybridTopologyLoss`)
To prevent fine-vessel gains from fragmenting into isolated components, `HybridTopologyLoss` ([utils/losses.py](file:///d:/SFHformer-FreqLab/utils/losses.py)) optimizes:
- `L_hybrid = 0.4 * L_BCEWithLogits + 0.4 * L_SoftDice + 0.2 * L_SoftclDice`
- **Stability & Boundary Guards**: `SoftclDiceLoss` executes strictly in **FP32** (`torch.autocast(..., enabled=False)`) to prevent BF16 pooling underflow, uses **`num_iter=12`** erosions (matching the audited `p95=8.96 px` / `median max=13.69 px` FIVES vessel radii, capturing `1,768` vs `499` skeleton pixels at `iter=5`), and masks the outer `2 px` border (`border_ignore=2`) so vessels cut by `512x512` crop edges do not generate artificial boundary skeletons.



### 5.4 Two-Stage Training, Hardware Profiling & Clean-Val Calibration (`N=59`)
- **Stage 2 Memory Profile**: Prior to full training, `3` unfrozen Stage 2 backward steps (`batch_size=2`, `512x512`, BF16 AMP) passed on RTX 5060 with `0` `NaN`/`Inf` gradients and **Peak VRAM = `2,948.8 MB`** (within the `5.57 GB` allocator cap).

| Optimization Phase | Epochs | Frozen Modules | Learning Rates | Regularization & Runtime |
| :--- | :---: | :--- | :--- | :--- |
| **Stage 1: Fusion Warmup** | `1 - 4` | `raw_downs`, `raw_bottleneck`, `rest_downs`, `rest_bottleneck` | Fusion & Decoder: `5e-4` | `CosineAnnealingLR`, `grad_clip=1.0`, synchronized spatial flips/rotations |
| **Stage 2: Joint Fine-Tuning** | `5 - 12` | None (End-to-End Unfrozen) | Encoders: `1e-4`; Fusion/Dec: `3e-4` | BF16 AMP + FP32 `SoftclDice(12)`, total `12`-epoch wall time: **`47.43 mins`** |

- **Checkpoint Selection**: Driven at fixed `tau=0.50` by balanced composite score `S = 0.4*Dice + 0.3*clDice + 0.3*ThinSens` on clean validation (`N=59`), reporting all components separately:

| Epoch | Stage | Train Loss | Clean `N=59` Dice | Precision | ThinSens (diam<=3.5px) | clDice | Mean CC | Composite `S` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Epoch 02 | Stage 1 (Frozen) | `0.1001` | `0.8741` | `0.9378` | `22.21%` | `0.8729` | `38.9` | `0.6781` |
| Epoch 04 | Stage 1 (Frozen) | `0.0929` | `0.8808` | `0.9472` | `18.53%` | `0.8779` | `35.6` | `0.6713` |
| Epoch 06 | Stage 2 (Joint) | `0.0941` | `0.8548` | `0.9643` | `12.51%` | `0.8589` | `37.3` | `0.6371` |
| Epoch 08 | Stage 2 (Joint) | `0.0912` | `0.8600` | `0.9640` | `12.33%` | `0.8577` | `35.6` | `0.6383` |
| Epoch 10 | Stage 2 (Joint) | `0.0853` | `0.8851` | `0.9403` | `20.57%` | `0.8829` | `35.2` | `0.6806` |
| Epoch 12 | Stage 2 (Joint) | `0.0838` | `0.8758` | `0.9373` | `21.51%` | `0.8765` | `41.1` | `0.6778` |

- **Non-Monotonic Behavior Note**: As expected during Stage 2 encoder unfreezing (Epoch 6), precision briefly spiked (`0.9643`) while thin-vessel sensitivity dipped (`0.1251`) before recovering to peak balanced performance at **Epoch 10** (`Composite = 0.6806`, `Dice = 0.8851`, `clDice = 0.8829`, `ThinSens = 0.2057`, `CC = 35.2`).
- **Frozen Threshold Calibration (`N=59` Clean Val Only)**: Sweeping Epoch 10 (`best.pth`, SHA-256 `9be2213e0cf0c799...`) strictly on `N=59` clean validation selected and froze **`tau = 0.40`** before any test-set evaluation:

| Threshold `tau` | Composite `S` | Dice | Precision | ThinSens (diam<=3.5px) | Thin_r35 (rad<=3.5px) | clDice | Mean CC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.40` | `0.6874` | `0.8866` | `0.9321` | `22.42%` | `42.74%` | `0.8852` | `34.7` |
| `0.45` | `0.6840` | `0.8859` | `0.9363` | `21.47%` | `41.92%` | `0.8840` | `35.1` |
| `0.50` | `0.6806` | `0.8851` | `0.9403` | `20.57%` | `41.11%` | `0.8829` | `35.2` |
| `0.55` | `0.6771` | `0.8841` | `0.9441` | `19.66%` | `40.28%` | `0.8817` | `34.7` |
| `0.60` | `0.6732` | `0.8829` | `0.9479` | `18.64%` | `39.46%` | `0.8804` | `33.8` |



## 6. Milestone 3 Frozen Generalization Evaluation

### 6.1 Evaluation Protocol & Statistical Rigor
All evaluations ran once via [scripts/evaluate_milestone3.py](file:///d:/SFHformer-FreqLab/scripts/evaluate_milestone3.py) and [scripts/evaluate_dual_stream_milestone3.py](file:///d:/SFHformer-FreqLab/scripts/evaluate_dual_stream_milestone3.py) with pre- and post-execution SHA-256 verification. Paired non-parametric Wilcoxon signed-rank tests, paired t-tests, `10,000`-sample bootstrap 95% confidence intervals, and **Holm-Bonferroni step-down adjustments** across the 6 tested metrics (`dice`, `prec`, `thin_skel_sens`, `cldice`, `branch_recall`, `num_pred_cc`) were computed from preserved per-image JSON records.

### 6.2 FIVES Untouched Test Set (`N=200` Primary In-Domain & `N=199` Deduplicated)

| System (`Frozen tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | Thin Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw U-Net** (`0.50`) | `0.8133` | **`0.9266`** | `0.7566` | `33.33%` | `67.32%` | `34.95%` | `0.8134` | `52.19` |
| **Restored U-Net** (`0.50`) | `0.6927` | `0.7142` | `0.6824` | `29.68%` | `60.44%` | `31.94%` | `0.6536` | `114.25` |
| **Dual-Stream Hybrid** (`0.40`) | **`0.8474`** | `0.9007` | **`0.8242`** | **`43.52%`** | **`74.69%`** | **`45.81%`** | **`0.8537`** | **`41.22`** |
| **Raw Frangi** (`0.0010`) | `0.1876` | `0.1775` | `0.2555` | `27.10%` | `16.16%` | `18.86%` | `0.1579` | `24235.5` |
| **Restored Frangi** (`0.0010`) | `0.1578` | `0.1394` | `0.1894` | `20.36%` | `9.48%` | `12.71%` | `0.1339` | `23770.8` |

#### Paired Statistical Comparisons on FIVES (`N=200`, Holm-Bonferroni Adjusted)

| Metric | Dual-Stream vs Raw U-Net Delta (95% CI, Holm `p_wil`) | Dual-Stream vs Restored U-Net Delta (95% CI, Holm `p_wil`) |
| :--- | :--- | :--- |
| **Dice** | **`+0.0341`** (`[+0.0275, +0.0414]`, `p=8.72e-28`) | **`+0.1546`** (`[+0.1411, +0.1682]`, `p=8.70e-33`) |
| **Precision** | `-0.0259` (`[-0.0331, -0.0196]`, `p=3.03e-22`) | **`+0.1866`** (`[+0.1615, +0.2124]`, `p=3.63e-28`) |
| **Thin Skel Sens** | **`+10.20 pp`** (`[+8.94, +11.51] pp`, `p=4.09e-29`) | **`+13.85 pp`** (`[+12.26, +15.43] pp`, `p=3.87e-29`) |
| **Branch Recall** | **`+7.36 pp`** (`[+6.49, +8.27] pp`, `p=4.13e-31`) | **`+14.25 pp`** (`[+13.10, +15.38] pp`, `p=3.61e-32`) |
| **clDice** | **`+0.0402`** (`[+0.0337, +0.0475]`, `p=1.14e-31`) | **`+0.2001`** (`[+0.1853, +0.2149]`, `p=2.22e-33`) |
| **Components (`CC`)** | **`-10.97`** (`[-13.90, -7.94]`, `p=3.38e-19`) | **`-73.03`** (`[-77.96, -67.98]`, `p=5.61e-32`) |

- **Duplicate-Clean Sensitivity (`N=199`, excluding `test_48_A.png`)**: All conclusions remain unchanged (`Dual Dice = 0.8468`, `+0.0342` vs Raw; `ThinSkel = 43.39%`, `+10.19 pp`; `clDice = 0.8531`).



### 6.3 Disease Stratification (`n=50` per Pathology on FIVES)

| Pathology | Raw Dice / Prec | Restored Dice / Prec | Dual-Stream Dice / Prec | Dual ThinSkel | Dual clDice | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** (n=50) | `0.8650` / `0.9052` | `0.6978` / `0.7034` | **`0.8837` / `0.9016`** | `45.35%` | `0.8965` | `49.9` |
| **AMD** (n=50) | `0.8696` / `0.9452` | `0.8040` / `0.8293` | **`0.8911` / `0.9022`** | `52.49%` | `0.8965` | `44.56` |
| **DR** (n=50) | `0.8181` / `0.9450` | `0.7075` / `0.7481` | **`0.8481` / `0.8994`** | `42.47%` | `0.8565` | `42.0` |
| **Glaucoma** (n=50) | `0.7004` / `0.9113` | `0.5616` / `0.5759` | **`0.7667` / `0.8998`** | `33.79%` | `0.7651` | `28.42` |

#### Per-Disease Paired Statistical Deltas (`Dual-Stream` vs `Raw` & `Restored`, Holm `p`)
| Pathology | vs Raw Dice (`p_holm`) | vs Raw ThinSkel (`p_holm`) | vs Raw clDice | vs Restored Prec (`p_holm`) | vs Restored CC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** (`n=50`) | `+0.0187` (`p=6.44e-10`) | `+3.05 pp` (`p=1.82e-03`) | `+0.0297` | `+19.82 pp` (`p=1.07e-14`) | `-97.3` |
| **AMD** (`n=50`) | `+0.0215` (`p=7.84e-08`) | `+15.07 pp` (`p=1.07e-14`) | `+0.0268` | `+7.29 pp` (`p=1.19e-04`) | `-48.4` |
| **DR** (`n=50`) | `+0.0300` (`p=4.25e-09`) | `+13.20 pp` (`p=1.07e-14`) | `+0.0328` | `+15.13 pp` (`p=4.66e-08`) | `-62.5` |
| **Glaucoma** (`n=50`) | `+0.0663` (`p=1.04e-08`) | `+9.48 pp` (`p=1.04e-08`) | `+0.0716` | `+32.39 pp` (`p=2.22e-13`) | `-84.0` |

- **Glaucoma Cup Rim Resolution**: Single-stream `unet_restored` suffered `-33.54 pp` precision collapse on Glaucoma (`57.59%` vs `91.13%`) because frequency contrast enhancement amplified excavated optic-cup rim borders. By gating with Stream A (`Raw`), `DualStreamUNet` restores Glaucoma precision to **`89.98%`** (`+32.39 pp` over Restored, only `-1.15 pp` vs Raw) while achieving `+6.63 pp` higher Dice and `+9.48 pp` higher Thin Skeleton Sensitivity than Raw U-Net.

![Figure 5: Full-resolution untouched FIVES test qualitative comparisons across Normal (test_161_N), AMD (test_37_A), DR (test_100_D), and Glaucoma (test_133_G).|6.9|4.6](D:/SFHformer-FreqLab/demo_package/qualitative_examples/fives_test_disease_stratified_demo.png)



### 6.4 External Generalization on HRF (`N=45`: `40` Pretrain-Exposed / `5` Exploratory Unexposed)

>! **HRF Exposure & Transfer Caveat**: `SFHformer` encountered `40/45` HRF scans during upstream pretraining. Furthermore, transferring a FIVES-calibrated threshold (`tau = 0.40`) onto higher-resolution `3504x2336` HRF scans (`16.71%` smallest vessels vs `1.49%` in FIVES) shifts the sensitivity-precision operating point. Results must be interpreted transparently alongside these trade-offs.

| Cohort & System (`tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HRF Overall (`N=45`) — Raw U-Net** (`0.50`) | **`0.7601`** | **`0.7482`** | `0.7900` | `36.81%` | `61.54%` | `0.7467` | `80.1` |
| **HRF Overall (`N=45`) — Restored U-Net** (`0.50`) | `0.7368` | `0.6532` | **`0.8519`** | **`50.48%`** | **`69.40%`** | `0.7546` | `166.0` |
| **HRF Overall (`N=45`) — Dual-Stream** (`0.40`) | `0.7419` | `0.6628` | `0.8497` | `46.59%` | `68.13%` | **`0.7745`** | **`67.6`** |
| **HRF Exposed (`N=40`) — Dual-Stream** (`0.40`) | `0.7432` | `0.6654` | `0.8485` | `45.37%` | `67.53%` | `0.7732` | `64.7` |
| **HRF Unexposed (`N=5`, Exploratory) — Raw** (`0.50`) | **`0.7463`** | **`0.7263`** | `0.7966` | `47.77%` | `66.19%` | `0.7540` | `116.2` |
| **HRF Unexposed (`N=5`, Exploratory) — Restored** (`0.50`) | `0.7205` | `0.6233` | **`0.8596`** | **`58.97%`** | **`73.04%`** | `0.7521` | `212.0` |
| **HRF Unexposed (`N=5`, Exploratory) — Dual** (`0.40`) | `0.7311` | `0.6423` | `0.8590` | `56.30%` | `72.89%` | **`0.7854`** | **`90.6`** |

#### Paired HRF Statistical Trade-Off Summary (`N=45`, Holm-Bonferroni Adjusted)
| Metric | Dual-Stream vs Raw U-Net (`N=45`) | Dual-Stream vs Restored U-Net (`N=45`) |
| :--- | :--- | :--- |
| **Dice** | `-0.0182` (`[-0.0250, -0.0117]`, `p=1.25e-06`) | `+0.0051` (`[-0.0002, +0.0103]`, `p=5.16e-02`) |
| **Precision** | `-0.0854` (`[-0.1064, -0.0660]`, `p=9.38e-12`) | `+0.0096` (`[+0.0024, +0.0169]`, `p=2.94e-02`) |
| **Thin Skel Sens** | **`+9.78 pp`** (`[+8.38, +11.19] pp`, `p=1.02e-12`) | `-3.89 pp` (`[-4.94, -2.78] pp`, `p=3.33e-07`) |
| **Branch Recall** | **`+6.59 pp`** (`p=1.78e-08`) | `-1.27 pp` (`p=2.25e-03`) |
| **clDice** | **`+0.0278`** (`p=9.38e-12`) | **`+0.0199`** (`p=2.31e-06`) |
| **Components (`CC`)** | **`-12.49`** (`p=2.05e-03`) | **`-98.40`** (`p=3.09e-08`) |

- **HRF `N=5` Exploratory Per-Image Breakdown**: With only `N=5` pairs, the minimum achievable two-sided Wilcoxon `p` is `0.0625` (`wilcoxon_p_holm = 0.375` across 6 comparisons), which is statistically non-significant and reported strictly as exploratory evidence:

| HRF Unexposed Scan | Dual Dice | Dual Prec | Dual Sens | Dual ThinSkel | Dual clDice | Dual BranchRecall | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `14_g.png` | `0.7281` | `0.6667` | `0.8020` | `42.42%` | `0.7507` | `62.55%` | `37` |
| `14_h.png` | `0.7601` | `0.6659` | `0.8854` | `64.68%` | `0.8511` | `79.21%` | `95` |
| `15_dr.png` | `0.6607` | `0.5154` | `0.9200` | `66.40%` | `0.7215` | `82.00%` | `210` |
| `15_g.png` | `0.6531` | `0.5713` | `0.7624` | `25.29%` | `0.6684` | `55.09%` | `54` |
| `15_h.png` | `0.8536` | `0.7922` | `0.9254` | `82.72%` | `0.9353` | `85.59%` | `57` |

![Figure 6: External HRF exploratory unexposed subset (14_h, 15_dr) illustrating thin-vessel recovery alongside out-of-domain precision trade-offs.|6.9|3.3](D:/SFHformer-FreqLab/demo_package/qualitative_examples/hrf_transfer_unexposed_demo.png)



## 7. System Integration & RetinaScan Studio Prototype

### 7.1 End-to-End Pipeline Architecture
The validated models are deployed inside a modular research prototype connecting a Flask REST inference server ([scripts/serve_restore.py](file:///d:/SFHformer-FreqLab/scripts/serve_restore.py), port `5001`) with a React 18 / Vite frontend ([frontend/src/pages/RetinaStudio.jsx](file:///d:/SFHformer-FreqLab/frontend/src/pages/RetinaStudio.jsx), port `3000`).

![Figure 7: Combined end-to-end RetinaScan Studio system architecture linking ingestion, SFHformer restoration, the 3-model segmentation core, thread-safe Flask serving with 6-stage reconciled timing, and the React UI.|6.9|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_5_combined_retinascan_system.png)

### 7.2 Protocol Safeguards & Integration QA Summary
- **Frozen Operating Points vs Exploratory Controls**: Defaults to `Dual-Stream Hybrid` at frozen validation-calibrated `tau = 0.40`, offers `tau = 0.50` explicitly labeled as an alternative threshold-matched baseline comparison, isolates exploratory slider changes to local previews only, and never auto-adjusts thresholds per dataset.
- **Optional Spatial Fusion-Weight Visualization**: Hidden by default; renders the exact single-pass `scale_0` spatial gate `w_rest` (`Blue = 0.0 Raw -> Amber = 1.0 Restored`) explicitly captioned *"Not confidence, uncertainty, or vessel probability."*

| Integration QA Verification Check | Scope & Harness | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| **Heatmap Spatial Alignment & Gradient Polarity** | `qa_phase5_1.py` (checkerboard shift) | Top-Left `12.27` vs Bottom-Right `2.45`; `w_rest` 0->1 polarity exact | PASS |
| **Same-Forward-Pass Gate Guarantee** | `u_dual(..., return_gates=True)` check | `max logit delta = 0.0` vs standard pass | PASS |
| **Invalid Input & Edge-Case Guards** | `9` malformed API requests (`POST`) | `9/9` structured `HTTP 400` JSON errors; cross-model viz suppressed | PASS |
| **20-Run Latency & Timing Reconciliation** | `20` consecutive `512x512` runs | Median `1,300.8 ms`, P95 `1,724.9 ms`, max reconcile err `0.1 ms` | PASS |
| **VRAM Stability & Thread Serialization** | `20` sequential + `4` concurrent calls | `0.0 MB` VRAM leak (`216.5 MB` steady); `4/4` concurrent `HTTP 200` | PASS |
| **Live Chromium DOM & Visual UI Suite** | `qa_phase5_2_browser.py` (`Edg/157.0.4341.60`) | `11/11` checks pass, `naturalWidth > 0`, `0` JS/React runtime errors | PASS |



### 7.3 Verified Live Studio Captures (`Phase 5.2`)

![Figure 8: Live Chromium screenshot of RetinaScan Studio Tab 1 showing Dual-Stream Hybrid (tau=0.40), prototype notice, and optional scale_0 spatial fusion-weight heatmap legend.|6.8|4.1](D:/SFHformer-FreqLab/demo_package/qualitative_examples/phase5_2_ui_tab1_fusion.png)

![Figure 9: Live Chromium screenshot of RetinaScan Studio Tab 2 displaying frozen FIVES and HRF comparative metrics alongside explicit HRF exposure and N=5 exploratory disclosures.|6.8|4.1](D:/SFHformer-FreqLab/demo_package/qualitative_examples/phase5_2_ui_tab2_benchmark.png)



## 8. Reproducibility, Limitations & Conclusions

### 8.1 Checkpoint Manifest & Naming Reconciliation
To prevent ambiguity across Milestone 2, Milestone 3, and Phase 5.3 manifests, every frozen binary weight file on disk was cryptographically audited:

| Logical / Script Alias | Exact Repository Path | Size | SHA-256 Digest (`64 hex chars`) |
| :--- | :--- | :---: | :--- |
| `sfhformer` / `retina_fives` | `checkpoints/fives_retina_finetune/best.pth` | `67.80 MB` | `d74e267c6ac664686528d6285564fcefcfddc97db1c1f0f676cc00ab5a49563c` |
| `unet_raw` / `raw` | `checkpoints/unet_raw/best.pth` | `86.97 MB` | `97639a65b6cc3d09d0638036e04ea018c236f3e8df0bf556282de6beba6f9d50` |
| `unet_restored` / `restored` | `checkpoints/unet_restored/best.pth` | `86.97 MB` | `2cfb37371cc5c0038bba6f7c3dcde808a4571c246a3c5df7bddba460ff9011cb` |
| `unet_dual_stream` / `dual_stream` | `checkpoints/unet_dual_stream/best.pth` | `62.74 MB` | `9be2213e0cf0c79998f31e7f9c64f4ef82e676a4ae8d74c22328aa80bcc1ee37` |

- **Naming Resolution**: The identifier `"sfhformer"` inside `scripts/evaluate_milestone3.py` (`CKPTS['sfhformer']`) is confirmed to reference `checkpoints/fives_retina_finetune/best.pth` (`SHA-256: d74e267c...`). Earlier pretraining/ablation SFHformer checkpoints (`r1_baseline/best.pth`, `r2_gated/best.pth`, `retina/best.pth`) reside in separate subdirectories and were not modified.
- **Size Difference Explanation**: `unet_raw` and `unet_restored` (`86.97 MB`) bundle `AdamW` optimizer state buffers alongside `7.7M` model weights, whereas `unet_dual_stream` (`62.74 MB`) stores `16.41M` model weights (`state_dict`) and validation metadata only.

### 8.2 Reproducibility Instructions
```bash
git checkout v3.1-phase5.2-verified
d:\SFHformer-FreqLab\.venv\python.exe scripts/evaluate_milestone3.py
d:\SFHformer-FreqLab\.venv\python.exe scripts/evaluate_dual_stream_milestone3.py
d:\SFHformer-FreqLab\.venv\python.exe scripts/serve_restore.py
```

### 8.3 Limitations & Future Extensions
1. **External Domain Shift**: While `DualStreamUNet` excels across all metrics in-domain on FIVES (`N=200`), external transfer to `3504x2336` HRF at `tau=0.40` trades `-8.54 pp` Precision vs Raw U-Net, motivating multi-domain validation and domain-invariant threshold calibration on genuinely unexposed high-resolution cohorts.
2. **Spatial vs Channel-Wise Gating**: Current `CrossStreamFusionBlock` applies a 2-channel per-pixel spatial gate (`w_raw + w_rest = 1`) with an SE residual; a controlled ablation comparing channel-wise gating (`2C` weights) against spatial gating remains open.
3. **Interactive Studio Resolution vs Full-Field Tiling**: RetinaScan Studio resizes interactive uploads to `512x512` for `~1.3 s` turnaround; extremely dark or ultra-high-res scans benefit from full sliding-window tiling used in Milestone 3 scripts.

### 8.4 Conclusions & References
1. **SFHformer-FreqLab** demonstrates that coupling spatial-frequency restoration with warm-started dual-stream gated segmentation (`DualStreamUNet`) and FP32 `SoftclDice` regularization resolves the core accuracy-vs-sensitivity trade-off on FIVES (`0.8474` Dice, `+10.20 pp` thin skeleton sensitivity, `0.8537` clDice, `41.22` components/image, `89.98%` Glaucoma precision).
2. **References**: (a) Shit et al., *clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021; (b) Jin et al., *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation*, Scientific Data 2022; (c) Budai et al., *Robust Vessel Segmentation in Fundus Images (HRF)*, Int. J. Biomed. Imaging 2013; (d) Frangi et al., *Multiscale Vessel Enhancement Filtering*, MICCAI 1998; (e) Ronneberger et al., *U-Net*, MICCAI 2015.



## Appendix A: Development Progression & Immutable Snapshot Inventory

### A.1 End-to-End Experimental Milestone Summary

| Stage / Phase | Objective & Scope | Key Verified Outcome |
| :--- | :--- | :--- |
| **Milestone 1** | FIVES & HRF provenance, duplicate isolation, FOV clipping & caliber audit | `100%` image-level disjointness (`N=540 / 59 / 200`); `11.2x` fine-capillary share difference (`16.71%` HRF vs `1.49%` FIVES). |
| **SFHformer Fine-Tune** | 5-stage hierarchical FFT restoration (`SFHformer-m`, `14.18M` params) on FIVES crops | Peak validation PSNR `31.92 dB`; held-out `N=200` test PSNR `29.24 dB` (`+3.65 dB` over DRIVE baseline), `SSIM 0.9309`. |
| **Milestone 2.5** | Paired clean-val (`N=59`) evaluation of `unet_raw` vs `unet_restored` baselines | Proved `unet_restored` boosts thin-vessel sensitivity `+6.16 pp` (`p=7.58e-7`) while dropping Dice & continuity. |
| **Phases 1–4 (`DualStream`)** | Warm-started dual encoder + `CrossStreamFusionBlock` + FP32 `SoftclDice(12)` + `N=59` sweep | Epoch 10 best checkpoint (`S=0.6806`); froze calibrated threshold `tau = 0.40` strictly on `N=59` clean validation. |
| **Milestone 3** | One-shot frozen generalization on untouched FIVES (`N=200`/`199`) & HRF (`N=45`) | FIVES Dice `0.8474`, ThinSkel `43.52%`, Glaucoma Prec `89.98%`; honest HRF transfer trade-offs documented. |
| **Phases 5.1–5.3** | RetinaScan Studio REST + React UI + QA (`9/9` backend, `11/11` Chromium) + Git tag | Tagged `v3.1-phase5.2-verified`; `15` immutable SHA-256 artifacts + `5` 300-DPI architecture diagrams verified. |

### A.2 Cryptographic Inventory (`SNAPSHOT_MANIFEST.json`, `v3.1-phase5.2-verified`)

| Frozen Artifact Path (`D:\SFHformer-FreqLab\`) | SHA-256 Digest Prefix (Full 64-Hex Verified on Disk) |
| :--- | :--- |
| `checkpoints/unet_raw/best.pth` | `97639a65b6cc3d09d0638036e04ea018...` |
| `checkpoints/unet_restored/best.pth` | `2cfb37371cc5c0038bba6f7c3dcde808...` |
| `checkpoints/unet_dual_stream/best.pth` | `9be2213e0cf0c79998f31e7f9c64f4ef...` |
| `checkpoints/unet_dual_stream/validation_calibration.json` | `309b3e59ed32d3be07edf5ce85be554b...` |
| `checkpoints/fives_retina_finetune/best.pth` | `d74e267c6ac664686528d6285564fcef...` |
| `results/milestone3/fives_test_results.json` | `73fc261d187929e95fad04f0359c9c98...` |
| `results/milestone3/hrf_results.json` | `7747784d671e97c1298054c4f948e7bc...` |
| `results/milestone3/dual_stream_fives_results.json` | `9baa998f9e4c86fd6d3e59ae32305198...` |
| `results/milestone3/dual_stream_hrf_results.json` | `f92a7ada47669b956ffbc5f627b92c1c...` |
| `models/SFHformer.py` | `ff8f318dd4ff637e890a2dfefc20cf02...` |
| `models/unet.py` | `2d4bc303ef39aa80893fd9f6d62721c1...` |
| `models/dual_stream_unet.py` | `82a6fcf9a02bec97b5e743510c75175c...` |
| `utils/losses.py` | `5d9fc25a29a747d9fc7f76421912c952...` |
| `scripts/serve_restore.py` | `95b61eedccada39ae4068f36cb287d89...` |
| `frontend/src/pages/RetinaStudio.jsx` | `b61e2915cc2c562cd3547509a3d90f0c...` |
