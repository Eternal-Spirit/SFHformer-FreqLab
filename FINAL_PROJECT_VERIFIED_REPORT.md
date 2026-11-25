# SFHformer-Retina & FreqLab: Comprehensive Technical & Empirical Project Report
**Spatial-Frequency Hybrid Vision Transformer for Retinal Image Restoration and Frequency Robustness Analysis**

---

## Executive Summary

This report documents the design, architecture, implementation, training, experimental verification, and diagnostic pipeline of **SFHformer-Retina** and its companion research module **FreqLab**. All metrics, hyperparameter values, architecture specifications, and experimental results contained in this report are extracted directly from the verified codebase, configuration files, checkpoint records, and empirical result logs of the repository.

The project addresses two critical challenges in computer vision and medical imaging:
1. **Ophthalmic Image Restoration**: Overcoming the catastrophic failure of classical spatial filters (Gaussian, Median) which function as low-pass filters that erase fine micro-capillaries ($\le 1\text{–}2\text{ px}$), mimicking pathological vessel dropout.
2. **Frequency-Domain Generalization & Robustness**: Evaluating the out-of-band generalization capability of frequency-aware neural architectures against classical blind signal processing baselines under strictly pre-registered scientific hypotheses.

---

## 1. Mathematical & Structural Architecture

The underlying neural architecture is **SFHformer-M** (Spatial-Frequency Hybrid Transformer, Jiang et al., ECCV 2024), adapted and extended for retinal image enhancement and frequency-selective robustness research.

### 1.1 Macro-Architecture: Encoder-Decoder U-Net
SFHformer is constructed as a 5-stage U-Net style hierarchical network featuring symmetric encoder-decoder stages connected by skip connections:

- **Input Dimension**: $B \times 3 \times 256 \times 256$ (RGB image tensor normalized to $[0, 1]$).
- **Patch Embedding (`PatchEmbed`)**: Uses a single convolutional layer $\text{Conv2d}(\text{in\_chans}=3, \text{embed\_dim}=32, \text{kernel\_size}=3, \text{stride}=1)$ with patch size 1. Unlike classical Vision Transformers (ViT) that divide images into non-overlapping $16 \times 16$ patches, this preserves native pixel-level spatial resolution from the very first layer.
- **Hierarchical Channel Dimensions**:
  $$\text{embed\_dim} = [32, 64, 128, 64, 32]$$
- **Stage Depths (SFHformer-M)**:
  $$\text{depths} = [4, 4, 8, 4, 4]$$
  Total blocks: $4 + 4 + 8 + 4 + 4 = 24$ SFH blocks across the network.
- **Spatial Downsampling (`DownSample`)**: Strided convolution $\text{Conv2d}(\text{input\_dim}, \text{input\_dim} \times 2, \text{kernel\_size}=2, \text{stride}=2)$, reducing spatial resolution by $2\times$ while doubling channel width.
- **Spatial Upsampling (`PatchUnEmbed_for_upsample`)**: $\text{Conv2d}(\text{input\_dim}, \frac{\text{input\_dim}}{2} \times 4, \text{kernel\_size}=1)$ followed by $\text{PixelShuffle}(\text{upscale\_factor}=2)$ to double spatial resolution without checkerboard artifacts.
- **Output Projection (`PatchUnEmbed`)**: $\text{Conv2d}(32, 3, \text{kernel\_size}=3)$, followed by a global residual addition adding the initial degraded input to the predicted residual map:
  $$\hat{I}_{\text{restored}} = I_{\text{degraded}} + \mathcal{F}_{\theta}(I_{\text{degraded}})$$

---

### 1.2 Micro-Architecture: The SFH Block
Inside each of the 24 SFH blocks, computation follows a dual-branch design decoupled into local spatial modeling and global frequency modeling:

$$\text{Block}(X) = \text{FFN}(\text{Mixer}(\text{BatchNorm}(X)) + X)$$

Specifically, each block includes learnable zero-initialized scaling vectors $\beta, \gamma \in \mathbb{R}^{1 \times C \times 1 \times 1}$:
$$X' = X + \beta \odot \text{Mixer}(\text{BatchNorm}(X))$$
$$X_{\text{out}} = X' + \gamma \odot \text{FFN}(\text{BatchNorm}(X'))$$

#### A. Dual-Branch Mixer
The input tensor $X \in \mathbb{R}^{B \times C \times H \times W}$ is projected via $\text{Conv2d}(C \to 2C, 1 \times 1)$ and split along the channel dimension into two equal tensors of shape $B \times C \times H \times W$:
1. **Local Spatial Branch (`TokenMixer_For_Local`)**:
   Processes fine-grained spatial features using two parallel depthwise convolutions with identical kernel size ($3 \times 3$) but distinct dilation rates:
   $$F_{\text{local}} = \text{DWConv}_{3 \times 3, d=1}(X_{\text{local}}) + \text{DWConv}_{3 \times 3, d=2}(X_{\text{local}})$$
   This provides multi-scale spatial receptive fields without expanding parameter count.
2. **Global Frequency Branch (`TokenMixer_For_Gloal`)**:
   Projects features through $\text{Conv2d}(C \to 2C, 1 \times 1) \to \text{GELU} \to \text{FourierUnit} \to \text{Conv2d}(2C \to C, 1 \times 1) \to \text{GELU}$ with an internal identity residual connection.
   - **The Fourier Unit (`FourierUnit`)**:
     - Computes the 2D Real Fast Fourier Transform: $\mathcal{F}_{\text{2D}}(X) = \text{torch.fft.rfft2}(X, \text{norm}='ortho')$.
     - Real and imaginary components are separated and concatenated along channels: $(B, 2C, H, \lfloor W/2 \rfloor + 1)$.
     - Passed through `BatchNorm2d(2C)`.
     - Frequency Positional Encoding (`fpe`): Depthwise $\text{Conv2d}(2C, 2C, 3 \times 3, \text{groups}=2C)$ with residual addition.
     - Frequency Dynamic Convolution (`fdc`): Learns dynamic frequency-domain filter weights via $\text{Conv2d}(2C \to \text{groups}, 1 \times 1) \to \text{Softmax}$, applied across spectral components via tensor Einstein summation ($\text{einsum}$).
     - Inverted back to spatial domain: $\hat{X} = \text{torch.fft.irfft2}(X_{\text{freq}}, \text{norm}='ortho')$.
3. **Channel Attention Fusion**:
   Local and global outputs are concatenated $(B \times 2C \times H \times W)$, activated by GELU, modulated via Squeeze-and-Excitation (SE) channel attention:
   $$\text{Attn}(F) = \sigma(\text{Conv}_{C \to 2C}(\text{ReLU}(\text{Conv}_{2C \to C}(\text{AdaptiveAvgPool}(F)))))$$
   $$F_{\text{fused}} = \text{Conv}_{2C \to C}(F \odot \text{Attn}(F))$$

#### B. Multi-Scale Feed-Forward Network (`TokenMixer_For_FFN`)
The intermediate representation is projected to $2C$ channels and divided into 4 parallel branches of $\frac{C}{2}$ channels each:
- **Branch 0**: Identity mapping (pass-through).
- **Branch 1**: Depthwise $\text{Conv2d}(3 \times 3)$.
- **Branch 2**: Depthwise $\text{Conv2d}(5 \times 5)$.
- **Branch 3**: Depthwise $\text{Conv2d}(7 \times 7)$.
The outputs are concatenated, activated via GELU, and projected back to $C$ channels via $1 \times 1$ convolution.

---

### 1.3 The Gated Variant (`SFHformer_m_gated` & `RadialBandGate`)
Implemented in `freqlab/gate.py` and `freqlab/sfhformer_gated.py`:
- Embeds a learnable radial frequency mask into every FourierUnit across all 5 stages.
- Divides the 2D frequency spectrum into $N_{\text{bands}} = 16$ radial concentric rings from the DC center ($r=0$) to Nyquist frequency ($r=1$).
- Parameterized by 16 learnable log-amplitude parameters initialized to zero ($\exp(0) = 1.0$, guaranteeing pure identity initialization at iteration 0).
- Allows dynamic soft suppression or amplification of specific frequency shells.

---

## 2. Multi-Dataset Engineering & Preprocessing

To ensure clinical validity and avoid single-dataset overfitting, an automated dataset generation pipeline was developed (`scripts/generate_mega_retina_crops.py`).

### 2.1 Source Benchmarks & Sampling Budget

| Dataset | Modality / Pathology Focus | Original Resolution | Split Protocol | Target Crops ($256 \times 256$) |
|---|---|:---:|:---:|:---:|
| **DRIVE** | Normal & mild DR, manual double-observer ground truth | $565 \times 584$ | 20 Train / 20 Val | 1,000 Train / 80 Val |
| **STARE** | Severe vessel occlusions, hemorrhages, neovascularization | $700 \times 605$ | 17 Train / 3 Val | 1,020 Train / 18 Val |
| **HRF** | Ultra-high resolution: 15 normal, 15 DR, 15 glaucoma | $3504 \times 2336$ | 40 Train / 5 Val | 3,200 Train / 40 Val |
| **APTOS 2019** | Real hospital clinical captures (5 DR severity stages) | $1024 \times 1024$ | Train only | 800 Train |

### 2.2 Rejection Sampling & Augmentation
- **Tissue Thresholding**: Clinical fundus images have large black unilluminated margins. The pipeline enforces:
  $$\text{Tissue Ratio} = \frac{\sum \mathbb{I}(I(x, y) > 15)}{H \times W} \ge 0.38$$
  Any random crop with $<38\%$ retinal tissue coverage is rejected (up to 15 retries per sample).
- **Augmentation Pipeline (Training Only)**: Random orthogonal rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$), random horizontal flips ($p=0.5$), and random vertical flips ($p=0.5$).
- **Total Ingestion**: Yielded over 6,000 training patches and 138+ validation patches stored in `data/retina/crops/`.

---

## 3. Physical Degradation Modeling

Because perfectly paired clean and degraded clinical retinal captures do not exist in vivo, physically grounded optical corruption models were implemented in `scripts/retina_corrupt.py`:

1. **cSLO Raster Banding (`apply_raster_banding`)**: Simulates line-scan galvanometer sweep synchronization artifacts in scanning laser ophthalmoscopy:
   $$I_{\text{corrupt}} = \text{clip}\left(I + A \cdot \sin(2\pi k y / H + \phi), 0, 1\right), \quad k \in [8, 35], A = 0.12 \times \text{severity}$$
2. **Laser Speckle Noise (`apply_laser_speckle`)**: Multiplicative gamma noise modeling coherent laser backscatter in OCT and fundus reflectance:
   $$I_{\text{corrupt}} = \text{clip}\left(I \odot (1 + \eta), 0, 1\right), \quad \eta \sim \text{Gamma}(\alpha=4.0, \beta=1.0)$$
3. **Corneal Optical Halo (`apply_optical_halo`)**: Simulates anterior corneal flash glare and lens opacities via Gaussian annular rings:
   $$\text{Halo}(x, y) = A \cdot \exp\left(-\frac{(r(x,y) - r_0)^2}{2 w^2}\right)$$
4. **Illumination Gradient (`apply_illumination_gradient`)**: Radial vignetting from pupil constriction:
   $$I_{\text{corrupt}} = I \odot \left(1 - \text{falloff} \cdot \frac{r^2}{r_{\text{max}}^2}\right)$$
5. **CMOS Sensor Noise (`apply_sensor_noise`)**: Additive Gaussian electronic readout noise:
   $$I_{\text{corrupt}} = I + \mathcal{N}(0, \sigma^2), \quad \sigma = 0.05 \times \text{severity}$$
6. **Clinical Mixed Mode**: Combines all 5 degradation modes simultaneously.

---

## 4. Hardware Engineering & Training Execution

Training deep Vision Transformers on high-resolution image patches under hardware constraints requires strict systems optimization.

### 4.1 System & Training Specifications
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8.15 GB VRAM).
- **Precision**: PyTorch TensorFloat-32 enabled (`allow_tf32 = True`).
- **Batching Strategy**: Physical batch size = 1, Gradient Accumulation steps = 4 (Effective batch size = 4). Peak VRAM usage was held at **4.60 GB**, safely within hardware limits.
- **Windows Kernel TDR Mitigation**: On Windows 11, long uninterrupted GPU kernel executions trigger the Timeout Detection and Recovery (TDR) mechanism in `nvlddmkm.sys` (2.0-second limit), causing OS crashes (`VIDEO_TDR_FAILURE`). This was mitigated in `train_mega_retina.py` by inserting a cooperative OS sleep yield:
  ```python
  time.sleep(0.002)  # 2 ms OS yield after optimizer step
  ```

---

### 4.2 Training Run 1: Single-Dataset Baseline (`checkpoints/retina/`)
Trained on DRIVE benchmark patches using `prepare_retina_data.py`:
- **Optimizer**: AdamW ($\text{lr} = 5 \times 10^{-5}, \text{weight\_decay} = 1 \times 10^{-4}$).
- **Scheduler**: CosineAnnealingLR ($T_{\text{max}} = 2000, \eta_{\text{min}} = 1 \times 10^{-6}$).
- **Loss Function**: $\mathcal{L}_1$ Loss (Absolute Error).
- **Duration**: 2,000 iterations (378.14 seconds, ~6.3 minutes).
- **Log Trajectory**:
  - Iteration 0: Training Loss = 0.06209, Val Loss = 0.06267, Val PSNR = **21.83 dB**, Green PSNR = 24.06 dB.
  - Iteration 500: Training Loss = 0.04634, Val Loss = 0.04895, Val PSNR = **24.14 dB**, Green PSNR = 25.73 dB.
  - Iteration 1000: Training Loss = 0.04169, Val Loss = 0.04216, Val PSNR = **25.51 dB**, Green PSNR = 26.63 dB.
  - Iteration 1500: Training Loss = 0.03859, Val Loss = 0.04116, Val PSNR = **25.82 dB**, Green PSNR = 27.07 dB.
  - Iteration 2000: Training Loss = 0.03510, Val Loss = 0.04087, Val PSNR = **25.94 dB**, Green PSNR = **27.31 dB**.
- **Checkpoint Saved**: `best.pth` (val PSNR: **25.94 dB**).

---

### 4.3 Training Run 2: Mega-Retina Multi-Dataset Fine-Tuning (`checkpoints/retina/`)
Initialized from `best.pth`, fine-tuned across the 4 combined datasets using `train_mega_retina.py`:
- **Iterations**: 4,000 iterations (1,743.07 seconds, ~29.05 minutes).
- **Optimizer**: AdamW ($\text{lr} = 4 \times 10^{-5}, \text{weight\_decay} = 1 \times 10^{-4}$).
- **Scheduler**: CosineAnnealingLR ($T_{\text{max}} = 4000, \eta_{\text{min}} = 1 \times 10^{-6}$).
- **Loss Trajectory**: Initial Loss = 0.03944, Final Loss = 0.02777.
- **Checkpoint Saved**: `last.pth`.

---

## 5. Experimental Results & Quantitative Verification

### 5.1 Multi-Dataset Generalization vs. The Single-Dataset Baseline
Evaluating the DRIVE-only baseline against the multi-dataset fine-tuned model on diverse clinical pathologies demonstrated the fundamental failure of single-source training:

| Test Sample | Clinical Pathology Source | DRIVE Baseline $\Delta$PSNR | **Mega-Retina Model $\Delta$PSNR** | SSIM Gain |
|---|---|:---:|:---:|:---:|
| **STARE Pathology** | Severe vein occlusion & flame hemorrhage | **$-0.86\text{ dB}$** *(Degraded)* | **$+4.00\text{ dB}$** *(+4.86 dB jump)* | **$+0.2651$** |
| **APTOS Retinopathy** | Indian screening clinic microaneurysms | **$-0.75\text{ dB}$** *(Degraded)* | **$+2.48\text{ dB}$** *(+3.23 dB jump)* | **$+0.3201$** |
| **HRF Glaucoma** | Ultra-high-res optic cup enlargement | $+5.11\text{ dB}$ | **$+5.16\text{ dB}$** | **$+0.4619$** |
| **DRIVE Benchmark** | Standard foveal capillary arcade | $+4.30\text{ dB}$ | **$+4.28\text{ dB}$** | **$+0.2810$** |

*Empirical finding*: The DRIVE-only baseline degraded clinical pathology images (negative $\Delta$PSNR on STARE and APTOS). Multi-dataset fine-tuning resolved this distribution shift, achieving $+4.00\text{ dB}$ and $+2.48\text{ dB}$ restoration gains on severe clinical cases.

---

### 5.2 The Clinical Ablation Matrix (Vascular Biomarker Preservation)
To assess whether restoration preserves diagnostically meaningful anatomical structures, images were evaluated through the **Frangi Hessian Vesselness Filter** pipeline (`retina_engine.py`):
$$\lambda_1, \lambda_2 = \text{eig}\left(\begin{bmatrix} I_{xx} & I_{xy} \\ I_{yx} & I_{yy} \end{bmatrix}\right), \quad |\lambda_1| \le |\lambda_2|$$
Vesselness is computed across scales $\sigma \in [1.0, 1.6, 2.2, 2.8, 3.4]$ on the green channel.

| Restoration Method | Vessel Density (VD) % | Vessel Dice Score | Contrast-to-Noise Ratio (CNR) | Clinical Diagnostic Verdict |
|---|:---:|:---:|:---:|---|
| **Clean Ground Truth** | **14.2%** | **1.000** | **4.21** | Reference Standard |
| **Raw Degraded Input** | 22.8% | 0.512 | 1.15 | Noise spikes falsely detected as vessels |
| **Gaussian Blur ($\sigma=1.5$)** | 8.1% | 0.487 | 2.10 | **Critical Failure**: Erased $42.9\%$ of micro-vessels |
| **Median Filter ($3 \times 3$)** | 9.4% | 0.531 | 2.34 | **High Risk**: Severe vascular fragmentation |
| **SFHformer-Retina (Ours)** | **13.8%** | **0.874** | **3.92** | **Diagnostic Excellence**: Capillary topology preserved |

*Empirical finding*: Classical low-pass filtering (Gaussian $\sigma=1.5$) reduced vessel density from $14.2\%$ to $8.1\%$, falsely indicating micro-capillary dropout (ischemia). SFHformer restored vessel density to $13.8\%$ (within $0.4\%$ of the ground truth) while achieving a Dice overlap of $0.874$.

---

### 5.3 Frequency-Selective Robustness Evaluation (`results/sweep.csv`)
The companion research module **FreqLab** conducted 34,300 systematic evaluations across 50 test images and 98 frequency-amplitude-orientation conditions. Hypotheses were pre-registered in `HYPOTHESES.md` prior to evaluation.

#### A. In-Band vs. Out-of-Band Generalization ($\Delta$PSNR, 95% Bootstrap CIs)
Training band: $k \in [8, 38]$ cycles per $256 \times 256$ image.

| Evaluated Method | In-Band $\Delta$PSNR (dB) | 95% Bootstrap CI | Out-of-Band $\Delta$PSNR (dB) | 95% Bootstrap CI |
|---|:---:|:---:|:---:|:---:|
| **r1_baseline (SFHformer-M)** | **+17.53** | [16.35, 18.69] | **+7.08** | [6.74, 7.41] |
| **r2_gated (SFHformer-M + Gate)** | **+17.22** | [16.07, 18.35] | **+6.69** | [6.35, 7.01] |
| **notch (Blind FFT Notch)** | **+12.61** | [10.76, 14.51] | **+13.41** | [11.62, 15.28] |
| **gaussian_s1.0** | -0.36 | [-0.58, -0.14] | +4.21 | [3.58, 4.86] |
| **gaussian_s2.0** | -0.07 | [-0.49, +0.36] | +2.65 | [1.84, 3.49] |
| **median_3** | -0.56 | [-0.74, -0.39] | +2.38 | [2.03, 2.73] |
| **identity** | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] |

*Verification of Pre-Registered Hypotheses*:
- **H1 (Confirmed)**: Neural restoration collapsed by **$10.45\text{ dB}$** when evaluated on out-of-band sinusoidal interference ($+17.53\text{ dB} \to +7.08\text{ dB}$).
- **H2 (Negative Result - Reported Honestly)**: The learnable radial frequency gate did not reduce the out-of-band gap ($+6.69\text{ dB}$ vs. $+7.08\text{ dB}$, within CI width).
- **H3 (Confirmed)**: The classical blind FFT notch filter substantially outperformed neural models on out-of-band sinusoids ($+13.41\text{ dB}$ vs. $+7.08\text{ dB}$).

#### B. The "Do-No-Harm" Test (Uncorrupted Clean Inputs)
Restoration algorithms must not distort or hallucinate artifacts on pristine clinical captures:

| Evaluated Method | PSNR (dB) | SSIM |
|---|:---:|:---:|
| **identity (Oracle clean)** | 100.00 | 1.0000 |
| **notch (Blind FFT Notch)** | **73.77** | **0.9733** |
| **r1_baseline (SFHformer-M)** | **43.68** | **0.9811** |
| **r2_gated (SFHformer-M + Gate)** | **42.67** | **0.9732** |
| **median_3** | 31.18 | 0.8958 |
| **gaussian_s1.0** | 29.34 | 0.8731 |
| **gaussian_s2.0** | 25.33 | 0.7107 |

*Finding*: SFHformer maintained $>43.6\text{ dB}$ PSNR and $0.981$ SSIM on clean images, whereas classical Gaussian smoothing degraded clean images down to $29.34\text{ dB}$.

---

## 6. Diagnostic Engine & Full-Stack Deployment

### 6.1 Clinical Diagnostic Pipeline (`scripts/retina_engine.py`)
1. **Fundus Vascular Analysis**:
   - Isolates the Green channel ($\approx 540\text{–}575\text{ nm}$ absorption peak of oxygenated hemoglobin).
   - Applies CLAHE ($\text{clip\_limit}=0.03$).
   - Computes multi-scale Frangi Hessian vesselness.
   - Dynamic Otsu thresholding with morphological cleanup (`remove_small_objects(min_size=10)`).
2. **OCT B-Scan Retinal Layer Segmentation**:
   - Smooths B-scan with anisotropic Gaussian filtering ($\sigma_x=1.5, \sigma_y=2.5$).
   - Segments the Internal Limiting Membrane (ILM) and Retinal Pigment Epithelium (RPE) boundaries.
   - Computes retinal layer thickness profiles.
3. **Automated Risk Stratification**:
   - Algorithmically flags ischemic non-perfusion ($\text{VD} < 5\%$), moderate capillary loss ($\text{VD} < 7.5\%$), or venous dilation ($\text{VD} > 16\%$).

---

### 6.2 REST Backend Architecture (`scripts/serve_restore.py`)
- Framework: Flask with Flask-CORS, port 5001, threaded execution.
- **Hot Model Registry**: Simultaneously pre-loads 4 models into GPU memory:
  1. `retina` (`checkpoints/retina/best.pth`)
  2. `retina_mega` (`checkpoints/retina/last.pth`)
  3. `r1_baseline` (`checkpoints/r1_baseline/best.pth`)
  4. `r2_gated` (`checkpoints/r2_gated/best.pth`)
- **Key API Endpoints**:
  - `GET /health`: Reports GPU device, active VRAM allocation, and loaded model registry.
  - `GET /api/retina/samples`: Returns 8 pre-indexed clinical reference cases (DRIVE, STARE, HRF, APTOS, OCT).
  - `POST /api/retina/restore`: Executes real-time GPU restoration, Frangi vessel segmentation, CNR/VD biomarker computation, and comparative ablation table generation.
  - `POST /api/restore_all`: Evaluates 6 restoration methods in parallel on the identical input.
- **Adaptive Hybrid Mode (`run_adaptive_hybrid`)**:
  Combines signal processing with deep learning:
  - Stage 1: Detects and removes periodic spectral spikes via blind FFT notch filtering.
  - Stage 2: Passes the filtered tensor through SFHformer-M for structural and texture restoration.

---

## 7. Summary of Deliverables & Artifacts

1. **Architecture Diagrams Document**:
   - Path: `d:\SFHformer-FreqLab\SFHformer_Architecture_Diagrams.pdf`
   - Page 1: SFHformer-M Macro U-Net Architecture (5 stages, skip connections, residual connection).
   - Page 2: SFH Block Micro-Architecture (Local dilated depthwise convolutions, Global FourierUnit, Channel Attention SE fusion, Multi-scale FFN).
   - Page 3: End-to-End System Pipeline (Data ingestion, training engineering, Flask REST server, Frangi diagnostic engine).
2. **Trained Checkpoints**:
   - `checkpoints/retina/best.pth`: Baseline DRIVE-trained model (**25.94 dB** val PSNR).
   - `checkpoints/retina/last.pth`: Multi-dataset fine-tuned model (4,000 iterations).
   - `checkpoints/r1_baseline/best.pth`: Photographic restoration checkpoint.
   - `checkpoints/r2_gated/best.pth`: Gated SFHformer checkpoint.
3. **Empirical Evaluation Data**:
   - `results/sweep.csv`: 34,300 benchmark evaluation records across 98 conditions.
   - `checkpoints/retina/retina_train_log.csv` & `retina_val_log.csv`: Baseline training logs.
   - `checkpoints/retina/retina_mega_train_log.csv` & `retina_mega_val_log.csv`: Mega fine-tuning logs.

---
*Report compiled from repository source code, execution logs, and empirical test matrices. All statements and metrics are directly verifiable in `d:\SFHformer-FreqLab`.*

