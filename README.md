# SFHformer-FreqLab & Dual-Stream Hybrid U-Net (`RetinaScan Studio`)

**Spatial-Frequency Retinal Fundus Image Restoration and Topology-Preserving Dual-Stream Vessel Segmentation**

- **Immutable Snapshot Tag**: `v3.1-phase5.2-verified` (`42ac0565d1d1517ef5bad9fdc1f392608272db1b`)
- **Hardware Validation Target**: NVIDIA GeForce RTX 5060 Laptop GPU (`8 GB` VRAM, BF16 AMP + FP32 `SoftclDice`)
- **Primary In-Domain Test**: FIVES Untouched Test (`N=200`, `N=199` deduplicated) | **External Transfer**: HRF (`N=45`: `40` pretrain-exposed, `5` unexposed exploratory)

> **Scope & Clinical Disclaimer**: RetinaScan Studio and all neural models in this repository constitute a **research prototype for retinal vessel segmentation and spatial-frequency restoration analysis**. They have not undergone prospective clinical validation or regulatory clearance and must **not** be used as a diagnostic medical device.

---

## Clean Repository Structure

```text
D:\SFHformer-FreqLab\
├── README.md                              # Project overview, benchmarks & quickstart
├── LICENSE                                # Repository license
├── requirments.txt                        # Python environment dependencies
├── models/                                # Core neural architectures
│   ├── SFHformer.py                       # 5-stage SFHformer-m restoration backbone (14.18M params)
│   ├── unet.py                            # 4-stage DoubleConv + Bilinear clinical U-Net (7.7M params)
│   └── dual_stream_unet.py                # DualStreamUNet + CrossStreamFusionBlock (16.41M params)
├── freqlab/                               # Spectral gating, FFT mixers, image corruptions & metrics
│   ├── gate.py                            # RadialBandGate (16 learnable log-amplitude spectral bands)
│   ├── sfhformer_gated.py                 # Frequency-gated SFHformer variant (R2)
│   ├── classical.py / corrupt.py / metrics.py
│   └── retina_corrupt.py / retina_data.py / retina_engine.py
├── utils/                                 # Topology losses & shared training utilities
│   └── losses.py                          # BCEDiceLoss, FP32 SoftSkeletonize(12), HybridTopologyLoss
├── scripts/                               # Active training, evaluation, auditing, serving & doc scripts
│   ├── train_fives_retina.py              # SFHformer restoration fine-tuning (FIVES online crops)
│   ├── train_unet.py                      # Single-stream Raw & Restored U-Net baselines
│   ├── train_dual_stream_unet.py          # Two-stage DualStreamUNet training + N=59 calibration
│   ├── evaluate_fives_retina.py           # Held-out 200-scan synthetic restoration evaluation
│   ├── evaluate_milestone3.py             # Frozen single-stream & Frangi Milestone 3 evaluation
│   ├── evaluate_dual_stream_milestone3.py # Frozen Dual-Stream Milestone 3 eval + Holm-Bonferroni stats
│   ├── baseline_frangi.py                 # Calibrated multiscale Hessian Frangi control (tau=0.0010)
│   ├── milestone1_data_audit.py           # Dataset disjointness, caliber profile & FOV clipping audit
│   ├── milestone1_followup_audit.py       # Follow-up pHash & coverage resolution
│   ├── milestone2_5_verification.py       # Clean-val N=59 paired verification (Raw vs Restored)
│   ├── serve_restore.py                   # Flask inference API (:5001) with GPU_LOCK & 6-stage timing
│   ├── retina_engine.py / retina_corrupt.py / retina_data.py
│   ├── qa_phase5_1.py                     # Automated API, alignment, latency & concurrency QA
│   ├── generate_phase5_3_diagrams.py      # 300-DPI technical architecture diagram suite (diag_1..5)
│   ├── build_phase5_3_demo_package.py     # Qualitative comparison & immutable snapshot packager
│   └── generate_phase6_documentation.py   # Phase 6 Master Technical Report & manifest compiler
├── frontend/                              # RetinaScan Studio React 18 / Vite web UI (:3000)
├── tests/                                 # Unit tests (`pytest tests/`)
├── checkpoints/                           # Active weights & clean-val calibration records ONLY
│   ├── fives_retina_finetune/best.pth     # SFHformer Milestone 3 restoration weights (67.80 MB)
│   ├── unet_raw/best.pth                  # Raw U-Net weights + training history (86.97 MB)
│   ├── unet_restored/best.pth             # Restored U-Net weights + training history (86.97 MB)
│   ├── unet_dual_stream/best.pth          # Dual-Stream Hybrid weights (62.74 MB) + validation_calibration.json
│   └── retina/, r1_baseline/, r2_gated/   # Baseline & ablation restoration models loaded by serve_restore.py
├── data/                                  # Retinal cohorts ONLY (`fives/`, `hrf/`, `retina/`, `retina_samples/`)
├── results/                               # Verified retinal audits & frozen Milestone 3 JSON benchmarks
│   ├── fives_evaluation_results.json      # 200-scan restoration fidelity evaluation
│   ├── milestone3/                        # Frozen per-image & statistical test results
│   ├── dataset_audit/                     # Milestone 1, follow-up pHash & Milestone 2.5 audits
│   └── benchmarks/frangi_baseline/        # Calibrated Frangi control outputs
├── demo_package/                          # Phase 5.3 300-DPI diagrams, qualitative figures & SNAPSHOT_MANIFEST.json
├── final_documentation/                   # Phase 6 Final_Technical_Report, Engineering_Doc, Summary & Manifest
└── archive_unused/                        # Non-deleted archive of legacy/upstream/superseded files (ARCHIVE_INDEX.md)
```

---

## Frozen Milestone 3 Summary (Untouched FIVES `N=200`)

| System (`Frozen tau`) | Dice | Precision | Sensitivity | Thin Skel (`r <= 3.5px`) | Branch Recall | `clDice` | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw U-Net** (`0.50`) | `0.8133` | **`0.9266`** | `0.7566` | `33.33%` | `67.32%` | `0.8134` | `52.19` |
| **Restored U-Net** (`0.50`) | `0.6927` | `0.7142` | `0.6824` | `29.68%` | `60.44%` | `0.6536` | `114.25` |
| **Dual-Stream Hybrid** (`0.40`) | **`0.8474`** | `0.9007` | **`0.8242`** | **`43.52%`** | **`74.69%`** | **`0.8537`** | **`41.22`** |

- External HRF transfer (`N=45`, `40` upstream pretrain-exposed / `5` exploratory): `DualStreamUNet` improves thin skeleton sensitivity (`+9.78 pp` vs Raw), branch recall (`+6.59 pp`), `clDice` (`0.7745`), and component count (`-12.49` CCs/img) while trading `-8.54 pp` Precision and `-1.82 pp` Dice vs Raw U-Net (`min Wilcoxon p = 0.0625`, `Holm p = 0.375` on exploratory `N=5`).

---

## Quickstart Commands

```powershell
# Run unit tests
d:\SFHformer-FreqLab\.venv\python.exe -m pytest tests/ -q

# Start RetinaScan Studio Flask Backend (:5001) & React UI (:3000)
d:\SFHformer-FreqLab\.venv\python.exe -u scripts\serve_restore.py
Set-Location frontend; npm run dev

# Re-run Frozen Milestone 3 Dual-Stream Evaluation
d:\SFHformer-FreqLab\.venv\python.exe scripts\evaluate_dual_stream_milestone3.py

# Regenerate Phase 6 PDFs & DOCUMENTATION_MANIFEST.json
d:\SFHformer-FreqLab\.venv\python.exe scripts\generate_phase6_documentation.py
```

---

## Documentation & Archived Files

- **Phase 6 Final Dossier**: See [`final_documentation/Final_Technical_Report.pdf`](final_documentation/Final_Technical_Report.pdf), [`final_documentation/Engineering_Documentation.pdf`](final_documentation/Engineering_Documentation.pdf), and [`final_documentation/Milestone3_Evaluation_Summary.pdf`](final_documentation/Milestone3_Evaluation_Summary.pdf).
- **Non-Destructive Archive**: All unused upstream multi-task code (haze/rain/snow/blur/lowlight/underwater), DIV2K crops (`1.1 GB+`), early draft reports/diagrams, scratch checkpoints, and legacy scripts were moved without deletion to [`archive_unused/`](archive_unused/ARCHIVE_INDEX.md).
