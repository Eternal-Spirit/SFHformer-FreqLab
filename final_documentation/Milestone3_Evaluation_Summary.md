# Milestone 3 Final Evaluation Summary

> **Protocol Snapshot**: `v3.1-phase5.2-verified` | **Thresholds**: Dual-Stream `0.40` (FIVES Clean-Val Calibrated) vs Single-Stream Baselines `0.50` vs Frangi `0.0010`  
> **Scope Notice**: RetinaScan Studio is a **research prototype for retinal vessel segmentation**, not a clinically validated diagnostic tool.


## 1. Principal Contributions & Proposed Architecture
Single-stream vessel segmentation faces a sharp trade-off: training on native images (`Raw U-Net`) maximizes macro precision (`0.9266`) and continuity (`clDice 0.8134`) while missing low-contrast thin capillaries, whereas training on `SFHformer`-restored images (`Restored U-Net`) increases thin-vessel sensitivity on HRF (`+13.67 pp`) at the cost of coarse-label precision drops, fragmentation (`114.25` components/image on FIVES), and Glaucoma optic-cup rim false positives (`57.59%` precision).

`DualStreamUNet` resolves this divergence by fusing warm-started Raw and SFHformer-Restored encoders across all 5 scales via per-pixel spatial softmax gates (`w_raw + w_rest = 1`) and SE cross-residuals, optimized under FP32 `HybridTopologyLoss` (`SoftSkeletonize(iter=12, border_ignore=2)`).

![Proposed Dual-Stream Hybrid U-Net Architecture|6.9|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_4_dual_stream_hybrid_unet.png)



## 2. Primary In-Domain Benchmark: FIVES Untouched Test (`N=200`)

| System (`Frozen tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw U-Net** (`0.50`) | `0.8133` | **`0.9266`** | `0.7566` | `33.33%` | `67.32%` | `0.8134` | `52.19` |
| **Restored U-Net** (`0.50`) | `0.6927` | `0.7142` | `0.6824` | `29.68%` | `60.44%` | `0.6536` | `114.25` |
| **Dual-Stream Hybrid** (`0.40`) | **`0.8474`** | `0.9007` | **`0.8242`** | **`43.52%`** | **`74.69%`** | **`0.8537`** | **`41.22`** |

- **Statistical Gains vs Raw U-Net (`N=200`, Holm-Bonferroni corrected)**: Dice `+0.0341` (`p=8.72e-28`), Thin Skeleton Sensitivity `+10.20 pp` (`p=4.09e-29`), Branch Recall `+7.36 pp`, `clDice +0.0402`, Fragmentation `-10.97` CCs/img (`N=199` deduplicated matches identically).
- **Statistical Gains vs Restored U-Net (`N=200`)**: Dice `+0.1546`, Precision `+0.1866`, Thin Skeleton Sensitivity `+13.85 pp`, `clDice +0.2001`, Fragmentation `-73.03` CCs/img.

### FIVES Pathology Stratification (`n=50` per Class)

| Pathology | Raw Dice / Prec | Restored Dice / Prec | Dual-Stream Dice / Prec | Dual ThinSkel | Dual clDice | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** (n=50) | `0.8650` / `0.9052` | `0.6978` / `0.7034` | **`0.8837` / `0.9016`** | `45.35%` | `0.8965` | `49.9` |
| **AMD** (n=50) | `0.8696` / `0.9452` | `0.8040` / `0.8293` | **`0.8911` / `0.9022`** | `52.49%` | `0.8965` | `44.56` |
| **DR** (n=50) | `0.8181` / `0.9450` | `0.7075` / `0.7481` | **`0.8481` / `0.8994`** | `42.47%` | `0.8565` | `42.0` |
| **Glaucoma** (n=50) | `0.7004` / `0.9113` | `0.5616` / `0.5759` | **`0.7667` / `0.8998`** | `33.79%` | `0.7651` | `28.42` |

![Full-Resolution FIVES Disease-Stratified Qualitative Evidence|6.9|4.6](D:/SFHformer-FreqLab/demo_package/qualitative_examples/fives_test_disease_stratified_demo.png)



## 3. External HRF Transfer (`N=45`: `40` Exposed / `5` Exploratory)

>! **Disclosure**: `40/45` HRF images were seen upstream during `SFHformer` pretraining. The remaining `5` unexposed images carry insufficient power (`min Wilcoxon p = 0.0625`, `Holm p = 0.375`) and are presented as exploratory evidence only.

| Cohort & System (`tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HRF Overall (`N=45`) — Raw** (`0.50`) | **`0.7601`** | **`0.7482`** | `0.7900` | `36.81%` | `61.54%` | `0.7467` | `80.1` |
| **HRF Overall (`N=45`) — Restored** (`0.50`) | `0.7368` | `0.6532` | **`0.8519`** | **`50.48%`** | **`69.40%`** | `0.7546` | `166.0` |
| **HRF Overall (`N=45`) — Dual-Stream** (`0.40`) | `0.7419` | `0.6628` | `0.8497` | `46.59%` | `68.13%` | **`0.7745`** | **`67.6`** |
| **HRF Unexposed (`N=5`) — Raw** (`0.50`) | **`0.7463`** | **`0.7263`** | `0.7966` | `47.77%` | `66.19%` | `0.7540` | `116.2` |
| **HRF Unexposed (`N=5`) — Restored** (`0.50`) | `0.7205` | `0.6233` | **`0.8596`** | **`58.97%`** | **`73.04%`** | `0.7521` | `212.0` |
| **HRF Unexposed (`N=5`) — Dual** (`0.40`) | `0.7311` | `0.6423` | `0.8590` | `56.30%` | `72.89%` | **`0.7854`** | **`90.6`** |

### Exploratory Unexposed Per-Image Breakdown (`N=5`)

| Scan | Dual Dice | Dual Prec | Dual ThinSkel | Dual clDice | Dual BranchRecall | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `14_g.png` | `0.7281` | `0.6667` | `42.42%` | `0.7507` | `62.55%` | `37` |
| `14_h.png` | `0.7601` | `0.6659` | `64.68%` | `0.8511` | `79.21%` | `95` |
| `15_dr.png` | `0.6607` | `0.5154` | `66.40%` | `0.7215` | `82.00%` | `210` |
| `15_g.png` | `0.6531` | `0.5713` | `25.29%` | `0.6684` | `55.09%` | `54` |
| `15_h.png` | `0.8536` | `0.7922` | `82.72%` | `0.9353` | `85.59%` | `57` |

![HRF Unexposed Exploratory Transfer Comparison|6.9|3.1](D:/SFHformer-FreqLab/demo_package/qualitative_examples/hrf_transfer_unexposed_demo.png)



## 4. Checkpoint Provenance & End-to-End Summary
- `sfhformer` / `fives_retina_finetune/best.pth` (SHA-256: `d74e267c6ac664686528...`)
- `unet_raw/best.pth` (SHA-256: `97639a65b6cc3d09d063...`)
- `unet_restored/best.pth` (SHA-256: `2cfb37371cc5c0038bba...`)
- `unet_dual_stream/best.pth` (SHA-256: `9be2213e0cf0c79998f3...`)

![Combined End-to-End System Architecture|6.9|3.8](D:/SFHformer-FreqLab/demo_package/diagrams/diag_5_combined_retinascan_system.png)
