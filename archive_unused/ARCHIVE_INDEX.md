# Archived Unused & Legacy Assets (`archive_unused/`) Index

**Policy**: Zero files were deleted during repository restructuring. Every upstream non-retinal module, superseded report/diagram draft, early FreqLab sweep artifact, scratch checkpoint, and legacy script was moved here with its original relative path preserved inside categorized buckets.

- **Total Relocated Entries**: `89` (`670` files, `1138.02 MB`)
- **Immutable Core Preserved**: All files in `SNAPSHOT_MANIFEST.json` (`v3.1-phase5.2-verified`) and `DOCUMENTATION_MANIFEST.json` remain untouched in their canonical paths.

| Category Bucket | Original Path | Files | Reason |
| :--- | :--- | :---: | :--- |
| `legacy_upstream_tasks` | `train_for_haze_its.py` | `1` | Upstream SFHformer ITS dehazing script (non-retinal) |
| `legacy_upstream_tasks` | `train_for_low_lol.py` | `1` | Upstream SFHformer LOL low-light script (non-retinal) |
| `legacy_upstream_tasks` | `train_for_motion_blur_gopro.py` | `1` | Upstream SFHformer GoPro motion-blur script (non-retinal) |
| `legacy_upstream_tasks` | `train_for_rain_rain100H.py` | `1` | Upstream SFHformer Rain100H script (non-retinal) |
| `legacy_upstream_tasks` | `train_for_rain_rain100L.py` | `1` | Upstream SFHformer Rain100L script (non-retinal) |
| `legacy_upstream_tasks` | `train_for_raindrop.py` | `1` | Upstream SFHformer Raindrop script (non-retinal) |
| `legacy_upstream_tasks` | `train_for_underwater_lsui.py` | `1` | Upstream SFHformer LSUI underwater script (non-retinal) |
| `legacy_upstream_tasks` | `dehazing` | `18` | Upstream SFHformer dehazing module (non-retinal) |
| `legacy_upstream_tasks` | `lowlight` | `16` | Upstream SFHformer low-light module (non-retinal) |
| `legacy_upstream_tasks` | `motion_blur` | `10` | Upstream SFHformer motion-blur module (non-retinal) |
| `legacy_upstream_tasks` | `image` | `21` | Upstream README sample images (haze/rain/snow) |
| `legacy_upstream_tasks` | `configs` | `25` | Upstream multi-task JSON configs (non-retinal) |
| `legacy_upstream_tasks` | `datasets` | `11` | Upstream non-retinal dataset loaders (ITS/LoL/Rain/Snow/etc.) |
| `legacy_upstream_tasks` | `dataset_spilt` | `7` | Upstream LSUI/UIEB dataset split text files |
| `legacy_upstream_tasks` | `splits` | `5` | Legacy DIV2K split index files |
| `legacy_upstream_tasks` | `runs` | `0` | Empty upstream TensorBoard/runs folder |
| `superseded_docs_and_diagrams` | `ARCHITECTURE_AND_SYSTEM_DESIGN_EXPLAINED.html` | `1` | Superseded draft HTML (0 KB) |
| `superseded_docs_and_diagrams` | `ARCHITECTURE_AND_SYSTEM_DESIGN_EXPLAINED.md` | `1` | Superseded draft MD (0 KB) |
| `superseded_docs_and_diagrams` | `ARCHITECTURE_AND_SYSTEM_DESIGN_EXPLAINED.pdf` | `1` | Superseded pre-DualStream architecture PDF |
| `superseded_docs_and_diagrams` | `EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.html` | `1` | Superseded deep-dive draft HTML |
| `superseded_docs_and_diagrams` | `EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.md` | `1` | Superseded deep-dive draft MD |
| `superseded_docs_and_diagrams` | `EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.pdf` | `1` | Superseded deep-dive draft PDF |
| `superseded_docs_and_diagrams` | `FINAL_PROJECT_VERIFIED_REPORT.html` | `1` | Superseded pre-Phase6 HTML report |
| `superseded_docs_and_diagrams` | `FINAL_PROJECT_VERIFIED_REPORT.md` | `1` | Superseded pre-Phase6 MD report |
| `superseded_docs_and_diagrams` | `FINAL_PROJECT_VERIFIED_REPORT.pdf` | `1` | Superseded pre-Phase6 PDF report |
| `superseded_docs_and_diagrams` | `PROJECT_REPORT_COMPLETE.md` | `1` | Superseded early FreqLab project report MD |
| `superseded_docs_and_diagrams` | `PROJECT_REPORT_SFHFORMER_RETINA.pdf` | `1` | Superseded early SFHformer-Retina report PDF |
| `superseded_docs_and_diagrams` | `MASTER_ARCHITECTURE_DIAGRAMS.pdf` | `1` | Superseded diagram PDF (replaced by demo_package/diagrams/diag_1..5.png) |
| `superseded_docs_and_diagrams` | `MASTER_DIAGRAM_1_TRAINING_AND_ARCHITECTURE.png` | `1` | Superseded root diagram PNG |
| `superseded_docs_and_diagrams` | `MASTER_DIAGRAM_2_IMAGE_PROCESSING_AND_DIAGNOSTICS.png` | `1` | Superseded root diagram PNG |
| `superseded_docs_and_diagrams` | `SFHformer_Architecture_Diagrams.pdf` | `1` | Superseded diagram PDF |
| `superseded_docs_and_diagrams` | `diagram_1.png` | `1` | Superseded early FreqLab diagram PNG |
| `superseded_docs_and_diagrams` | `diagram_2.png` | `1` | Superseded early FreqLab diagram PNG |
| `superseded_docs_and_diagrams` | `diagram_3.png` | `1` | Superseded early FreqLab diagram PNG |
| `superseded_docs_and_diagrams` | `docs` | `4` | Legacy pre-DualStream diagram set (docs/diagrams/arch_diag1..4.png) |
| `superseded_docs_and_diagrams` | `HYPOTHESES.md` | `1` | Early Phase-1 hypotheses scratchpad |
| `superseded_docs_and_diagrams` | `NOTES.md` | `1` | Early development notes |
| `superseded_docs_and_diagrams` | `RESULTS.md` | `1` | Early DIV2K spectral results notes |
| `superseded_docs_and_diagrams` | `README_FREQLAB.md` | `1` | Legacy standalone FreqLab README (consolidated into root README.md) |
| `superseded_docs_and_diagrams` | `README_RETINASCAN.md` | `1` | Legacy standalone RetinaScan README (consolidated into root README.md) |
| `superseded_docs_and_diagrams/upstream_sfhformer` | `README.md` | `1` | Original Beihang ECCV24 SFHformer README (replaced cleanly at root) |
| `legacy_scripts` | `scripts/auto_pipeline.ps1` | `1` | Early DIV2K FreqLab automation script |
| `legacy_scripts` | `scripts/benchmark_speedup_experiments.py` | `1` | One-off PyTorch compile/AMP speedup benchmark |
| `legacy_scripts` | `scripts/compile_exhaustive_deep_dive.py` | `1` | Replaced by scripts/generate_phase6_documentation.py |
| `legacy_scripts` | `scripts/download_fives.py` | `1` | One-time dataset fetcher (data/fives/ already ingested) |
| `legacy_scripts` | `scripts/frangi_check.py` | `1` | Superseded by scripts/baseline_frangi.py |
| `legacy_scripts` | `scripts/generate_architecture_explanation_pdf.py` | `1` | Replaced by generate_phase5_3_diagrams.py & generate_phase6_documentation.py |
| `legacy_scripts` | `scripts/generate_diagrams.py` | `1` | Replaced by scripts/generate_phase5_3_diagrams.py |
| `legacy_scripts` | `scripts/generate_master_diagrams.py` | `1` | Replaced by scripts/generate_phase5_3_diagrams.py |
| `legacy_scripts` | `scripts/generate_mega_retina_crops.py` | `1` | Legacy mega-crop prep (superseded by prepare_fives_dataset.py) |
| `legacy_scripts` | `scripts/generate_pdf_report.py` | `1` | Replaced by scripts/generate_phase6_documentation.py |
| `legacy_scripts` | `scripts/make_crops.py` | `1` | DIV2K crop generator (non-retinal) |
| `legacy_scripts` | `scripts/plot_results.py` | `1` | Early DIV2K sweep plotter |
| `legacy_scripts` | `scripts/run_frangi_experiments_cd.py` | `1` | Superseded by baseline_frangi.py & evaluate_milestone3.py |
| `legacy_scripts` | `scripts/sanity_check.py` | `1` | Early FreqLab synthetic sanity check |
| `legacy_scripts` | `scripts/spectrum_figs.py` | `1` | Early DIV2K spectrum plot generator |
| `legacy_scripts` | `scripts/sweep.py` | `1` | Early DIV2K radial frequency sweep script |
| `legacy_scripts` | `scripts/test_samples.py` | `1` | Early quick-check sample script |
| `legacy_scripts` | `scripts/train_freq.py` | `1` | Early DIV2K R1/R2 frequency-band training script |
| `legacy_scripts` | `scripts/train_mega_retina.py` | `1` | Superseded by scripts/train_fives_retina.py |
| `scratch_checkpoints` | `checkpoints/smoke_test` | `3` | Early training smoke-test checkpoint logs |
| `scratch_checkpoints` | `checkpoints/speed_benchmark` | `5` | Early AMP speedup benchmark checkpoint |
| `legacy_freq_results` | `results/do_no_harm.csv` | `1` | Early DIV2K radial gate CSV |
| `legacy_freq_results` | `results/freq_response.png` | `1` | Early DIV2K frequency response figure |
| `legacy_freq_results` | `results/pipeline_complete.txt` | `1` | Early DIV2K completion stamp |
| `legacy_freq_results` | `results/radial_error_in_band_k20.png` | `1` | Early DIV2K radial error figure |
| `legacy_freq_results` | `results/radial_error_out_of_band_k80.png` | `1` | Early DIV2K radial error figure |
| `legacy_freq_results` | `results/spectrum_in_band_k20.png` | `1` | Early DIV2K spectrum figure |
| `legacy_freq_results` | `results/spectrum_out_of_band_k80.png` | `1` | Early DIV2K spectrum figure |
| `legacy_freq_results` | `results/speedup_benchmark_results.json` | `1` | One-off GPU speedup benchmark JSON |
| `legacy_freq_results` | `results/summary_band.csv` | `1` | Early DIV2K band summary CSV |
| `legacy_freq_results` | `results/sweep.csv` | `1` | Early DIV2K frequency sweep CSV |
| `legacy_div2k_data` | `data/DIV2K_valid_HR.zip` | `1` | DIV2K validation archive (438 MB, non-retinal) |
| `legacy_div2k_data` | `data/crops` | `370` | DIV2K image crops (non-retinal) |
| `legacy_div2k_data` | `data/raw` | `100` | Unpacked DIV2K_valid_HR images (non-retinal) |
| `legacy_freq_results` | `results/heatmap1_freq_orient_gaussian_s1.0.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap1_freq_orient_gaussian_s2.0.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap1_freq_orient_identity.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap1_freq_orient_median_3.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap1_freq_orient_notch.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap1_freq_orient_r1_baseline.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap1_freq_orient_r2_gated.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_gaussian_s1.0.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_gaussian_s2.0.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_identity.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_median_3.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_notch.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_r1_baseline.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
| `legacy_freq_results` | `results/heatmap2_freq_amp_r2_gated.png` | `1` | Early DIV2K spectral orientation/amplitude heatmap |
