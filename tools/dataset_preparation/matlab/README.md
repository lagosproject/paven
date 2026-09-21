# MATLAB Dataset Preprocessing Suite

This directory contains MATLAB routines used during the research and development of **PAVEN** (UPM / Elsevier EAAI 2025) to convert raw eye-tracking coordinates into Gaussian fixation heatmaps.

---

## Script Overview

### 1. Fixation Mask Generators (`make_gauss_*.m`)
- [`make_gauss_masks.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/make_gauss_masks.m): Standard 2D Gaussian kernel dropping centered at $(x, y)$ fixation points.
- [`make_gauss_masks4.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/make_gauss_masks4.m): Optimized variant with configurable Gaussian window radius (`fixsize`), used for LEDOV and EyeFixationMaps.
- [`make_gauss_masks2.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/make_gauss_masks2.m): Fixation-duration weighted Gaussian accumulation.
- [`make_gauss_edited.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/make_gauss_edited.m): Standardized orientation kernel used for the JVET CTC test sequence ground-truth generation.
- [`make_gauss_masks_hd.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/make_gauss_masks_hd.m): Calibrated for 1080p/4K high-density displays ($1^\circ \approx 60$ px).

### 2. Dataset Processing Pipelines
- [`ledov_generate_masks.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/ledov_generate_masks.m):
  Batch processing pipeline for the LEDOV dataset. Reads `VideoNameList.mat` and `Data.mat`, extracts fixation timestamps, filters visual center fixations, applies `make_gauss_masks4`, and writes normalized grayscale saliency maps (`0001.jpg`, etc.).
- [`hvec_generate_groundtruth.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/hvec_generate_groundtruth.m):
  Processes `video_database.mat` for the test sequences, exporting both:
  1. Continuous ground-truth saliency heatmaps (`frame_%04d.jpg`).
  2. Binary discrete fixation matrices (`[Video]_fixmaps.mat`) for AUC-Judd / NSS evaluation.
- [`demo_fixation_overlay.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/demo_fixation_overlay.m):
  Demonstration script that superimposes generated fixation heatmaps onto sample video frames using MATLAB's `jet` colormap.

---

## Citation & Reference
- **Thesis**: Pablo Fernández Lagos, *"Codificación perceptual de secuencias de vídeo basada en redes neuronales"*, ETSI Informáticos, Universidad Politécnica de Madrid, 2025. ([UPM Archive](https://oa.upm.es/88254/))
- **Paper**: Elsevier *Engineering Applications of Artificial Intelligence*, 2025. DOI: [10.1016/j.engappai.2025.111664](https://doi.org/10.1016/j.engappai.2025.111664)
