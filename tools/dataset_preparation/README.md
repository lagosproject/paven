# Dataset Preparation Suite

This directory contains data preparation tools used to acquire, clean, and format visual attention datasets (LEDOV, DHF1K, MVS, AVS1K, HVEC, etc.) for PAVEN model training.

---

## 🛠️ Roles of MATLAB vs Python in the Research Pipeline

During the research and development of PAVEN (documented in Chapter 4 of the Master's Thesis and Section 3 of the EAAI 2025 paper), data preparation was divided between **MATLAB** and **Python**:

| Task | Language / Tool | Key Files | Rationale |
|---|:---:|---|---|
| **Gaussian Fixation Mask Generation** | **MATLAB** | [`matlab/make_gauss_masks4.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/make_gauss_masks4.m)<br>[`matlab/ledov_generate_masks.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/ledov_generate_masks.m)<br>[`matlab/hvec_generate_groundtruth.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/matlab/hvec_generate_groundtruth.m) | Official DHF1K benchmark standard; exact analytical Gaussian formula matching the literature. |
| **Video Extraction & Slicing** | **Python** | [`video_splitter.py`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/video_splitter.py) | Fast extraction of sequential image frames from raw video files (MP4/AVI). |
| **Frame / Mask Alignment & Integrity** | **Python** | [`frame_counter.py`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/frame_counter.py) | Filters out videos $<32$ frames (minimum model temporal window) and ensures 1:1 frame-to-mask correspondence. |
| **High-Performance Resizing** | **Python** | [`resize_dataset_parallel.py`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/resize_dataset_parallel.py) | Multithreaded in-place Lanczos resizing ($384 \times 224$) with resume logging on HPC clusters. |
| **Pure Python KDE Alternative** | **Python** | [`compute_saliency_kde.py`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/compute_saliency_kde.py) | Standalone Python alternative for researchers without a MATLAB license. |
| **Saliency Overlay Preview** | **Python** | [`video_fusion_overlay.py`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/dataset_preparation/video_fusion_overlay.py) | Superimposes saliency heatmaps with a JET colormap on RGB video frames. |

---

## 🚀 Step-by-Step Dataset Preparation Workflow

### 1. Download Datasets
```bash
bash tools/dataset_preparation/download_dhf1k.sh
bash tools/dataset_preparation/download_ledov.sh
```

### 2. Extract Frames
```bash
python tools/dataset_preparation/video_splitter.py --input /path/to/videos --output /path/to/frames
```

### 3. Generate Ground-Truth Saliency Maps
- **Option A (Original / Recommended):** Run the MATLAB routines in `matlab/`:
  ```matlab
  % In MATLAB:
  cd('tools/dataset_preparation/matlab');
  ledov_generate_masks;
  ```
- **Option B (Pure Python fallback):**
  ```bash
  python tools/dataset_preparation/compute_saliency_kde.py -i /path/to/annotations -o /path/to/maps
  ```

### 4. Resize and Standardize ($384 \times 224$)
```bash
python tools/dataset_preparation/resize_dataset_parallel.py --directory /path/to/dataset --width 384 --height 224 --workers 8
```

### 5. Verify Dataset Integrity
```bash
python tools/dataset_preparation/frame_counter.py --dir /path/to/dataset --min-frames 32
```
