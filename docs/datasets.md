# PAVEN Dataset Inventory and Acquisition Guide

This document details the 13 free-viewing eye-tracking video datasets analyzed and utilized in the development and validation of the PAVEN architecture (based on Section 4.3.1 and Table 4.2 of Pablo Fernández Lagos's Master Thesis, UPM 2025).

---

## 1. Overview of Evaluated Datasets

All datasets were recorded under **Free Viewing** conditions (viewers freely observed video sequences without constrained action-recognition prompts), preventing behavioral biases towards specific human activities.

| Dataset | Total Videos | Used Videos | Video Duration | Resolution | Reference / Paper Citation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **AVAD** | 45 | 45 | 5–10 s | $\le 1280 \times 720$ | Min et al., IEEE TIP 2016 |
| **AVS1K** | 1,000 | 709 | $\sim 5.9$ s | $\le 1280 \times 720$ | Drone-view visual attention benchmark (2021) |
| **Coutrot 1** | 60 | 60 | $\sim 17$ s | $720 \times 576$ | Coutrot et al., Neurocomputing 2014 |
| **Coutrot 2** | 15 | 15 | $\sim 44$ s | $1232 \times 504$ | Coutrot et al., PLOS ONE 2015 |
| **DHF1K** | 1,000 | 700 | 17–42 s | $640 \times 360$ | Wang et al., CVPR 2018 |
| **DIEM** | 85 | 81 | 27–217 s | $\le 1280 \times 720$ | Mital et al., Cognitive Computation 2011 |
| **ETMD** | 12 | 12 | 3–3.5 min | $340 \times 256$ | Film excerpts with color/grayscale trials |
| **HD & UHD** | 35 | 35 | 12 s | $\le 3840 \times 2160$ | High and Ultra-HD Eye Tracking (IRCCYN IVC) |
| **HVEC Eye** | 33 | 33 | 5–25 s | $\le 1920 \times 1080$ | JVET reference video sequences |
| **LEDOV** | 538 | 538 | 5–60 s | $1280 \times 720$ | Lai et al., IEEE TMM 2019 |
| **MVVA** | 300 | 299 | 10–30 s | $1280 \times 720$ | Multi-view video attention database |
| **MVS** | 1,007 | 1,007 | 5–20 s | $720 \times 1280$ | TikTok vertical videos (eliminates horizontal bias) |
| **SumMe** | 25 | 25 | 38–388 s | $340 \times 256$ | Gygli et al., ECCV 2014 |
| **Total** | **4,154** | **3,559** | | | **1,376,884 usable frames analyzed** |

---

## 2. Official Acquisition Links

### A. DHF1K (Primary Benchmark)
- **Website:** [https://hengshuangzhao.github.io/projects/DHF1K.html](https://hengshuangzhao.github.io/projects/DHF1K.html)
- **Automatic script:** `bash tools/dataset_preparation/download_dhf1k.sh`
- **Structure:**
  ```
  data/DHF1K/
  ├── video/       # Frames in JPG format (0001.jpg, 0002.jpg...)
  ├── annotation/  # Binary fixation point coordinates
  └── maps/        # Continuous Gaussian ground truth maps
  ```

### B. LEDOV (Large-Scale Eye-Tracking Over Videos)
- **GitHub Repository:** [https://github.com/remega/LEDOV](https://github.com/remega/LEDOV)
- **Automatic script:** `bash tools/dataset_preparation/download_ledov.sh`

### C. MVS (Vertical Video Saliency)
- Critical dataset consisting of 1,007 vertical format videos from TikTok, specifically incorporated in the PAVEN retraining process to avoid horizontal-screen spatial bias.

---

## 3. Data Preprocessing Pipeline

To prepare raw downloaded datasets for training:

1. **Resize Frames:** Standardize all video frames and ground truth saliency maps to $384 \times 224$:
   ```bash
   python tools/dataset_preparation/resize_dataset.py --input-dir /path/to/raw --output-dir /path/to/resized
   ```
2. **Generate Continuous Saliency Maps from Fixations:**
   ```bash
   python tools/dataset_preparation/compute_saliency_kde.py -i /path/to/annotations -o /path/to/maps
   ```
3. **Verify Frame Count Integrity:**
   ```bash
   python tools/dataset_preparation/frame_counter.py
   ```
