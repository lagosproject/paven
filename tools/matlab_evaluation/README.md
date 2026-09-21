# Video Saliency Evaluation Benchmark (MATLAB)

This directory contains the official evaluation suite for video saliency prediction metrics based on the **MIT Saliency Benchmark** and **DHF1K benchmark** (Wang et al., CVPR 2018 / TPAMI 2019).

---

## Supported Saliency Metrics

| Metric | Function | Requires Continuous Map? | Requires Discrete Fixations? | Range / Direction |
|---|---|:---:|:---:|---|
| **CC** | [`CC.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/CC.m) | Yes | No | $[-1, 1]$ ($\uparrow$ better) |
| **Similarity (SIM)** | [`similarity.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/similarity.m) | Yes | No | $[0, 1]$ ($\uparrow$ better) |
| **NSS** | [`NSS.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/NSS.m) | Yes | Yes | $(-\infty, +\infty)$ ($\uparrow$ better) |
| **AUC-Judd** | [`AUC_Judd.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/AUC_Judd.m) | Yes | Yes | $[0, 1]$ ($\uparrow$ better) |
| **AUC-Borji** | [`AUC_Borji.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/AUC_Borji.m) | Yes | Yes | $[0, 1]$ ($\uparrow$ better) |
| **sAUC (Shuffled)** | [`AUC_shuffled.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/AUC_shuffled.m) | Yes | Yes (with other fixations) | $[0, 1]$ ($\uparrow$ better) |
| **KL-Divergence** | [`KLdiv.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/KLdiv.m) | Yes | No | $[0, +\infty)$ ($\downarrow$ better) |
| **Information Gain** | [`InfoGain.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/InfoGain.m) | Yes | Yes | $(-\infty, +\infty)$ ($\uparrow$ better) |
| **Earth Mover's Dist.**| [`EMD.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/EMD.m) | Yes | No | $[0, +\infty)$ ($\downarrow$ better) |

---

## How to Run Evaluation

### 1. Benchmark Evaluation on JVET CTC Test Sequences
The script [`HVECTest.m`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/HVECTest.m) evaluates predicted saliency maps against eye-tracking ground truth for test sequences:

```matlab
% In MATLAB:
cd('tools/matlab_evaluation');
HVECTest
```

Configure `options.SALIENCY_DIR` (where PAVEN outputs are stored) and `options.DS_GT_DIR` (where ground truth `.jpg` and `_fixmaps.mat` are saved) inside `HVECTest.m`.

### 2. FastEMD Compilation (Optional)
To compute EMD, compile the MEX wrapper:
```matlab
cd FastEMD
compile_FastEMD
```
A precompiled Linux 64-bit MEX binary is provided at [`emd_hat_gd_metric_mex.mexa64`](file:///media/vant/Elements/PAVEN_RELEASE/github_paven/tools/matlab_evaluation/emd_hat_gd_metric_mex.mexa64).

---

## Citations
```bibtex
@InProceedings{Wang_2018_CVPR,
  author    = {Wang, Wenguan and Shen, Jianbing and Guo, Fang and Cheng, Ming-Ming and Borji, Ali},
  title     = {Revisiting Video Saliency: A Large-Scale Benchmark and a New Model},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}

@Article{Wang_2019_revisitingVS,
  author  = {Wang, Wenguan and Shen, Jianbing and Xie, Jiawei and Cheng, Ming-Ming and Ling, Haibin and Borji, Ali},
  title   = {Revisiting Video Saliency Prediction in the Deep Learning Era},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2019}
}
```