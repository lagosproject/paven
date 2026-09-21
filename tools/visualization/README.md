# Multi-Panel Video Comparison Visualization

This module contains tools to generate synchronized multi-panel video comparisons demonstrating the complete PAVEN pipeline:
1. **Top Row**:
   - Original RGB video frame.
   - Ground-truth gaze weighted frame (human eye-tracking fovea).
   - PAVEN predicted saliency weighted frame (neural visual attention).
2. **Middle Row**:
   - Segmented gaze clusters and contours.
   - Ground-truth human fixation density map.
   - PAVEN continuous neural saliency heatmap.
3. **Bottom Row**:
   - Discrete Coding Tree Unit (CTU, $128 \times 128$) quantization grid with $\Delta QP \in \{0, +4, +8\}$ assignment and convex hull boundaries.

---

## 🎬 Generating a Comparison Video

To generate a multi-panel comparison video for any video sequence:

```bash
python tools/visualization/visualize_comparison.py \
    --rgb_folder path/to/extracted_frames/ \
    --gt_folder path/to/groundtruth_fixation_maps/ \
    --pred_folder path/to/paven_saliency_predictions/ \
    --output output_comparison_4k.mp4 \
    --fps 25 \
    --resolution 3840 2160
```

The script automatically synchronizes frames, computes coincidence contours, draws the CTU grid with delta-QP values, and renders a 4K broadcast-quality MP4 video.
