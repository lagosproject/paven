# Experimental Analysis: Impact of Color Spaces on Deep Convolutional Networks for Video Saliency

This directory documents the experimental findings conducted on the **CeSViMa Magerit Supercomputing Cluster** (as detailed in Section 4.2 and Table 4.1 of Pablo Fernández Lagos's Master Thesis, UPM 2025).

## Motivation

Standard video codecs (VVC/H.266, HEVC/H.265, AVC/H.264) store video bitstreams in planar $YUV$ color formats, typically $YUV4:2:0$, where chrominance components ($U, V$) are subsampled to half the spatial resolution of the luminance channel ($Y$).

However, state-of-the-art vision and saliency models (such as S3D, ViNet, MobileNet, EfficientNet) are pre-trained on natural RGB imagery from large-scale visual benchmarks (ImageNet, Kinetics). Feeding raw planar $YUV$ directly into RGB-pre-trained convolutional backbones produces severe accuracy degradation and undesirable inductive biases:

1. **Massive Accuracy Drop:** Directly passing $YUV$ frames without architectural adaptation causes a performance drop of over $40\text{--}50\%$ in classification accuracy.
2. **Spurious Color Bias:** When $YUV$ components are mapped onto RGB channels, the predominant energy in the blue/cyan chrominance subspace causes the network to heavily favor marine and aquatic classifications (e.g., categorizing diverse natural scenes as ocean or underwater environments).

## Empirical Evidence (Table 4.1 from Master Thesis)

The table below summarizes top-1 and top-5 classification accuracy across 1,000 validation images from ImageNet evaluated on MobileNetV2 and EfficientNetV2-B0 under different color space representations:

| Format / Representation | MobileNetV2 Top-1 (%) | EfficientNetV2-B0 Top-1 (%) | MobileNetV2 Top-5 (%) | EfficientNetV2-B0 Top-5 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **RGB (Original baseline)** | 77.80% | 85.69% | 94.30% | 97.20% |
| **YUV (Direct planar conversion)** | 25.80% | 46.85% | 49.24% | 70.17% |
| **YYY (PAVEN Luminance Replication)** | **56.67%** | **77.87%** | **79.93%** | **95.12%** |

## Key Insights and Architectural Decision

- **Why YYY works:** The luminance component ($Y$) carries virtually all high-frequency spatial gradients, edges, and texture information to which human visual perception (foveal vision) is most sensitive. By replicating the luminance channel across the three input channels ($Y \rightarrow YYY$), the input distribution matches the expected input shape $(3, T, H, W)$ while avoiding the chromatic confusion introduced by un-normalized subsampled chroma planes.
- **Top-5 Performance Recovery:** In EfficientNetV2, the top-5 accuracy under $YYY$ reaches **95.12%**, virtually matching the original RGB baseline of 97.20%.
- **PAVEN Design:** Consequently, the PAVEN neural network is designed and retrained with the $YYY$ representation, eliminating the need for slow multi-channel color transformations while maintaining peak visual saliency prediction performance.

## Reproduction Script

To evaluate and reproduce these metrics on your local environment:
```bash
python evaluate_color_spaces.py
```
