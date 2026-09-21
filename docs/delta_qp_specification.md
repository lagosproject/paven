# Technical Specification: Delta-QP Configuration File Format (.qp)

Paper: *PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks*  
Elsevier *Engineering Applications of Artificial Intelligence*, 2025  
UPM Master Thesis: *Codificación Perceptual de Vídeos en el Estándar VVC Usando Técnicas Basadas en el Aprendizaje Profundo* (Pablo Fernández Lagos, 2025)

---

## 1. Overview and Core Philosophy

The PAVEN perceptual quantization framework intentionally decouples saliency map prediction and CTU-level quantization parameter modulation from the internal mechanics of reference video encoders (such as VTM for VVC, HM for HEVC, or x265/SVT-AV1).

Instead of producing absolute QP values (which would bind the file to a single fixed base QP), PAVEN outputs a **Delta-QP matrix** $\Delta QP \in \{0, +4, +8\}$.

```
Actual CTU QP = Base_QP + Delta_QP
```

This decoupled contract provides significant advantages:
- **Base QP Invariance:** A single `.qp` configuration file generated for a video sequence can be reused across any target operating point (e.g. QP 22, 27, 32, or 37) without recalculation.
- **Cross-Codec Portability:** Any compliant video encoder capable of reading external CTU or Macroblock quantization deltas can directly consume this format.

---

## 2. Mathematical Definition of Quantization Levels

Following the empirical findings in Chapter 5 of the Master Thesis, visual perception tolerates a maximum bounded delta difference of up to 8 units without incurring noticeable perceptual degradation:

| Level | Delta QP ($\Delta QP$) | Visual Region | Perceptual Objective |
| :---: | :---: | :---: | :--- |
| **Level 1** | $\mathbf{0}$ | **Foveal Center (Primary Attention)** | Full fidelity, lowest compression. Applied to CTUs intersecting $\ge 1\%$ of primary attention contours AND $\ge 25\%$ of the CTU area. |
| **Level 2** | $\mathbf{+4}$ | **Periphery & Saccadic Transition** | Intermediate compression. Formed by a 2-repetition morphological dilation ring around Level 1 cells, covering transitional gaze saccades. |
| **Level 3** | $\mathbf{+8}$ | **Background / Non-Salient** | Maximum compression, massive bitrate reduction. Assigned to all remaining CTUs. |

---

## 3. File Layout and Syntax

The `.qp` output is an ASCII text file with comma-delimited integer values.

### A. Frame Structure
- **Line Index:** Each line corresponds to one video frame in sequential display order (Picture Order Count, $\text{POC} = 0, 1, 2, \dots, N-1$).
- **Total Lines:** Equal to the total number of frames encoded in the video stream.

### B. CTU Raster Ordering
Within each line, the Delta-QP values for each Coding Tree Unit (CTU, typically $128 \times 128$ pixels) are listed in raster scan order (left-to-right, top-to-bottom):

```
Grid Dimensions:
  Grid_Width  = ceil(Frame_Width  / CTU_Size)
  Grid_Height = ceil(Frame_Height / CTU_Size)
  Total_CTUs  = Grid_Width * Grid_Height
```

For a video sequence with grid dimensions `(Grid_Height, Grid_Width)`:
- Value at index `z = x + y * Grid_Width` corresponds to the CTU at spatial coordinate $(x, y)$.

### C. Example
For a small sequence with `Grid_Height = 2`, `Grid_Width = 4` (8 CTUs per frame) and 3 frames:

```
0, 0, 4, 8, 4, 4, 8, 8,
0, 4, 4, 8, 4, 4, 8, 8,
4, 0, 0, 4, 8, 4, 8, 8,
```

- **Trailing comma:** In accordance with the reference parsing implementation, each line terminates with a trailing comma followed by a newline `\n`.
- **Parsing Ingestion:** In C++/Python encoders, lines are read sequentially using `getline(file, token, ',')`, mapped into an internal array `qp_matrix[poc][ctu_index]`, and added to the encoder's global slice QP:
  ```cpp
  int final_cu_qp = base_qp + delta_qp;
  ```
