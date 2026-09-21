"""
PAVEN: Perceptual Quantization Grid Generator (QpGrid)
Paper: "PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks"
       (Elsevier EAAI, 2025, DOI: 10.1016/j.engappai.2025.111664)
Author: Pablo Fernández Lagos (UPM Master Thesis, Chapter 5, 2025)

Transforms continuous 2D visual saliency maps into discrete CTU-level (128x128) Delta-QP
matrices for video encoders (VTM, HM, x265, SVT-AV1).
"""

import cv2
import math
import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Union
from shapely.geometry import Polygon

# Constants defined in TFM Chapter 5
DEFAULT_CELL_SIZE: int = 128
DEFAULT_SEED_THRESHOLD: int = 70
DEFAULT_LEVEL2_MULTIPLIER: int = 2
DEFAULT_MAX_QP_DIFF: int = 8
DIRECTIONS_4: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS_8: List[Tuple[int, int]] = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]


class QpGrid:
    """
    Quantization Parameter (QP) Grid Generator for Perceptual Video Coding.

    Implements the 3-level hierarchical QP modulation algorithm:
      - Level 1 (Fovea / Highest Attention): Delta QP = 0 (Preserve maximum visual fidelity)
      - Level 2 (Periphery / Saccadic Transition): Delta QP = +4 (Intermediate compression)
      - Level 3 (Background / Low Attention): Delta QP = +8 (Aggressive rate reduction)
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        qp_grid_vis_path: Optional[str] = None,
        qp_target: int = 0,
        img_size: Tuple[int, int] = (1920, 1080),
        frame_count: int = 0,
        cell_size: int = DEFAULT_CELL_SIZE,
        qp_max_difference: int = DEFAULT_MAX_QP_DIFF,
        seed_threshold: int = DEFAULT_SEED_THRESHOLD,
        level2_multiplier: int = DEFAULT_LEVEL2_MULTIPLIER,
        num_levels: int = 3,
        use_convex_hull: bool = True,
        verbose: int = 0,
    ):
        """
        Initialize the QP Grid generator.

        Args:
            output_path: Target path to write the final .qp file.
            qp_grid_vis_path: Optional directory to save debug visualization images.
            qp_target: Base QP offset (typically 0 when exporting Delta-QP matrices).
            img_size: Video frame resolution as (width, height).
            frame_count: Expected total number of frames in the video sequence.
            cell_size: Size of CTU block in pixels (default: 128).
            qp_max_difference: Maximum Delta-QP step (default: 8).
            seed_threshold: Saliency seed threshold (default: 70).
            level2_multiplier: Ring dilation multiplier (default: 2).
            num_levels: Number of modulation levels (3 or 2, default: 3).
            use_convex_hull: Whether to fill the convex hull of attention contours (default: True).
            verbose: Verbosity level (0, 1, or 2).
        """
        self.output_path = output_path
        self.qp_grid_vis_path = qp_grid_vis_path
        self.qp_target = qp_target
        self.img_size = img_size
        self.width, self.height = img_size
        self.frame_count = frame_count
        self.cell_size = cell_size
        self.cell_area = cell_size * cell_size
        self.qp_max_difference = qp_max_difference
        self.seed_threshold = seed_threshold
        self.level2_multiplier = level2_multiplier
        self.num_levels = num_levels
        self.use_convex_hull = use_convex_hull
        self.verbose = verbose

        # Level mapping: internal indices [0, 1, 2] -> delta QPs [+8, +4, 0]
        # index 0 (Background) -> +qp_max_difference (+8)
        # index 1 (Level 2)    -> +qp_max_difference // 2 (+4)
        # index 2 (Level 1)    -> 0
        self.levels = np.array([self.qp_max_difference, self.qp_max_difference // 2, 0])

        self.width, self.height = img_size
        self.grid_width = math.ceil(self.width / self.cell_size)
        self.grid_height = math.ceil(self.height / self.cell_size)

        self.qp_lines: List[str] = [""] * frame_count if frame_count > 0 else []

        if self.verbose >= 1:
            print(f"[QpGrid] Dimensions: {self.width}x{self.height} -> CTU Grid: {self.grid_width}x{self.grid_height}")

    def find_attention_points(
        self, saliency_map: np.ndarray
    ) -> Tuple[List, Optional[np.ndarray], List[np.ndarray]]:
        """
        Extract primary attention contours and convex hull enclosing multiple focal points.
        """
        # Ensure image is in 0-255 uint8 format
        if saliency_map.dtype != np.uint8:
            if saliency_map.max() <= 1.0:
                saliency_map = (saliency_map * 255.0).astype(np.uint8)
            else:
                saliency_map = np.clip(saliency_map, 0, 255).astype(np.uint8)

        _, binary_image = cv2.threshold(saliency_map, self.seed_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        if len(contours) == 0:
            return centroids, None, []

        if len(contours) == 1:
            return centroids, None, contours

        # Compute Convex Hull around all attention regions (TFM Section 5.1)
        combined_contour = np.concatenate(contours)
        hull = cv2.convexHull(combined_contour)
        return centroids, hull, contours

    def assign_contours_to_level1(
        self, contours: List[np.ndarray], hull: Optional[np.ndarray], grid: np.ndarray
    ) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
        """
        Assign Level 1 (Delta QP = 0) to CTUs intersecting attention contours or convex hull.
        """
        cell_polygons: Dict[Tuple[int, int], Polygon] = {
            (i, j): Polygon([
                (j * self.cell_size, i * self.cell_size),
                ((j + 1) * self.cell_size, i * self.cell_size),
                ((j + 1) * self.cell_size, (i + 1) * self.cell_size),
                (j * self.cell_size, (i + 1) * self.cell_size)
            ])
            for i in range(self.grid_height)
            for j in range(self.grid_width)
        }

        level1_cells: Set[Tuple[int, int]] = set()

        # Check intersection with individual focal contours (threshold: >= 1% of CTU area)
        for contour in contours:
            try:
                if len(contour) < 4:
                    continue
                contour_poly = Polygon(contour[:, 0, :])
                if not contour_poly.is_valid:
                    contour_poly = contour_poly.buffer(0)
                for (i, j), cell_poly in cell_polygons.items():
                    if (i, j) in level1_cells:
                        continue
                    intersection = cell_poly.intersection(contour_poly)
                    if intersection.area / self.cell_area >= 0.01:
                        grid[i, j] = 2  # Level 1 marker
                        level1_cells.add((i, j))
            except Exception as e:
                if self.verbose >= 2:
                    print(f"[QpGrid] Warning during contour processing: {e}")

        # Check intersection with overall convex hull (threshold: >= 25% of CTU area)
        if self.use_convex_hull and hull is not None:
            try:
                hull_poly = Polygon(hull[:, 0, :])
                if not hull_poly.is_valid:
                    hull_poly = hull_poly.buffer(0)
                for (i, j), cell_poly in cell_polygons.items():
                    if (i, j) in level1_cells:
                        continue
                    intersection = cell_poly.intersection(hull_poly)
                    if intersection.area / self.cell_area >= 0.25:
                        grid[i, j] = 2  # Level 1 marker
                        level1_cells.add((i, j))
            except Exception as e:
                if self.verbose >= 2:
                    print(f"[QpGrid] Warning during hull processing: {e}")

        return grid, level1_cells

    def assign_level2_to_grid(
        self, grid: np.ndarray, level1_cells: Set[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Assign Level 2 (Delta QP = +4) to peripheral dilation rings around Level 1 cells.
        Replicates Level 1 boundary cells according to LEVEL2_MULTIPLIER (Table 5.1 of TFM).
        """
        update_list = list(level1_cells)
        level2_cell_count = len(level1_cells)
        level1_cell_count = 0

        while level1_cell_count < level2_cell_count * self.level2_multiplier:
            new_update_list = []
            for cell in update_list:
                i, j = cell
                for dx, dy in DIRECTIONS_4:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width and grid[ni, nj] == 0:
                        grid[ni, nj] = 1
                        new_update_list.append((ni, nj))
            level1_cell_count += len(update_list)
            update_list = new_update_list
            if not update_list:
                break

        return grid

    def grid_to_qp(self, grid: np.ndarray) -> np.ndarray:
        """Convert internal level grid [0, 1, 2] to Delta-QP values [+8, +4, 0]."""
        return self.qp_target + self.levels[grid]

    def compute_frame_grid(self, saliency_map: np.ndarray) -> np.ndarray:
        """
        Compute the Delta-QP matrix for a single 2D saliency map.

        Returns:
            np.ndarray: 2D array of shape (grid_height, grid_width) with Delta-QP values.
        """
        # Resize saliency map if necessary
        if (saliency_map.shape[1], saliency_map.shape[0]) != (self.width, self.height):
            saliency_map = cv2.resize(saliency_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Apply 11x11 Gaussian smoothing (TFM Chapter 5)
        saliency_smoothed = cv2.GaussianBlur(saliency_map, (11, 11), 0)

        grid = np.zeros((self.grid_height, self.grid_width), dtype=int)
        _, hull, contours = self.find_attention_points(saliency_smoothed)
        grid, level1_cells = self.assign_contours_to_level1(contours, hull, grid)
        if self.num_levels >= 3:
            grid = self.assign_level2_to_grid(grid, level1_cells)

        return self.grid_to_qp(grid)

    def process_image(self, saliency_map: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Process a single video frame, store its formatted QP line, and optionally save visual debug.
        """
        qp_matrix = self.compute_frame_grid(saliency_map)

        # Format line strictly matching official codebase and VTM parser:
        # raster order: for j in range(width) for i in range(height)
        line = ', '.join(
            str(qp_matrix[i, j])
            for j in range(qp_matrix.shape[1])
            for i in range(qp_matrix.shape[0])
        ) + ","

        if frame_idx < len(self.qp_lines):
            self.qp_lines[frame_idx] = line
        else:
            self.qp_lines.append(line)

        return qp_matrix

    def save_qp_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Write all frame QP lines to output file.
        """
        target = filepath or self.output_path
        if not target:
            raise ValueError("No output path specified to save QP file.")

        with open(target, 'w') as f:
            f.write('\n'.join(self.qp_lines))
        return target
