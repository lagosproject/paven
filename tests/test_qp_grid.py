import pytest
import numpy as np
from paven.qp_grid import QpGrid


def test_qp_grid_dimensions():
    """Verify CTU grid dimensions for 1080p and 4K sequences."""
    # 1080p: ceil(1920/128) = 15, ceil(1080/128) = 9
    grid_1080p = QpGrid(img_size=(1920, 1080), frame_count=1)
    assert grid_1080p.grid_width == 15
    assert grid_1080p.grid_height == 9

    # 4K: ceil(3840/128) = 30, ceil(2160/128) = 17
    grid_4k = QpGrid(img_size=(3840, 2160), frame_count=1)
    assert grid_4k.grid_width == 30
    assert grid_4k.grid_height == 17


def test_qp_grid_uniform_background():
    """Verify that an image with zero saliency receives uniform Level 3 (+8) Delta-QP."""
    grid_gen = QpGrid(img_size=(1920, 1080), frame_count=1)
    empty_saliency = np.zeros((1080, 1920), dtype=np.uint8)
    qp_matrix = grid_gen.compute_frame_grid(empty_saliency)

    # All CTUs should be assigned Delta-QP = +8
    assert np.all(qp_matrix == 8)


def test_qp_grid_foveal_hotspot():
    """Verify that a prominent central saliency hotspot creates Level 1 (0) and Level 2 (+4) rings."""
    grid_gen = QpGrid(img_size=(1920, 1080), frame_count=1)
    H, W = 1080, 1920
    y, x = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    hotspot = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 120**2))
    saliency = (hotspot * 255).astype(np.uint8)

    qp_matrix = grid_gen.compute_frame_grid(saliency)

    # Must contain all 3 hierarchical quantization levels: 0 (fovea), 4 (dilation), 8 (background)
    unique_qps = set(np.unique(qp_matrix))
    assert 0 in unique_qps, "Level 1 (Delta QP = 0) missing from hotspot"
    assert 4 in unique_qps, "Level 2 (Delta QP = 4) missing from dilation"
    assert 8 in unique_qps, "Level 3 (Delta QP = 8) missing from background"

    # Center CTU must be Level 1 (0)
    center_ctu_y = (H // 2) // 128
    center_ctu_x = (W // 2) // 128
    assert qp_matrix[center_ctu_y, center_ctu_x] == 0


def test_qp_line_formatting():
    """Verify that output line formatting terminates with a comma and matches raster order."""
    grid_gen = QpGrid(img_size=(384, 256), frame_count=2)
    saliency = np.zeros((256, 384), dtype=np.uint8)
    grid_gen.process_image(saliency, frame_idx=0)

    line = grid_gen.qp_lines[0]
    assert line.endswith(",")
    # 384/128 = 3 CTUs width, 256/128 = 2 CTUs height -> 6 CTUs total
    elements = [x.strip() for x in line.rstrip(',').split(',')]
    assert len(elements) == 6
    assert all(e == "8" for e in elements)


def test_qp_grid_ablation_modes():
    """Verify 2-level mode and non-convex hull mode."""
    H, W = 1080, 1920
    y, x = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    hotspot = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 120**2))
    saliency = (hotspot * 255).astype(np.uint8)

    # 2-levels mode: only 0 (fovea) and 8 (background), no 4 (dilation)
    grid_2lvl = QpGrid(img_size=(W, H), frame_count=1, num_levels=2)
    qp_2lvl = grid_2lvl.compute_frame_grid(saliency)
    unique_2lvl = set(np.unique(qp_2lvl))
    assert 0 in unique_2lvl
    assert 8 in unique_2lvl
    assert 4 not in unique_2lvl

    # No convex hull mode: should execute cleanly
    grid_nohull = QpGrid(img_size=(W, H), frame_count=1, use_convex_hull=False)
    qp_nohull = grid_nohull.compute_frame_grid(saliency)
    assert qp_nohull.shape == (9, 15)
