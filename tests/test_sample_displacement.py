"""Sampling the per-pixel displacement field from the block-center grid.

The field used to be sampled with `cv2.remap`, whose 1/32-cell fixed-point
positions made it a staircase: flat for 32 output rows (with 1024px blocks),
then a jump. Across a domain step those jumps showed as seams in the warp.
"""

import numpy as np
import scipy.ndimage as ndi

from palom.align import _sample_displacement

BLOCK = 1024
GRID = (6, 7)
OUT_SHAPE = (GRID[0] * BLOCK, GRID[1] * BLOCK)


def _positions(rows, cols):
    gy = (rows + 0.5) * (GRID[0] / OUT_SHAPE[0]) - 0.5
    gx = (cols + 0.5) * (GRID[1] / OUT_SHAPE[1]) - 0.5
    return np.meshgrid(gy, gx, indexing="ij")


def test_linear_matches_float_map_coordinates():
    grid = np.random.default_rng(0).normal(0, 20, (*GRID, 2)).astype("float32")
    # spans the border cells too, where both replicate the edge
    rows = np.arange(0, OUT_SHAPE[0], 7, dtype="float64")
    cols = np.arange(0, OUT_SHAPE[1], 5, dtype="float64")
    dy, dx = _sample_displacement(rows, cols, grid, OUT_SHAPE)
    gy, gx = _positions(rows, cols)
    for got, k in ((dy, 0), (dx, 1)):
        want = ndi.map_coordinates(
            grid[..., k].astype("float64"), [gy, gx], order=1, mode="nearest"
        )
        np.testing.assert_allclose(got, want, atol=1e-4)


def test_domain_step_has_no_staircase():
    # a 128px dx step between block rows 2 and 3
    grid = np.zeros((*GRID, 2), dtype="float32")
    grid[3:, :, 1] = 128
    rows = np.arange(2 * BLOCK + BLOCK // 2, 3 * BLOCK + BLOCK // 2, dtype="float64")
    _, dx = _sample_displacement(rows, np.array([0.0]), grid, OUT_SHAPE)
    steps = np.diff(dx[:, 0])
    # between the two block centers the field must rise every row by the same
    # 128/1024 px -- not hold still and then jump
    np.testing.assert_allclose(steps, 128 / BLOCK, atol=1e-4)
