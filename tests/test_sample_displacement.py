"""Sampling the per-pixel displacement field from the block-center grid.

The field used to be sampled with `cv2.remap`, whose 1/32-cell fixed-point
positions made it a staircase: flat for 32 output rows (with 1024px blocks),
then a jump. Across a domain step those jumps showed as seams in the warp.
"""

import cv2
import numpy as np
import pytest
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
    dy, dx = _sample_displacement(rows, cols, grid, OUT_SHAPE, cv2.INTER_LINEAR)
    gy, gx = _positions(rows, cols)
    for got, k in ((dy, 0), (dx, 1)):
        want = ndi.map_coordinates(
            grid[..., k].astype("float64"), [gy, gx], order=1, mode="nearest"
        )
        np.testing.assert_allclose(got, want, atol=1e-4)


@pytest.mark.parametrize("interp", [cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_domain_step_has_no_staircase(interp):
    # a 128px dx step between block rows 2 and 3
    grid = np.zeros((*GRID, 2), dtype="float32")
    grid[3:, :, 1] = 128
    rows = np.arange(2 * BLOCK + BLOCK // 2, 3 * BLOCK + BLOCK // 2, dtype="float64")
    _, dx = _sample_displacement(rows, np.array([0.0]), grid, OUT_SHAPE, interp)
    steps = np.diff(dx[:, 0])
    # between the two block centers the field must rise every row, and by the
    # same 128/1024 px for bilinear -- not hold still and then jump
    assert (steps > 0).all()
    if interp == cv2.INTER_LINEAR:
        np.testing.assert_allclose(steps, 128 / BLOCK, atol=1e-4)
    assert steps.max() < 1.0


@pytest.mark.parametrize("interp", [cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_same_kernel_as_cv2_where_cv2_is_exact(interp):
    # cv2.remap is exact at multiples of 1/32 cell -- here every 32nd output
    # row/col, offset by half a pixel -- so there the two must agree
    grid = np.random.default_rng(1).normal(0, 20, (*GRID, 2)).astype("float32")
    rows = np.arange(0, OUT_SHAPE[0], 32, dtype="float64") - 0.5
    cols = np.arange(0, OUT_SHAPE[1], 32, dtype="float64") - 0.5
    dy, dx = _sample_displacement(rows, cols, grid, OUT_SHAPE, interp)
    gy, gx = _positions(rows, cols)
    for got, k in ((dy, 0), (dx, 1)):
        want = cv2.remap(
            np.ascontiguousarray(grid[..., k]),
            gx.astype("float32"), gy.astype("float32"),
            interp, borderMode=cv2.BORDER_REPLICATE,
        )
        np.testing.assert_allclose(got, want, atol=1e-3)
