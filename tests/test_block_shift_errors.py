"""The per-block phase-correlation confidence, which used to be discarded.

`block_shifts` computed `-log(peak / total_amplitude)` and dropped it, so
`constrain_block_shifts`' two triangle thresholds were the only thing deciding
which blocks to trust. This propagates it; nothing acts on it yet.
"""

import cv2
import dask.array as da
import numpy as np
import pytest

from palom import align, register
from palom.align_multi_res import MultiResAligner

from test_qc_level_histogram import _StubAligner, _aligner  # noqa: F401


def _tiles(shape=(64, 64), grid=(2, 2), seed=0, shift=(0, 0)):
    """A reference image and a copy displaced by `shift`, chunked into blocks."""
    rng = np.random.default_rng(seed)
    h, w = shape[0] * grid[0], shape[1] * grid[1]
    ref = cv2.GaussianBlur(
        rng.integers(0, 255, (h, w)).astype("float32"), (0, 0), 1.5
    )
    moving = np.roll(ref, shift, axis=(0, 1))
    return (
        da.from_array(ref, chunks=shape),
        da.from_array(moving, chunks=shape),
    )


# --- shape and ordering -----------------------------------------------------


def test_block_shifts_emits_three_values_per_block():
    ref, moving = _tiles(grid=(3, 2))
    out = align.block_shifts(ref, moving).compute().reshape(-1, 3)
    assert out.shape == (6, 3)


def test_error_column_is_finite_for_a_matchable_block():
    ref, moving = _tiles()
    out = align.block_shifts(ref, moving).compute().reshape(-1, 3)
    assert np.isfinite(out[:, 2]).all()


def test_masked_blocks_report_infinite_shift_and_error():
    ref, moving = _tiles(grid=(2, 2))
    mask = da.from_array(np.array([[True, False], [True, True]]), chunks=1)
    out = align.block_shifts(ref, moving, mask=mask).compute().reshape(-1, 3)
    assert np.isinf(out[1]).all()          # the masked block, row-major
    assert np.isfinite(out[[0, 2, 3]]).all()


def test_a_constant_block_is_reported_infinite_not_confidently_zero():
    """`phase_cross_correlation`'s constant-image guard, surfaced."""
    (shift, error) = register.phase_cross_correlation(
        np.zeros((32, 32), "float32"), np.zeros((32, 32), "float32"), sigma=0
    )
    assert np.isinf(error) and np.isinf(shift).all()


# --- the error means what the docstring says --------------------------------


def test_a_worse_match_scores_a_higher_error():
    """Lower is better: -log(peak / amplitude)."""
    ref, aligned = _tiles(shift=(0, 0))
    _, displaced = _tiles(shift=(17, 11))
    good = align.block_shifts(ref, aligned).compute().reshape(-1, 3)[:, 2]
    # correlate the reference against an unrelated field
    rng = np.random.default_rng(99)
    junk = da.from_array(
        cv2.GaussianBlur(
            rng.integers(0, 255, ref.shape).astype("float32"), (0, 0), 1.5
        ),
        chunks=ref.chunksize,
    )
    bad = align.block_shifts(ref, junk).compute().reshape(-1, 3)[:, 2]
    assert np.median(good) < np.median(bad)
    del displaced


# --- Aligner exposes it alongside the shifts --------------------------------


class _MiniAligner(align.Aligner):
    def __init__(self, ref, moving):
        self.ref_img, self._moving = ref, moving

    @property
    def affine_matrix(self):
        return np.eye(3)

    def affine_transformed_moving_img(self, mxs=None):
        return self._moving


def test_aligner_splits_shifts_and_errors():
    ref, moving = _tiles(grid=(2, 3))
    al = _MiniAligner(ref, moving)
    al.compute_shifts()
    assert al.shifts.shape == (6, 2)
    assert al.shift_errors.shape == (6,)
    assert np.isfinite(al.shift_errors).all()


def test_shifts_stay_two_wide_for_existing_consumers():
    """`constrain_block_shifts`, the warps and the QC plots all take (N, 2)."""
    ref, moving = _tiles()
    al = _MiniAligner(ref, moving)
    al.compute_shifts()
    assert align.constrain_block_shifts(al.shifts, (2, 2)).shape == (4, 2)


# --- the cross-rung pick carries the winner's error -------------------------


def test_multires_takes_the_error_from_the_rung_that_won():
    mr = _StubAligner([[True, False], [True, True]], 2)
    mr.aligners[0].shift_errors = np.array([1.0, 9.0])
    mr.aligners[1].shift_errors = np.array([7.0, 2.0])
    mr.constrain_shifts()
    # block 0 -> rung 0 (valid, finest), block 1 -> rung 1
    assert mr.result_levels.ravel().tolist() == [0, 1]
    assert mr.shift_errors.tolist() == [1.0, 2.0]


def test_multires_error_is_infinite_where_no_rung_resolved():
    mr = _StubAligner([[False, False], [False, False]], 2)
    mr.aligners[0].shift_errors = np.array([3.0, 3.0])
    mr.constrain_shifts()
    assert np.isinf(mr.shift_errors).all()


def test_multires_errors_are_not_rescaled_across_levels():
    """A correlation confidence is not a length; rescaling it would be wrong."""
    mr = _StubAligner([[False, False], [True, True]], 2)
    mr.aligners[1].shift_errors = np.array([4.0, 6.0])
    mr.constrain_shifts()
    assert mr.shift_errors.tolist() == [4.0, 6.0]
