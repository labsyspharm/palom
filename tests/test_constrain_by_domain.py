"""Fitting one plane per domain instead of one over the whole grid.

`constrain_block_shifts` assumes a single trend. Where the field has two --
two tissue pieces at different rigid offsets, or the two sides of a stitching
seam -- `threshold_triangle` assumes a unimodal residual histogram, lands
between the modes, and overwrites the minority with the majority's plane.
"""

import numpy as np
import pytest

from palom.align import (
    constrain_block_shifts,
    constrain_block_shifts_by_domain,
)

OFFSET = (60.0, 30.0)


def _two_domains(grid=(20, 20), minority=0.3, noise=1.0, seed=0):
    h, w = grid
    cut = int(w * (1 - minority))
    field = np.zeros((h, w, 2))
    field[:, cut:] = OFFSET
    field += np.random.default_rng(seed).normal(0, noise, field.shape)
    return field.reshape(-1, 2), grid, cut


def _minority_error(out, grid, cut):
    h, w = grid
    return np.linalg.norm(
        out.reshape(h, w, 2)[:, cut:] - OFFSET, axis=-1
    ).mean()


# --- the failure, and the fix -----------------------------------------------


@pytest.mark.parametrize("minority", [0.5, 0.35, 0.25, 0.15, 0.08])
def test_whole_grid_destroys_the_minority_domain(minority):
    shifts, grid, cut = _two_domains(minority=minority)
    out = constrain_block_shifts(shifts, grid)
    # the full offset is lost -- not degraded, erased
    assert _minority_error(out, grid, cut) > 0.9 * np.linalg.norm(OFFSET)


@pytest.mark.parametrize("minority", [0.5, 0.35, 0.25, 0.15, 0.08])
def test_per_domain_keeps_it(minority):
    shifts, grid, cut = _two_domains(minority=minority)
    out = constrain_block_shifts_by_domain(shifts, grid)
    assert _minority_error(out, grid, cut) < 3.0


# --- it must not change what already worked ---------------------------------


def test_a_single_domain_is_unchanged_by_the_split():
    """One trend means one domain; the two paths must agree."""
    rng = np.random.default_rng(1)
    h, w = 12, 12
    field = np.stack(np.meshgrid(np.arange(w) * 1.5, np.arange(h) * 1.5), -1)
    shifts = (field + rng.normal(0, 0.5, field.shape)).reshape(-1, 2)
    assert np.allclose(
        constrain_block_shifts(shifts, (h, w)),
        constrain_block_shifts_by_domain(shifts, (h, w)),
    )


def test_outliers_inside_a_domain_are_still_replaced():
    """Per-domain must not mean per-domain-anything-goes."""
    shifts, grid, _ = _two_domains(minority=0.0, noise=0.5)
    shifts[5] = (400.0, 400.0)
    out = constrain_block_shifts_by_domain(shifts, grid)
    assert np.linalg.norm(out[5]) < 20.0


def test_falls_back_to_the_whole_grid_when_nothing_agrees():
    rng = np.random.default_rng(2)
    shifts = rng.normal(0, 300, (36, 2))
    assert np.allclose(
        constrain_block_shifts_by_domain(shifts, (6, 6)),
        constrain_block_shifts(shifts, (6, 6)),
    )


def test_a_two_block_domain_is_left_to_the_global_fit():
    """Under three blocks there is no plane, and two agreeing blocks are not
    enough to be trusted as a trend of their own."""
    field = np.zeros((6, 6, 2))
    field[0, :2] = (90.0, 90.0)
    shifts = field.reshape(-1, 2)
    out = constrain_block_shifts_by_domain(shifts, (6, 6))
    assert not np.allclose(out.reshape(6, 6, 2)[0, :2], (90.0, 90.0))


# --- non-finite handling carries over ---------------------------------------


def test_infinite_blocks_do_not_join_a_domain():
    shifts, grid, cut = _two_domains(minority=0.3)
    shifts[0] = (np.inf, np.inf)
    out = constrain_block_shifts_by_domain(shifts, grid)
    assert _minority_error(out, grid, cut) < 3.0


def test_output_shape_and_dtype_are_preserved():
    shifts, grid, _ = _two_domains()
    out = constrain_block_shifts_by_domain(shifts, grid)
    assert out.shape == shifts.shape
    assert np.isfinite(out).all()


# --- the threshold is global even though the planes are not ------------------


def test_thresholding_stays_as_permissive_as_the_whole_grid_path():
    """Per-domain planes with per-domain thresholds reject far too much.

    `threshold_triangle` scales to whatever spread it is handed, so computing
    it inside a domain -- where residuals are tight by construction -- rejects
    blocks that are barely off. On the reference slides that dropped a
    single-domain control's finest-level share from 89% to 39%.
    """
    rng = np.random.default_rng(4)
    h, w = 16, 16
    field = np.zeros((h, w, 2))
    field[:, 10:] = OFFSET
    field += rng.normal(0, 1.0, field.shape)
    shifts = field.reshape(-1, 2)

    changed = lambda out: (~np.isclose(out, shifts).all(axis=1)).sum()  # noqa: E731
    per_domain = constrain_block_shifts_by_domain(shifts, (h, w))
    whole_grid_on_one_domain = constrain_block_shifts(
        field[:, :10].reshape(-1, 2), (h, 10)
    )
    rejected_rate = changed(per_domain) / len(shifts)
    baseline_rate = (
        ~np.isclose(whole_grid_on_one_domain, field[:, :10].reshape(-1, 2))
        .all(axis=1)
    ).sum() / (h * 10)
    # within a factor of ~3 of what one domain alone would reject
    assert rejected_rate < max(0.05, 3 * baseline_rate)


def test_a_domain_too_small_to_fit_falls_back_to_the_global_plane():
    field = np.zeros((6, 6, 2))
    field[0, :2] = (90.0, 90.0)          # two blocks: no plane
    out = constrain_block_shifts_by_domain(field.reshape(-1, 2), (6, 6))
    assert np.isfinite(out).all()
