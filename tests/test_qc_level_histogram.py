"""The per-level breakdown of where each block's shift came from.

Without it a field mostly extrapolated from coarse levels reads identically to
a measured one in the QC summary.
"""

import numpy as np
import pytest

from palom.align_multi_obj import MultiObjAligner
from palom.align_multi_res import MultiResAligner


class _FakeRung:
    def __init__(self, shifts, valid, errors=None):
        self.shifts = np.asarray(shifts, dtype=float)
        self.original_shifts = np.where(
            np.asarray(valid)[:, None], self.shifts, self.shifts + 99
        )
        self.shift_errors = (
            np.zeros(len(self.shifts)) if errors is None else np.asarray(errors, float)
        )
        self.grid_shape = (1, len(self.shifts))

    def constrain_shifts(self):
        pass


class _StubAligner(MultiResAligner):
    """`block_footprints` / `downsample_factors` are read-only properties.

    Both are pinned to 1:1 so every level shares the finest grid and no shift
    is rescaled -- the cross-level pick is what is under test here, not the
    footprint mapping.
    """

    def __init__(self, valid_per_level, n):
        self.aligners = [
            _FakeRung(np.zeros((n, 2)), valid) for valid in valid_per_level
        ]
        self.levels = list(range(len(valid_per_level)))

    @property
    def block_footprints(self):
        return [(1, 1)] * len(self.aligners)

    @property
    def downsample_factors(self):
        return [1] * len(self.aligners)


def _aligner(valid_per_level, n=4):
    return _StubAligner(valid_per_level, n)


# --- result_levels ----------------------------------------------------------


def test_finest_valid_level_wins():
    mr = _aligner([[True, False, False, False], [True, True, True, False]])
    mr.constrain_shifts()
    # block 0 valid at both -> level 0; blocks 1,2 only at level 1
    assert mr.result_levels.ravel().tolist() == [0, 1, 1, -1]


def test_blocks_no_level_resolved_are_marked_minus_one():
    """`argmax` of an all-False column returns 0; that must not read as level 0."""
    mr = _aligner([[False, False], [False, False]], n=2)
    mr.constrain_shifts()
    assert mr.result_levels.ravel().tolist() == [-1, -1]


def test_level_histogram_counts_and_respects_mask():
    mr = _aligner([[True, False, False, False], [True, True, True, False]])
    mr.constrain_shifts()
    assert mr.level_histogram() == {-1: 1, 0: 1, 1: 2}
    # restricted to the first two blocks
    mask = np.array([True, True, False, False])
    assert mr.level_histogram(mask) == {0: 1, 1: 1}


# --- formatting -------------------------------------------------------------


@pytest.mark.parametrize(
    "hist, expected",
    [
        ({0: 100}, "100"),
        ({0: 90, 1: 10}, "90/10"),
        ({0: 50, 1: 25, 2: 25}, "50/25/25"),
        ({0: 8, 1: 1, -1: 1}, "80/10 +10!"),
        ({-1: 4}, " +100!"),
        ({}, "-"),
        (None, "-"),
    ],
)
def test_format_levels(hist, expected):
    assert MultiObjAligner._format_levels(hist) == expected


def test_summary_row_carries_the_histogram():
    mo = MultiObjAligner.__new__(MultiObjAligner)
    mo.object_qc = [
        {
            "object": 0, "label": 1, "n_blocks": 10, "affine": "object+refine",
            "preferred": "object+refine", "scores": {"object+refine": 0.31},
            "refine": None, "shift_median": 1.2, "shift_max": 9.0,
            "levels": {0: 8, 1: 1, -1: 1}, "plot_failed": False,
        }
    ]
    out = mo.qc_summary()
    assert "lvl%" in out
    assert "80/10 +10!" in out
    assert "no level resolved" in out
