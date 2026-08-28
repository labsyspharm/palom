"""Partitioning a block-shift field into domains of agreeing shifts.

The two cases that matter are built explicitly: two tissue pieces separated by
background (a gap of invalid blocks), and a stitching seam running through the
middle of one connected piece with no gap at all. Morphology can find the first
and not the second; agreement finds both.
"""

import numpy as np
import pytest

from palom import shift_domains as sd


def _field(rows):
    """Build (shifts, grid_shape) from a grid of (dy, dx) tuples or None."""
    h, w = len(rows), len(rows[0])
    out = np.full((h, w, 2), np.nan)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            if v is not None:
                out[r, c] = v
    return out.reshape(-1, 2), (h, w)


A, B = (0.0, 0.0), (80.0, 40.0)


# --- the two real cases -----------------------------------------------------


def test_two_pieces_separated_by_background():
    shifts, grid = _field([
        [A, A, None, B, B],
        [A, A, None, B, B],
    ])
    labels = sd.label_domains(shifts, grid)
    assert set(np.unique(labels)) == {sd.LOOSE, 0, 1}
    assert (labels[:, :2] == labels[0, 0]).all()
    assert (labels[:, 3:] == labels[0, 3]).all()
    assert labels[0, 0] != labels[0, 3]
    assert (labels[:, 2] == sd.LOOSE).all()


def test_a_seam_splits_one_connected_piece():
    """No gap, no morphological cue -- only the shifts disagree."""
    shifts, grid = _field([
        [A, A, B, B],
        [A, A, B, B],
        [A, A, B, B],
    ])
    labels = sd.label_domains(shifts, grid)
    assert len(set(np.unique(labels)) - {sd.LOOSE}) == 2
    assert (labels[:, :2] == labels[0, 0]).all()
    assert (labels[:, 2:] == labels[0, 2]).all()
    assert labels[0, 1] != labels[0, 2]


def test_a_uniform_field_is_one_domain():
    shifts, grid = _field([[A] * 4] * 3)
    labels = sd.label_domains(shifts, grid)
    assert (labels == 0).all()
    assert sd.max_separation(shifts, labels) == 0.0


def test_smooth_deformation_is_not_split():
    """Genuine local deformation drifts gradually; neighbours still agree."""
    h, w = 6, 6
    field = np.stack(np.meshgrid(np.arange(w) * 2.0, np.arange(h) * 2.0), -1)
    shifts = field.reshape(-1, 2)
    labels = sd.label_domains(shifts, (h, w), tol=15.0)
    assert (labels == 0).all()


# --- the tolerance ----------------------------------------------------------

@pytest.mark.parametrize("tol, n_domains", [(1.0, 2), (50.0, 1)])
def test_tolerance_decides_where_a_gradient_breaks(tol, n_domains):
    shifts, grid = _field([[(0.0, 0.0), (0.0, 10.0)]])
    labels = sd.label_domains(shifts, grid, tol=tol, min_size=1)
    assert len(set(np.unique(labels)) - {sd.LOOSE}) == n_domains


def test_separation_is_measured_between_domain_offsets():
    shifts, grid = _field([[A, A, None, B, B]])
    labels = sd.label_domains(shifts, grid)
    assert sd.max_separation(shifts, labels) == pytest.approx(np.hypot(80, 40))


# --- trust ------------------------------------------------------------------


def test_a_lone_block_is_not_a_domain():
    shifts, grid = _field([
        [A, A, None, B],
        [A, A, None, None],
    ])
    labels = sd.label_domains(shifts, grid)
    assert (labels == 0).sum() == 4
    assert labels[0, 3] == sd.LOOSE      # B has no corroborating neighbour


def test_min_size_can_admit_singletons():
    shifts, grid = _field([[A, None, B]])
    labels = sd.label_domains(shifts, grid, min_size=1)
    assert len(set(np.unique(labels)) - {sd.LOOSE}) == 2


def test_non_finite_blocks_are_loose():
    shifts, grid = _field([[A, A, A]])
    shifts[1] = (np.inf, np.inf)
    labels = sd.label_domains(shifts, grid)
    assert labels.ravel()[1] == sd.LOOSE


def test_explicit_valid_mask_is_honoured():
    shifts, grid = _field([[A, A, A, A]])
    valid = np.array([[True, True, False, True]])
    labels = sd.label_domains(shifts, grid, valid=valid)
    assert labels.ravel()[2] == sd.LOOSE


def test_domain_zero_is_the_largest():
    shifts, grid = _field([
        [B, B, None, A, A],
        [None, None, None, A, A],
    ])
    labels = sd.label_domains(shifts, grid)
    assert (labels == 0).sum() == 4      # the A block, not the B one
    assert np.allclose(sd.domain_offsets(shifts, labels)[0], A)


def test_everything_loose_when_nothing_agrees():
    shifts, grid = _field([[(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]])
    labels = sd.label_domains(shifts, grid)
    assert (labels == sd.LOOSE).all()


# --- resolve_loose ----------------------------------------------------------


def test_loose_blocks_take_the_nearest_domain():
    shifts, grid = _field([[A, A, None, None, B, B]])
    labels = sd.label_domains(shifts, grid)
    filled = sd.resolve_loose(labels)
    assert sd.LOOSE not in filled
    assert filled.ravel().tolist() == [0, 0, 0, 1, 1, 1]


def test_resolve_loose_is_a_no_op_without_loose_blocks():
    shifts, grid = _field([[A, A]])
    labels = sd.label_domains(shifts, grid)
    assert (sd.resolve_loose(labels) == labels).all()


def test_resolve_loose_survives_an_all_loose_field():
    labels = np.full((2, 2), sd.LOOSE)
    assert (sd.resolve_loose(labels) == sd.LOOSE).all()


# --- offsets and summary ----------------------------------------------------


def test_offsets_use_the_median_not_the_mean():
    """One rim block across a seam must not drag a domain's offset."""
    shifts, grid = _field([[A, A, A, (900.0, 900.0)]])
    labels = np.zeros(grid, dtype=int)
    assert np.allclose(sd.domain_offsets(shifts, labels)[0], A)


def test_summary_reports_blocks_offset_spread_and_loose():
    shifts, grid = _field([
        [A, A, None, B, B],
        [A, A, None, B, B],
    ])
    labels = sd.label_domains(shifts, grid)
    s = sd.summarize(shifts, labels)
    assert s["n_loose"] == 2
    assert [r["blocks"] for r in s["domains"]] == [4, 4]
    assert all(r["spread"] == 0.0 for r in s["domains"])
    assert s["coverage"] == pytest.approx(8 / 10)


def test_summary_accounting_is_bounded_by_within():
    """Without it, every background block in the grid counts as loose."""
    shifts, grid = _field([
        [A, A, None, None],
        [A, A, None, None],
    ])
    labels = sd.label_domains(shifts, grid)
    within = np.zeros(grid, bool)
    within[:, :2] = True                      # only the tissue blocks
    s = sd.summarize(shifts, labels, within=within)
    assert s["n_blocks"] == 4 and s["n_loose"] == 0
    assert s["coverage"] == 1.0
    # unbounded, the four background blocks land in the loose count
    assert sd.summarize(shifts, labels)["n_loose"] == 4


def test_coverage_is_low_when_agreement_finds_only_islands():
    """The 23390 case: a mostly-unresolved field fragments into noise."""
    rng = np.random.default_rng(3)
    shifts = rng.normal(0, 200, (64, 2))
    s = sd.summarize(shifts, sd.label_domains(shifts, (8, 8)))
    assert s["coverage"] < 0.5


def test_spread_measures_residual_deformation_within_a_domain():
    shifts, grid = _field([[(0.0, 0.0), (0.0, 4.0), (0.0, -4.0), (0.0, 0.0)]])
    labels = np.zeros(grid, dtype=int)
    assert sd.summarize(shifts, labels)["domains"][0]["spread"] == pytest.approx(2.0)


# --- the QC summary surfaces a split ----------------------------------------


def test_qc_note_appears_only_when_domains_disagree():
    from palom.align_multi_obj import MultiObjAligner

    def row(**dom):
        return dict(
            object=0, label=1, n_blocks=100, affine="object", preferred="object",
            scores={"object": 0.4}, refine=None, shift_median=1.0, shift_max=2.0,
            levels={0: 100}, plot_failed=False, domains=dom,
        )

    mo = MultiObjAligner.__new__(MultiObjAligner)
    mo.object_qc = [row(domains=[1, 2, 3], separation=100.2, coverage=0.93)]
    assert "3 domains, max sep 100px, 93% covered" in mo.qc_summary()

    # one domain, or domains that agree, is not worth a note
    mo.object_qc = [row(domains=[1], separation=0.0, coverage=1.0)]
    assert "domains, max sep" not in mo.qc_summary()
    mo.object_qc = [row(domains=[1, 2], separation=0.0, coverage=1.0)]
    assert "domains, max sep" not in mo.qc_summary()


def test_qc_summary_survives_rows_without_domain_info():
    """Rows built by hand, and any older QC row, have no `domains` key."""
    from palom.align_multi_obj import MultiObjAligner

    mo = MultiObjAligner.__new__(MultiObjAligner)
    mo.object_qc = [dict(
        object=0, label=1, n_blocks=10, affine="baseline", preferred="baseline",
        scores={"baseline": 0.3}, refine=None, shift_median=1.0, shift_max=2.0,
        levels={0: 10}, plot_failed=False,
    )]
    assert "baseline" in mo.qc_summary()


def test_low_coverage_is_flagged_for_review_not_failed():
    """Tissue lost between scans reads as low coverage; the run still proceeds."""
    from palom.align_multi_obj import MultiObjAligner

    def row(cov, dom):
        return dict(
            object=0, label=1, n_blocks=300, affine="object", preferred="object",
            scores={"object": 0.3}, refine=None, shift_median=5.0, shift_max=50.0,
            levels={0: 300}, plot_failed=False,
            domains=dict(domains=dom, separation=99.0, coverage=cov),
        )

    mo = MultiObjAligner.__new__(MultiObjAligner)
    mo.object_qc = [row(0.37, list(range(19)))]
    out = mo.qc_summary()
    assert "REVIEW: shift field mostly unresolved" in out
    assert "1 for review" in out

    mo.object_qc = [row(0.93, [1, 2, 3])]
    assert "REVIEW" not in mo.qc_summary()


def test_coverage_of_zero_is_not_flagged_twice():
    """An object with no domains at all has coverage 0 and no partition to doubt."""
    from palom.align_multi_obj import MultiObjAligner

    mo = MultiObjAligner.__new__(MultiObjAligner)
    mo.object_qc = [dict(
        object=0, label=1, n_blocks=5, affine="baseline", preferred="baseline",
        scores={"baseline": 0.3}, refine=None, shift_median=1.0, shift_max=2.0,
        levels={0: 5}, plot_failed=False,
        domains=dict(domains=[], separation=0.0, coverage=0.0),
    )]
    assert "REVIEW" not in mo.qc_summary()
