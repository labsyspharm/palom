"""Fail-fast on a hopeless coarse fit, and batch survival of a bad row.

Calibrated in docs/08: over the 21 reference slides the committed ncc runs
0.009-0.60. The only slide that crashes scored 0.0093; the next lowest, 0.0590,
produced one of the best results in the set. So the floor is for the hopeless,
not for the merely weak.
"""

import csv

import cv2
import numpy as np
import pytest

import skimage.transform

from palom import block_affine, register_coarse, register_util
from palom.cli import align_he


@pytest.fixture
def textured():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (300, 400)).astype("float32")
    return cv2.GaussianBlur(img, (0, 0), 2)


# --- committed_ncc ----------------------------------------------------------


def test_identity_on_itself_scores_high(textured):
    assert register_coarse.committed_ncc(textured, textured, np.eye(3)) > 0.99


def test_unrelated_images_score_low(textured):
    rng = np.random.default_rng(8)
    other = cv2.GaussianBlur(
        rng.integers(0, 255, (300, 400)).astype("float32"), (0, 0), 2
    )
    assert register_coarse.committed_ncc(textured, other, np.eye(3)) < 0.1


def test_accepts_a_2x3_matrix(textured):
    """`coarse_register` returns 2x3; callers should not have to pad it."""
    mx23 = np.eye(3)[:2]
    assert register_coarse.committed_ncc(textured, textured, mx23) > 0.99


def test_ncc_falls_off_as_the_matrix_gets_worse(textured):
    """Only until the shift passes the image's decorrelation length.

    Beyond that the two images are simply uncorrelated and the score sits on a
    noise floor that jitters rather than keeps falling -- measured here as
    1.00 / 0.17 / 0.007 / 0.011 at dx = 0 / 5 / 30 / 120. So a low ncc bounds
    how wrong the matrix is, but its magnitude says nothing about how wrong.
    """
    perfect, near, far = (
        register_coarse.committed_ncc(
            textured, textured, register_util.translate_mx(dx, 0)
        )
        for dx in (0, 5, 30)
    )
    assert perfect > 0.99 > near > 0.05 > far


# --- the block_affine guard -------------------------------------------------


def test_oversized_source_crop_names_the_affine(monkeypatch):
    """A wrong affine inverse-maps a 1024px block across the whole image.

    cv2 raises a bare `(-215) src.cols < SHRT_MAX` assertion from three frames
    down; this should say what is actually wrong.
    """
    src = np.zeros((4, 40000), dtype="uint16")
    # x-scale of 1/20000 means the inverse maps a 2px-wide destination block
    # across the full 40000px width of `src`
    tform = skimage.transform.AffineTransform(
        matrix=np.array([[1 / 20000.0, 0, 0], [0, 1.0, 0], [0, 0, 1]])
    )
    with pytest.raises(ValueError, match="exceeds cv2's 32767 limit"):
        block_affine.block_affine(
            position=(0, 0), block_shape=(2, 2), transformation=tform, src_img=src
        )


# --- run_batch keeps going --------------------------------------------------


def _csv(tmp_path, rows):
    p = tmp_path / "files.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["p1", "p2", "out_dir"])
        w.writeheader()
        w.writerows(rows)
    return p


def test_batch_continues_past_a_failing_row(tmp_path, monkeypatch):
    seen = []

    def stub(p1: str, p2: str, out_dir: str = ""):
        seen.append(p2)
        if "bad" in p2:
            raise align_he.CoarseAlignmentFailed("nope")
        return 0

    monkeypatch.setattr(align_he, "align_he", stub)
    path = _csv(
        tmp_path,
        [  # the bad row FIRST -- that is the case that used to cost the rest
            {"p1": "a", "p2": "bad.vsi", "out_dir": str(tmp_path)},
            {"p1": "b", "p2": "good1.vsi", "out_dir": str(tmp_path)},
            {"p1": "c", "p2": "good2.vsi", "out_dir": str(tmp_path)},
        ],
    )
    rc = align_he.run_batch(path, print_args=False)
    assert seen == ["bad.vsi", "good1.vsi", "good2.vsi"]
    assert rc == 1  # non-zero so a caller can tell


def test_batch_returns_zero_when_every_row_succeeds(tmp_path, monkeypatch):
    def stub(p1: str, p2: str, out_dir: str = ""):
        return 0

    monkeypatch.setattr(align_he, "align_he", stub)
    path = _csv(tmp_path, [{"p1": "a", "p2": "ok.vsi", "out_dir": str(tmp_path)}])
    assert align_he.run_batch(path, print_args=False) == 0


def test_an_unexpected_exception_is_caught_too(tmp_path, monkeypatch):
    def stub(p1: str, p2: str, out_dir: str = ""):
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(align_he, "align_he", stub)
    path = _csv(tmp_path, [{"p1": "a", "p2": "x.vsi", "out_dir": str(tmp_path)}])
    assert align_he.run_batch(path, print_args=False) == 1
