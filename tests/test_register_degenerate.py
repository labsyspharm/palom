"""Degenerate-input behaviour of the feature-matching path.

Every case here is reachable in a normal run: `align_multi_obj` masks each
object's thumbnail down to its own bbox and fills the rest with a constant, and
the windowed coarse route scores tiles that land entirely on background. Before
2026-08-27 a keypoint-less crop returned `np.empty((1, 2))` -- uninitialized
memory -- which fed two random coordinates into every pooled RANSAC fit and made
identical inputs produce different outputs.
"""

import cv2
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from palom import register, register_coarse  # noqa: E402


@pytest.fixture
def blank():
    """No keypoints at all -- a fully masked-out object crop."""
    return np.zeros((256, 256), dtype="uint16")


@pytest.fixture
def noise():
    """Keypoints, but nothing that matches anything."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 4096, (256, 256)).astype("uint16")


# --- the correspondence sentinel -------------------------------------------


def test_no_keypoints_returns_empty_pairs(blank):
    src, dst = register.cv2_feature_detect_and_match(blank, blank)
    assert src.shape == (0, 2)
    assert dst.shape == (0, 2)
    # (0, 2) not (0,): callers `np.vstack` these together
    assert src.dtype == np.float32


def test_no_keypoints_is_deterministic(blank):
    """The regression test for the uninitialized-memory sentinel."""
    first = register.cv2_feature_detect_and_match(blank, blank)
    second = register.cv2_feature_detect_and_match(blank, blank)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_empty_pairs_are_falsy_for_the_config_search(blank):
    """`search_best_match_config` guards its fit with a length test.

    The old `(1, 2)` sentinel was truthy, so the guard never fired and
    `estimateAffine2D` ran on a single garbage pair.
    """
    src, _ = register.cv2_feature_detect_and_match(blank, blank)
    assert len(src) < register.MIN_AFFINE_POINTS


# --- what cv2 actually does, so the guards stay honest ----------------------


@pytest.mark.parametrize("n", [1, 2])
def test_cv2_returns_none_matrix_but_real_mask_below_three_points(n):
    pts = np.random.rand(n, 2).astype("float32")
    matrix, mask = cv2.estimateAffine2D(pts, pts, method=cv2.RANSAC)
    assert matrix is None
    # NOT None -- this is why the `mask.flatten()` deref never actually crashed
    assert mask is not None and mask.shape == (n, 1)


def test_cv2_raises_on_empty_points():
    """Why `MIN_AFFINE_POINTS` guards the pooled fit rather than trusting None."""
    with pytest.raises(cv2.error):
        cv2.estimateAffine2D(
            np.empty((0, 2), "float32"), np.empty((0, 2), "float32"),
            method=cv2.RANSAC,
        )


def test_cv2_degenerate_points_give_all_zero_mask():
    pts = np.array([[i, i] for i in range(6)], "float32")  # collinear
    matrix, mask = cv2.estimateAffine2D(pts, pts, method=cv2.RANSAC)
    assert matrix is None
    assert int(mask.sum()) == 0


# --- the pooled fit ---------------------------------------------------------


def test_ensambled_match_on_blanks_returns_none_matrix_and_zero_mask(blank):
    matrix, mask = register.ensambled_match(blank, blank, return_match_mask=True)
    assert matrix is None
    assert int(mask.sum()) == 0
    # mask is always an array, so `match.sum()` is safe for every caller
    assert mask.ndim == 2 and mask.shape[1] == 1


def test_ensambled_match_plotting_survives_a_failed_fit(blank):
    """`coarse_register_affine` defaults `plot_match_result=True`."""
    matrix, _ = register.ensambled_match(
        blank, blank, return_match_mask=True, plot_match_result=True
    )
    assert matrix is None


# --- the callers ------------------------------------------------------------


def test_search_best_match_config_on_blanks(blank):
    score, config = register_coarse.search_best_match_config(
        blank, blank, n_keypoints=200
    )
    assert score == 0
    assert len(config) == 3


def test_search_then_register_on_blanks_returns_identity(blank):
    matrix = register_coarse.search_then_register(
        blank, blank, n_keypoints=200, plot_match_result=False
    )
    assert np.allclose(matrix, np.eye(3)[:2])


def test_search_then_register_on_unmatchable_noise(noise):
    rng = np.random.default_rng(1)
    other = rng.integers(0, 4096, (256, 256)).astype("uint16")
    matrix, count = register_coarse.search_then_register(
        noise, other, n_keypoints=200, plot_match_result=False,
        return_match_count=True,
    )
    assert matrix.shape == (2, 3)
    assert count >= 0


# --- the good path must not move -------------------------------------------


def test_known_transform_is_recovered():
    """Guards against a degenerate-case fix changing ordinary behaviour."""
    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, (600, 800)).astype("float32")
    base = cv2.GaussianBlur(base, (0, 0), 3)
    base = cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC)

    truth = cv2.getRotationMatrix2D((800, 600), 4.0, 1.02)
    truth[:, 2] += (37, -21)
    moved = cv2.warpAffine(base, truth, (1600, 1200))

    matrix, count = register_coarse.search_then_register(
        base, moved, n_keypoints=2000, plot_match_result=False,
        return_match_count=True,
    )
    assert count > 100

    # `search_then_register` maps moved -> base, i.e. the inverse of `truth`
    expected = np.linalg.inv(np.vstack([truth, [0, 0, 1]]))[:2]
    corners = np.array([[0, 0], [1600, 0], [0, 1200], [1600, 1200]], "float32")
    got_pts = cv2.transform(corners[None], matrix)[0]
    want_pts = cv2.transform(corners[None], expected)[0]
    assert np.linalg.norm(got_pts - want_pts, axis=1).max() < 3.0
