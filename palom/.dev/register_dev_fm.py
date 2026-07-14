"""Experimental phase-correlation / Fourier-Mellin coarse alignment.

NOT production-ready. `phase_correlation_register` is a correlation-based alternative
to `register_coarse.search_then_register` for cross-modality / small-portion pairs. It is
kept isolated here because Fourier-Mellin couples scale+rotation and is fragile to
content mismatch under rotation (works reliably only near true rotation ~0); the
feature-based windowed route in `register_coarse` is the more robust path. See the
`palom-pc-coarse-align` design notes for the full findings.

`fm_backend="diplib"` uses diplib's `FourierMellinMatch2D` (optional dependency);
`fm_backend="skimage"` is a dependency-free port (GaussianTukey window + Ln|FFT| +
log-polar + phase-normalized correlation).
"""
import functools

import numpy as np
import scipy.fft
import scipy.special
import skimage.feature
import skimage.registration
import skimage.transform
from loguru import logger

# quarantined dev/reference module (Fourier-Mellin phase-correlation coarse
# alignment); kept out of the shipped package. Absolute import so it still runs
# standalone from palom/.dev/.
from palom import img_util, register_util, register


def _mx_scale(scale):
    mx = np.eye(3) * scale
    mx[2, 2] = 1
    return mx


def _pad_to_common(img1, img2):
    # pad bottom/right only so the coordinate origin (top-left) is preserved
    shape = np.maximum(img1.shape, img2.shape)
    p1 = np.pad(img1, [(0, shape[0] - img1.shape[0]), (0, shape[1] - img1.shape[1])])
    p2 = np.pad(img2, [(0, shape[0] - img2.shape[0]), (0, shape[1] - img2.shape[1])])
    return p1, p2


def _dip_fm_matrix(ref, mov):
    """Full similarity transform (3x3, x-y homogeneous) mapping `mov` -> `ref`,
    estimated by diplib's Fourier-Mellin matcher (a mature Reddy-Chatterji
    implementation with high-pass filtering, sub-pixel peak fitting, and internal
    180-degree disambiguation). Recovers rotation + scale + translation; does NOT
    recover reflection (handled by the caller's flip permutation).

    diplib returns params = [m00, m01, m10, m11, tx, ty] for a center-based
    transform with a column-major linear block. The mapping mov -> ref is
    ``T(center) @ T(tx, ty) @ [[m00, m10], [m01, m11]] @ T(-center)`` (verified
    against diplib's own resampled output)."""
    try:
        import diplib as dip
    except ImportError as e:  # optional dependency
        raise ImportError(
            "phase_correlation_register(estimate_rotation_scale=True) requires the "
            "optional 'diplib' package (`pip install diplib`)."
        ) from e
    ref, mov = _pad_to_common(ref, mov)
    _matched, params = dip.FourierMellinMatch2Dparams(
        dip.Image(np.ascontiguousarray(ref, dtype="float32")),
        dip.Image(np.ascontiguousarray(mov, dtype="float32")),
    )
    m00, m01, m10, m11, tx, ty = params
    h, w = ref.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    linear = np.array([[m00, m10, 0], [m01, m11, 0], [0, 0, 1]])
    return (
        register_util.translate_mx(cx, cy)
        @ register_util.translate_mx(tx, ty)
        @ linear
        @ register_util.translate_mx(-cx, -cy)
    )


@functools.lru_cache
def _gaussian_tukey_1d(n, sigma=10.0):
    # diplib's "GaussianTukey" window: flat interior with an erf edge taper of the
    # given sigma (matches dip.ApplyWindow(..., "GaussianTukey", 10) to ~1e-3).
    d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype("float64")
    return 0.5 * (1 + scipy.special.erf((d - 3 * sigma) / (np.sqrt(2) * sigma)))


def _gaussian_tukey(shape, sigma=10.0):
    return np.outer(
        _gaussian_tukey_1d(shape[0], sigma), _gaussian_tukey_1d(shape[1], sigma)
    ).astype("float32")


def _similarity_mx(img_shape, angle_deg, scale):
    # rotate + scale about the image center, (x, y) homogeneous coords
    cx, cy = (img_shape[1] - 1) / 2.0, (img_shape[0] - 1) / 2.0
    s = skimage.transform.SimilarityTransform(
        scale=scale, rotation=np.deg2rad(angle_deg)
    ).params
    return (
        register_util.translate_mx(cx, cy)
        @ s
        @ register_util.translate_mx(-cx, -cy)
    )


def _logpolar_angle_scale(ref, mov, upsample):
    """Rotation (deg) and scale of `mov` relative to `ref` from the log-polar
    transform of the log-magnitude spectrum (Fourier-Mellin). Rotation is only
    determined modulo 180 (magnitude-spectrum symmetry); scale is unique. The key
    detail is `normalization="phase"` (true phase correlation): without it the plain
    cross-correlation is swamped by the DC-dominated spectrum and the scale channel
    never registers."""
    size = max(ref.shape)
    maxr = float(np.min(np.array(ref.shape) / 2.0))

    def _logpolar(img):
        spectrum = np.log(
            np.abs(
                scipy.fft.fftshift(scipy.fft.fft2(img * _gaussian_tukey(img.shape)))
            )
            + 1e-6
        )
        return skimage.transform.warp_polar(
            spectrum, radius=maxr, output_shape=(size, size), scaling="log", order=1
        )

    shift = skimage.registration.phase_cross_correlation(
        _logpolar(ref),
        _logpolar(mov),
        upsample_factor=upsample,
        normalization="phase",
        return_error=False,
    )
    # warp_polar output axes are (angle=rows, radius=cols); the scale-exponent sign
    # is calibrated so the returned scale is in the mov->ref direction
    angle = float(shift[0] * 360.0 / size)
    scale = float(maxr ** (-shift[1] / (size - 1)))
    return angle, scale


def _skimage_fm_matrix(ref, mov, sigma=1.0, upsample=10):
    """Pure skimage/scipy Fourier-Mellin: full similarity transform (3x3, x-y
    homogeneous) mapping `mov` -> `ref`, a dependency-free alternative to
    `_dip_fm_matrix`. Recovers rotation + scale from the log-polar spectrum, then
    resolves the 180-degree rotation ambiguity and the translation together with a
    spatial (LoG-whitened) phase correlation, keeping whichever rotation gives the
    lower phase-correlation error."""
    ref, mov = _pad_to_common(ref, mov)
    angle0, scale = _logpolar_angle_scale(ref, mov, upsample)
    best = None
    for angle in (angle0, angle0 + 180.0):
        rs = _similarity_mx(ref.shape, angle, scale)
        moving_rs = skimage.transform.warp(
            mov,
            skimage.transform.AffineTransform(matrix=rs).inverse,
            output_shape=ref.shape,
            preserve_range=True,
        ).astype("float32")
        shift, error = register.phase_cross_correlation(
            ref, moving_rs, sigma, upsample=upsample
        )
        ty, tx = shift
        mx = register_util.translate_mx(tx, ty) @ rs
        if best is None or error < best[0]:
            best = (error, mx)
    return best[1]


def _fm_matrix(ref, mov, backend, sigma, upsample):
    if backend == "diplib":
        return _dip_fm_matrix(ref, mov)
    elif backend == "skimage":
        return _skimage_fm_matrix(ref, mov, sigma=sigma, upsample=upsample)
    raise ValueError(f"unknown fm_backend: {backend!r}")


def _match_template_shift(image, template):
    """Locate `template` inside `image` via locally-normalized NCC. Returns the
    (row, col) of the template center in `image` and the peak |NCC| score."""
    # match_template requires the template to fit inside the image; pad the image
    # (bottom/right, origin-preserving) if the template is larger in some dim
    ph = max(0, template.shape[0] - image.shape[0])
    pw = max(0, template.shape[1] - image.shape[1])
    if ph or pw:
        image = np.pad(image, [(0, ph), (0, pw)])
    resp = np.abs(skimage.feature.match_template(image, template, pad_input=True))
    peak = np.unravel_index(np.argmax(resp), resp.shape)
    return peak, float(resp.max())


def _translation_stage(ref, moving, sigma, upsample, method):
    """Estimate the residual translation mapping `moving` -> `ref` (as (tx, ty) in
    x, y) plus a scalar score (higher = better)."""
    if method == "match_template":
        ref_w = img_util.whiten(ref, sigma)
        moving_w = img_util.whiten(moving, sigma)
        # use the smaller-area image as the template (robust to partial overlap)
        if moving_w.size <= ref_w.size:
            (pr, pc), score = _match_template_shift(ref_w, moving_w)
            tx = pc - (moving_w.shape[1] - 1) / 2.0
            ty = pr - (moving_w.shape[0] - 1) / 2.0
        else:
            (pr, pc), score = _match_template_shift(moving_w, ref_w)
            tx = -(pc - (ref_w.shape[1] - 1) / 2.0)
            ty = -(pr - (ref_w.shape[0] - 1) / 2.0)
        return (tx, ty), score
    elif method == "phase_correlation":
        ref_p, moving_p = _pad_to_common(ref, moving)
        shift, error = register.phase_cross_correlation(
            ref_p, moving_p, sigma, upsample=upsample
        )
        ty, tx = shift
        return (tx, ty), -error
    raise ValueError(f"unknown translation_method: {method}")


def phase_correlation_register(
    img_left,
    img_right,
    max_size=2000,
    sigma=1.0,
    upsample=10,
    estimate_rotation_scale=True,
    fm_backend="diplib",
    translation_method="match_template",
    plot_match_result=True,
):
    """Correlation-based coarse alignment, a drop-in alternative to
    `search_then_register` for cross-modality / small-portion pairs where ORB +
    RANSAC feature matching struggles.

    Brute-forces a flip permutation (2 configs). When `estimate_rotation_scale` is
    True, recovers rotation + scale + translation per flip with Fourier-Mellin and
    picks the flip by whitened-intensity overlap correlation. `fm_backend` selects
    the Fourier-Mellin implementation: "diplib" (`_dip_fm_matrix`, optional
    dependency, most robust) or "skimage" (`_skimage_fm_matrix`, dependency-free
    port). When `estimate_rotation_scale` is False, recovers translation only via
    `match_template` (partial-overlap robust) or phase correlation.

    Returns a 2x3 (x, y) affine mapping `img_right` (moving) -> `img_left` (ref) at
    full resolution, matching `search_then_register`'s contract (identity
    fallback)."""
    assert translation_method in ("match_template", "phase_correlation")
    img_left = np.asarray(img_left).astype("float32")
    img_right = np.asarray(img_right).astype("float32")

    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.ceil(shape_max / max_size))
    ref = img_util.cv2_downscale_local_mean(img_left, downsize_factor)
    moving = img_util.cv2_downscale_local_mean(img_right, downsize_factor)

    flip_funcs = [np.array, np.flipud]
    flip_mxs = [np.eye(3), register_util.get_flip_mx(moving.shape, 0)]

    results = []
    for (ff, flip_mx) in zip(flip_funcs, flip_mxs):
        moving_f = ff(moving)
        if estimate_rotation_scale:
            # Fourier-Mellin recovers rotation + scale + translation directly
            mx_cfg = _fm_matrix(ref, moving_f, fm_backend, sigma, upsample) @ flip_mx
            score = register_util.score_overlap(ref, moving, mx_cfg, sigma)
        else:
            # translation-only: recover the residual shift on the (flipped) moving
            (tx, ty), score = _translation_stage(
                ref, moving_f, sigma, upsample, translation_method
            )
            mx_cfg = register_util.translate_mx(tx, ty) @ flip_mx
        results.append((score, mx_cfg))
        logger.debug(f"flip={ff.__name__:8} score={score:.4g}")

    scores = np.array([r[0] for r in results])
    if not np.isfinite(scores).any():
        logger.warning(
            "Phase-correlation registration failed. Returning identity matrix"
        )
        return np.eye(3)[:2]

    best = int(np.nanargmax(scores))
    mx = results[best][1]
    mx_full_res = _mx_scale(downsize_factor) @ mx @ _mx_scale(1 / downsize_factor)

    if plot_match_result:
        _plot_pc_overlay(ref, moving, mx)

    return mx_full_res[:2]


def _plot_pc_overlay(ref, moving, mx):
    import matplotlib.pyplot as plt

    warped = skimage.transform.warp(
        moving,
        skimage.transform.AffineTransform(matrix=mx).inverse,
        output_shape=ref.shape,
        preserve_range=True,
    )
    rgb = np.zeros((*ref.shape, 3), dtype="float32")
    rgb[..., 0] = img_util.cv2_to_uint8(ref) / 255.0
    rgb[..., 1] = img_util.cv2_to_uint8(warped) / 255.0
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_title("coarse alignment (R: ref, G: moving)", fontsize=8)
    ax.set_axis_off()
    return fig
