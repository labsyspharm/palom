"""Refine an Aligner's coarse affine matrix using block-wise translations.

The coarse affine is derived from low-resolution thumbnails, so its
non-translation terms (rotation / scale / shear) and its translation can carry
a residual of tens of pixels at full resolution. This module samples a handful
of tissue blocks, measures a per-block translation at full resolution with
phase correlation (palom's sign-robust LoG correlation, which works across
modalities), and fits a single global affine to those block-center
correspondences with RANSAC. The fitted affine replaces the coarse affine.

Each block contributes only a *translation*; the global rotation/scale/shear is
recovered from how those translations vary across the field-of-view, so the
blocks must be spread out (stratified sampling) for the angular terms to be
well constrained. RANSAC rejects blocks with unreliable correlation or local
tissue deformation.

Note: a single global affine assumes one rigid object. For scans with multiple
tissue pieces that moved independently, refine per object instead (see
`align_multi_obj`).
"""
import cv2
import numpy as np
import skimage.transform
from loguru import logger

from . import block_affine, img_util, register


def _block_edges(chunks):
    return np.r_[0, np.cumsum(chunks)]


def _sample_tissue_blocks(tissue_grid, n_cells, min_tissue=0.5):
    """Pick the most-tissue block within each cell of an n_cells x n_cells grid.

    The cells span the bounding box of the tissue (where `tissue_grid` exceeds
    `min_tissue`), so the blocks are spread across the actual tissue extent --
    important when `tissue_grid` is restricted to a single object.
    """
    rr, cc = np.where(tissue_grid >= min_tissue)
    if rr.size == 0:
        return []
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    bi_e = np.linspace(r0, r1, n_cells + 1).astype(int)
    bj_e = np.linspace(c0, c1, n_cells + 1).astype(int)
    picks = []
    for a in range(n_cells):
        for b in range(n_cells):
            sub = tissue_grid[bi_e[a]:bi_e[a + 1], bj_e[b]:bj_e[b + 1]]
            if sub.size == 0 or sub.max() < min_tissue:
                continue
            li, lj = np.unravel_index(np.argmax(sub), sub.shape)
            picks.append((bi_e[a] + li, bj_e[b] + lj))
    return sorted(set(picks))


def _extract_moving_block(affine, moving, y0, x0, bh, bw):
    """Warp `moving` into the ref block at (y0,x0) of size (bh,bw).

    `affine` maps moving(x,y)->ref(x,y). Reuses palom's block_affine, which
    reads only the bounded moving region needed for this block.
    """
    tform = skimage.transform.AffineTransform(matrix=affine)
    return block_affine.block_affine(
        (y0, x0), (bh, bw), tform, moving
    ).astype("float32")


def refine_affine_by_block_translation(
    aligner,
    n_cells=8,
    whiten_sigma=2.0,
    ransac_threshold=3.0,
    min_inliers=6,
    accept_residual=5.0,
    block_mask=None,
    plot=True,
):
    """Refine `aligner.coarse_affine_matrix` from block-wise PC translations.

    `block_mask`, if given, is a boolean array of shape `aligner.ref_img.numblocks`
    that restricts block sampling to a region (e.g. a single tissue object);
    only blocks where it is True are used.

    Returns the refined coarse_affine_matrix (in thumbnail frame, ready to
    assign back to `aligner.coarse_affine_matrix`), or None if the refinement
    is rejected (too few inliers, or it failed to tighten the inlier
    alignment) -- in which case the caller should keep the coarse affine.
    """
    ref = aligner.ref_img
    moving = aligner.moving_img
    A = aligner.affine_matrix  # full-res moving(x,y)->ref(x,y)
    inv_A = skimage.transform.AffineTransform(matrix=A)

    ys, xs = _block_edges(ref.chunks[0]), _block_edges(ref.chunks[1])
    nbi, nbj = ref.numblocks
    block_px = int(min(np.median(ref.chunks[0]), np.median(ref.chunks[1])))
    if block_px < 1024:
        logger.warning(
            f"Refinement block size is ~{block_px}px; phase correlation is more"
            f" reliable with larger blocks (set `ref_block_size` to ~2048-4096)"
        )

    # tissue fraction per block, from the coarse ref thumbnail resized to grid
    tissue = img_util.entropy_mask(
        np.asarray(aligner.ref_thumbnail).astype("float32")
    ).astype("float32")
    tissue_grid = cv2.resize(tissue, (nbj, nbi), interpolation=cv2.INTER_AREA)
    if block_mask is not None:
        tissue_grid = tissue_grid * np.asarray(block_mask, dtype="float32")
    picks = _sample_tissue_blocks(tissue_grid, n_cells)

    # per-block translation -> (block center, shift) correspondences
    centers, shifts = [], []
    for (bi, bj) in picks:
        y0, y1, x0, x1 = ys[bi], ys[bi + 1], xs[bj], xs[bj + 1]
        rb = np.asarray(ref[y0:y1, x0:x1]).astype("float32")
        mb = _extract_moving_block(A, moving, y0, x0, y1 - y0, x1 - x0)
        if rb.shape != mb.shape:
            continue
        (sy, sx), err = register.phase_cross_correlation(
            rb, mb, sigma=whiten_sigma, upsample=10
        )
        if not np.all(np.isfinite([sy, sx, err])):
            continue
        centers.append([(x0 + x1) / 2, (y0 + y1) / 2])
        shifts.append([sx, sy])
    centers, shifts = np.array(centers), np.array(shifts)
    logger.info(
        f"Affine refinement: {len(centers)} block correspondences from"
        f" {len(picks)} sampled tissue blocks (~{block_px}px)"
    )
    if len(centers) < min_inliers:
        logger.warning("Too few block correspondences; keeping coarse affine")
        return None

    # skimage convention: ref(x) ~ moving_warped(x - shift), so ref point c
    # corresponds to original moving point inv_A(c - shift)
    src = inv_A.inverse(centers - shifts).astype("float32")
    dst = centers.astype("float32")
    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold, maxIters=5000, confidence=0.999,
    )
    if M is None or inliers is None:
        logger.warning("RANSAC affine fit failed; keeping coarse affine")
        return None
    inliers = inliers.ravel().astype(bool)
    A_new = np.vstack([M, [0, 0, 1]])

    n_in = int(inliers.sum())
    res = np.hypot(*(
        skimage.transform.AffineTransform(matrix=A_new)(src[inliers]) - dst[inliers]
    ).T)
    median_res = float(np.median(res)) if n_in else np.inf
    if n_in < min_inliers or median_res > accept_residual:
        logger.warning(
            f"Affine refinement rejected (inliers {n_in}/{len(centers)},"
            f" inlier residual {median_res:.2f}px); keeping coarse affine"
        )
        return None

    correction = A_new @ np.linalg.inv(A)
    ct = skimage.transform.AffineTransform(matrix=correction)
    # physically meaningful translation = how far the ref image center moves
    # under the correction (origin-relative `ct.translation` is misleading when
    # the linear part rotates/scales about a point far from the origin)
    H, W = ref.shape
    center = np.array([W / 2, H / 2])
    center_shift = ct(center[None])[0] - center
    logger.info(
        f"Affine refinement accepted: inliers {n_in}/{len(centers)},"
        f" inlier residual {median_res:.2f}px; correction"
        f" rot={np.rad2deg(ct.rotation):+.3f}deg"
        f" scale=({ct.scale[0]:.5f},{ct.scale[1]:.5f})"
        f" shear={np.rad2deg(ct.shear):+.3f}deg"
        f" center-shift=({center_shift[0]:+.1f},{center_shift[1]:+.1f})px"
    )

    if plot:
        _plot_shift_field(aligner, centers, shifts, inliers, n_in, len(centers))

    # convert full-res affine back to thumbnail frame (inverse of the scaling
    # applied in Aligner.affine_matrix):
    #   affine_matrix = inv(ref_s) @ coarse @ mov_s
    #   coarse        = ref_s @ affine_matrix @ inv(mov_s)
    ref_s = skimage.transform.AffineTransform(
        scale=1 / aligner.ref_thumbnail_down_factor
    ).params
    mov_s = skimage.transform.AffineTransform(
        scale=1 / aligner.moving_thumbnail_down_factor
    ).params
    return ref_s @ A_new @ np.linalg.inv(mov_s)


def _plot_shift_field(aligner, centers, shifts, inliers, n_in, n_total):
    import matplotlib.pyplot as plt

    thumb = np.asarray(aligner.ref_thumbnail).astype("float32")
    sc = thumb.shape[1] / (aligner.ref_img.shape[1])
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.suptitle("coarse affine refinement (residual shift field)")
    ax.imshow(np.log1p(thumb), cmap="gray")
    cx, cy = centers.T
    ux, uy = shifts.T
    colors = ["lime" if i else "red" for i in inliers]
    ax.quiver(
        cx * sc, cy * sc, ux, -uy, color=colors,
        angles="xy", scale_units="xy", scale=8, width=0.004,
    )
    ax.set_title(
        f"green=inlier ({n_in}/{n_total}), red=outlier;"
        f" median |shift| {np.median(np.hypot(ux, uy)):.0f}px",
        fontsize=8,
    )
    ax.axis("off")
    return fig
