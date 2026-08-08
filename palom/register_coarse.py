"""Coarse, feature-based (ORB + RANSAC) affine registration of whole-slide images.

`coarse_register` is the entry point; it dispatches between a whole-image route
(`search_then_register`) and a sliding-window route (`windowed_search_then_register`)
for "small-portion" pairs, where one scan images only a fraction of the other. All
routes take two single-channel images and return a 2x3 affine mapping `img_right`
(moving) -> `img_left` (ref), in full-resolution pixel coordinates of the inputs.

Matching runs on downsampled copies and, before matching, searches a small space of
intensity/orientation configurations (which image is histogram-matched into which,
intensity inversion, vertical flip); cross-modality pairs (e.g. H&E vs. IF) and
mirrored scans otherwise defeat ORB matching. Alignment confidence is scored with
`register_util.score_overlap`.
"""

import itertools

import cv2
import numpy as np
from loguru import logger

from . import img_util, plot_util, register_util, register


def masked_match_histograms(img, ref_img, mask=None, ref_mask=None):
    """Histogram-match `img` to `ref_img` using only masked (tissue) pixels; the
    unmasked background is filled with the reference background's mean intensity.

    Masks default to `img_util.entropy_mask` computed at full resolution."""
    # NOT the same as register_util.masked_match_histograms, despite the shared
    # name: this one accepts explicit masks and computes them at full resolution
    # when omitted, while register_util's takes no masks and derives them from a
    # ~500px downsample. Both are exercised on the coarse path (this via
    # search_then_register, that via ensambled_match) and both now match with
    # `register_util.match_quantized` -- a ~20-30x faster, <0.01% mean-error
    # drop-in for skimage's `_match_cumulative_cdf`.
    if mask is None:
        mask = img_util.entropy_mask(img)
    if ref_mask is None:
        ref_mask = img_util.entropy_mask(ref_img)

    matched_img = np.zeros_like(img)
    matched_img[mask] = register_util.match_quantized(img[mask], ref_img[ref_mask])
    matched_img[~mask] = ref_img[~ref_mask].mean()
    return matched_img


def match_img_with_config(img1, img2, mask1, mask2, adjust_which, scalar, func):
    """Apply one matching config to an image pair, returning the transformed
    ``(img1, img2)``.

    - ``adjust_which`` ("left"/"right"): which image is histogram-matched into the
      other's intensity distribution (the other is left untouched);
    - ``scalar`` (1.0/-1.0): inverts that image's intensity before matching, which
      bridges dark- and light-background modalities;
    - ``func`` (``np.array``/``np.flipud``): always applied to ``img2``, covering a
      mirrored right image."""
    assert adjust_which in ("left", "right")
    if adjust_which == "right":
        return img1, func(masked_match_histograms(scalar * img2, img1, mask2, mask1))
    else:
        return masked_match_histograms(scalar * img1, img2, mask1, mask2), func(img2)


def format_config(config):
    """One-line rendering of a `search_best_match_config` config, for logging."""
    adjust_which, scalar, func = config
    return f"adjust={adjust_which}, scalar={scalar:+.0f}, flip={func.__name__}"


def search_best_match_config(
    img_left,
    img_right,
    max_size=500,
    auto_mask=True,
    n_keypoints=2000,
    min_fold_increase=5,
):
    """Search the 8 (`adjust_which`, `scalar`, flip) configs and return
    ``(n_inliers, config)`` for the best one; `config` is consumable by
    `match_img_with_config`.

    Both images are downscaled so their largest side is ~`max_size`, and each config
    is scored by the RANSAC inlier count of `cv2.estimateAffine2D` on its ORB matches.
    The winner is trusted only if it beats the mean of the other configs by
    `min_fold_increase`; otherwise the search restarts at twice the resolution, and
    keeps doubling until the images are no longer downscaled."""
    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.ceil(shape_max / max_size))

    def _process_img(img):
        img = img.astype("float32")
        img = img_util.cv2_downscale_local_mean(img, downsize_factor)
        mask = np.ones_like(img, dtype="bool")
        if auto_mask:
            mask = img_util.entropy_mask(img, kernel_size=9)
        return img, mask

    img1, mask1 = _process_img(img_left)
    img2, mask2 = _process_img(img_right)

    results = []
    logger.debug(
        f"downsize_factor={downsize_factor}, auto_mask={auto_mask}, n_keypoints={n_keypoints}"
    )
    for cc in itertools.product(["right", "left"], [1.0, -1.0], [np.array, np.flipud]):
        i1, i2 = match_img_with_config(img1, img2, mask1, mask2, *cc)
        p_src, p_dst = register.cv2_feature_detect_and_match(
            i1, i2, n_keypoints=n_keypoints
        )
        valid_match = np.zeros(1, dtype="int")
        if len(p_src):
            _affine_mx, valid_match = cv2.estimateAffine2D(
                p_dst, p_src, method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=5000
            )
        results.append((valid_match.sum(), cc))
        logger.debug(
            f"{valid_match.sum():6} matches, {cc[0]:5} {cc[1]:4} {cc[2].__name__:6}"
        )

    matches = np.array([rr[0] for rr in results])
    best = matches.max()
    fold_increase = best / np.mean(matches[matches < best])
    idx = np.argmax(matches)
    if fold_increase > min_fold_increase:
        logger.debug(
            f"config {format_config(results[idx][1])}: {best} inliers,"
            f" {fold_increase:.1f}x the mean of the rest"
        )
        return results[idx]
    if downsize_factor == 1:
        # the recursion bottomed out without any config standing clear of the
        # others, so this is the argmax of a flat field -- likely noise, and
        # coarser passes may well have favoured a different config
        logger.warning(
            f"No confident match config: best is {format_config(results[idx][1])}"
            f" with {best} inliers, only {fold_increase:.1f}x the mean of the"
            f" rest (need {min_fold_increase}x); using it anyway"
        )
        return results[idx]

    return search_best_match_config(
        img_left=img_left,
        img_right=img_right,
        max_size=2 * max_size,
        auto_mask=auto_mask,
        n_keypoints=n_keypoints,
        min_fold_increase=min_fold_increase,
    )


def _swap_config_sides(config):
    """Rewrite a config for a call whose `img_left`/`img_right` are swapped.

    Only `adjust_which` names a side, so flipping it keeps the histogram matching on
    the same *image* when the argument order changes. `scalar` follows the adjusted
    image, and `func` (applied to whichever image is `img_right`) lands on the other
    side -- harmless, since a mirror is an involution: flipping either operand makes
    the pair orientation-consistent."""
    adjust_which, scalar, func = config
    return ("left" if adjust_which == "right" else "right", scalar, func)


# The one coarse keypoint budget, whole-slide and per-object. Callers used to
# each carry their own (2 000 in `Aligner`, 20 000 for the multi-object
# baseline, 10 000 per object, 5 000 here), which meant the CLI, a bare
# `Aligner` and the golden fixture were all measuring different coarse fits.
# Distinct from `search_best_match_config`'s budget: that search runs 8 times on
# a much smaller image and is deliberately cheaper.
N_KEYPOINTS = 10_000


def search_then_register(
    img_left,
    img_right,
    max_size=2000,
    n_keypoints=N_KEYPOINTS,
    auto_mask=True,
    plot_match_result=True,
    search_kwargs=None,
    return_match_count=False,
    return_config=False,
    config=None,
):
    """Whole-image coarse registration: pick the best intensity/flip config, then
    feature-match with `register.ensambled_match`.

    Both images are downscaled so their largest side is ~`max_size` (`search_kwargs`
    is forwarded to `search_best_match_config`, which downsamples further on its own);
    the resulting affine is rescaled back to full-resolution coordinates. Returns the
    2x3 affine mapping `img_right` -> `img_left`; `return_match_count` and
    `return_config` each append an extra -- in that order -- to a returned tuple: the
    number of RANSAC-inlier keypoints backing the fit, and the config it matched with
    (searched or pinned), so a caller can reuse it rather than search again.

    `config` (as returned by `search_best_match_config`) skips the 8-config search and
    uses that config directly -- the config describes the image *pair* (modality
    relationship and mirroring), so a caller aligning many crops of the same pair can
    search once and pin the result.

    Failure to match is non-fatal: an identity matrix and a zero match count are
    returned with a warning."""
    search_kwargs = search_kwargs or {}
    img1 = img_left.astype("float32")
    img2 = img_right.astype("float32")

    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.ceil(shape_max / max_size))

    img1 = img_util.cv2_downscale_local_mean(img1, downsize_factor)
    img2 = img_util.cv2_downscale_local_mean(img2, downsize_factor)

    if config is None:
        _, config = search_best_match_config(img1, img2, **search_kwargs)
    else:
        # a searched config logs itself; a pinned one is otherwise invisible, so
        # a fit made with the wrong flip leaves no trace of what it matched with
        logger.debug(f"config {format_config(config)} (pinned by caller)")
    _img1, _img2 = match_img_with_config(
        img1,
        img2,
        img_util.entropy_mask(img1) if auto_mask else np.ones_like(img1, "bool"),
        img_util.entropy_mask(img2) if auto_mask else np.ones_like(img2, "bool"),
        *config,
    )
    mx, match = register.ensambled_match(
        _img1,
        _img2,
        n_keypoints=n_keypoints,
        plot_match_result=plot_match_result,
        return_match_mask=True,
        auto_invert_intensity=False,
        auto_mask=auto_mask,
    )
    mx_flip = np.eye(3)
    if config[2] == np.flipud:
        mx_flip = register_util.get_flip_mx(img2.shape, 0)
    if mx is None:
        logger.warning(
            "Feature matching failed. Returning identity matrix as placeholder"
        )
        mx = np.eye(3)[:2]
        match = np.zeros(1, "bool")
    mx = (np.vstack([mx, [0, 0, 1]]) @ mx_flip)

    def mx_scale(scale):
        mx = np.eye(3) * scale
        mx[2, 2] = 1
        return mx

    mx_full_res = mx_scale(downsize_factor) @ mx @ mx_scale(1 / downsize_factor)

    logger.debug(
        f"{match.sum():6} matches; {n_keypoints:6} keypoints; mask: {auto_mask}"
    )
    extras = []
    if return_match_count:
        extras.append(int(match.sum()))
    if return_config:
        extras.append(config)
    if extras:
        return (mx_full_res[:2], *extras)
    return mx_full_res[:2]


def _tile_origins(length, window, step):
    """Window start offsets tiling `length` with stride `step`, with the last window
    flush against the far edge so the tail is never dropped."""
    starts = list(range(0, max(1, length - window + 1), step))
    if starts[-1] != length - window:
        starts.append(max(0, length - window))
    return starts


def matched_geometry(shape_left, shape_right, pixel_size_left, pixel_size_right):
    """Physical geometry from nominal pixel sizes (um/pixel). Returns
    ``(area_ratio, left_is_bigger, footprint_px)`` where:
    - ``area_ratio`` = smaller / larger physical footprint area, in (0, 1] (the
      overlap fraction when the smaller scan is contained in the larger);
    - ``left_is_bigger`` = whether ``img_left`` has the larger physical footprint;
    - ``footprint_px`` = the SMALLER image's max linear extent measured in the LARGER
      image's pixel grid (the natural tile size for the windowed route)."""
    area_l = shape_left[0] * shape_left[1] * pixel_size_left ** 2
    area_r = shape_right[0] * shape_right[1] * pixel_size_right ** 2
    left_big = area_l >= area_r
    area_ratio = min(area_l, area_r) / max(area_l, area_r)
    big_px = pixel_size_left if left_big else pixel_size_right
    small_shape = shape_right if left_big else shape_left
    small_px = pixel_size_right if left_big else pixel_size_left
    footprint_px = max(small_shape) * small_px / big_px
    return area_ratio, left_big, footprint_px


def windowed_search_then_register(
    img_left,
    img_right,
    window=None,
    step=None,
    pixel_size_left=None,
    pixel_size_right=None,
    # kept equal to `coarse_register`'s: the dispatcher always forwards its own
    # value, so a different default here would only ever apply to direct callers
    window_margin=1.0,
    max_size=2000,
    n_keypoints=N_KEYPOINTS,
    auto_mask=True,
    plot_match_result=False,
    n_workers=1,
    return_config=False,
    config=None,
):
    """Coarse alignment for small-portion pairs (one scan images only a fraction of
    the other), where whole-image feature matching struggles.

    Slides a `window`x`window` tile over the *larger* image, runs
    `search_then_register(smaller, tile)` on each (feature matching handles the
    per-tile flip/rotation/scale), composes with the tile offset to get a full-image
    affine, warps the larger image into the smaller one's frame, and keeps the tile
    whose affine maximizes the whitened-intensity overlap correlation
    (`register_util.score_overlap`, tie-broken by matched-keypoint count). Returns a
    2x3 affine mapping `img_right` (moving) -> `img_left`
    (ref), matching `search_then_register`'s contract.

    When both nominal pixel sizes are given, they set which image is the larger
    (physical footprint, robust to pixel-size differences) and, unless `window` is
    given explicitly, the tile size (the smaller image's footprint in the larger's
    pixel grid, times `window_margin`). Otherwise the larger image is taken by pixel
    count and `window` defaults to 500."""
    img_left = np.asarray(img_left).astype("float32")
    img_right = np.asarray(img_right).astype("float32")

    if pixel_size_left is not None and pixel_size_right is not None:
        _ratio, left_big, footprint = matched_geometry(
            img_left.shape, img_right.shape, pixel_size_left, pixel_size_right
        )
        if window is None:
            window = int(np.ceil(footprint * window_margin))
    else:
        left_big = img_left.size >= img_right.size
    if window is None:
        window = 500
    big, small = (img_left, img_right) if left_big else (img_right, img_left)

    win = int(min(window, *big.shape))
    step = step or win

    # the per-tile call is search_then_register(small, tile), so when `big` is
    # img_left the tile sits on the right and the pinned config's sides are swapped
    tile_config = config
    if config is not None and left_big:
        tile_config = _swap_config_sides(config)

    row_origins = _tile_origins(big.shape[0], win, step)
    col_origins = _tile_origins(big.shape[1], win, step)
    origins = [(r0, c0) for r0 in row_origins for c0 in col_origins]
    n_tiles = len(origins)
    # tiles are tagged by their fixed position in `origins`, not by completion
    # order, so a tile's start and end lines carry the same number and pair up
    # even when several workers interleave them
    def _tag(k, r0, c0):
        return f"[{k + 1:>{len(str(n_tiles))}}/{n_tiles}] tile(r{r0},c{c0})"

    logger.info(
        f"Windowed coarse: {n_tiles} tile(s) of {win}px"
        f" ({len(row_origins)}x{len(col_origins)} grid, step {step}) over"
        f" {'img_left' if left_big else 'img_right'} {big.shape}, matching"
        f" {'img_right' if left_big else 'img_left'} {small.shape};"
        f" {n_workers} worker(s)"
    )

    def _eval_tile(k, r0, c0):
        # NOTE: cropping to a tile is a form of masked feature matching (it
        # localizes ORB detection to a subregion so the small portion's
        # correspondences aren't diluted by the whole image). An equivalent
        # variant would keep the full image and pass a mask to ORB
        # (`detect`/`detectAndCompute` accept one) via
        # register.cv2_feature_detect_and_match, which would also allow arbitrary
        # (e.g. segmentation-derived) region shapes. Cropping is used here for
        # simplicity and to preserve local resolution.
        tile = big[r0 : r0 + win, c0 : c0 + win]
        logger.info(f"  {_tag(k, r0, c0)} started")
        mx, n_match, cfg = search_then_register(
            small,
            tile,
            max_size=max_size,
            n_keypoints=n_keypoints,
            auto_mask=auto_mask,
            plot_match_result=False,
            return_match_count=True,
            return_config=True,
            config=tile_config,
        )
        # tile->small composed with big->tile gives big->small
        big2small = np.vstack([mx, [0, 0, 1]]) @ register_util.translate_mx(-c0, -r0)
        score = register_util.score_overlap(small, big, big2small, 1.0)
        # the nested search's own DEBUG lines carry no tile id, so with >1 worker
        # these bracketing lines are the only attributable record
        logger.info(f"  {_tag(k, r0, c0)} ncc={score:.3f} matches={n_match}")
        return (score, n_match, big2small, r0, c0, cfg)

    if n_workers == 1:
        results = [_eval_tile(k, r0, c0) for k, (r0, c0) in enumerate(origins)]
    else:
        # The per-tile work is CPU-bound OpenCV (ORB, BFMatcher, RANSAC) that
        # releases the GIL, and reads the shared `big`/`small` arrays read-only,
        # so a thread pool parallelizes without copying the (large) images.
        # OpenCV's own TBB/OpenMP threads are deliberately left at their default:
        # benchmarking showed pinning them to 1 is slower at every worker count
        # (they usefully fill the cores left idle while other tile threads are
        # blocked on the GIL during the numpy/skimage phases).
        import concurrent.futures

        # `register`'s per-match DEBUG lines (keypoint counts, two per config
        # tried) are emitted from every worker at once with nothing naming the
        # tile they belong to, so at DEBUG they shuffle into unreadable noise.
        # Muted only for the threaded loop -- the sequential branch above keeps
        # them, where they are attributable by position.
        logger.disable("palom.register")
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                results = list(
                    executor.map(
                        lambda ko: _eval_tile(ko[0], *ko[1]), enumerate(origins)
                    )
                )
        finally:
            logger.enable("palom.register")

    finite = [r for r in results if np.isfinite(r[0])]
    if not finite:
        logger.warning("Windowed registration found no valid tile; returning identity")
        return (np.eye(3)[:2], config) if return_config else np.eye(3)[:2]

    ranked = sorted(finite, key=lambda r: (r[0], r[1]), reverse=True)
    score, n_match, big2small, r0, c0, best_config = ranked[0]
    logger.info(
        f"Windowed coarse: best tile(r{r0},c{c0}) ncc={score:.3f}"
        f" matches={n_match}; {len(finite)}/{n_tiles} tile(s) scored"
        + (f", runner-up ncc={ranked[1][0]:.3f}" if len(ranked) > 1 else "")
    )

    if plot_match_result:
        # re-run the winning tile with the standard feature-match plot, then insert a
        # locator panel showing where the tile came from in the (agnostically chosen)
        # `big` image
        _plot_windowed_qc(
            small, big, r0, c0, win, score, n_match,
            which_big="img_left" if left_big else "img_right",
            max_size=max_size, n_keypoints=n_keypoints, auto_mask=auto_mask,
            config=tile_config,
        )

    # big->small back to moving(img_right)->ref(img_left)
    mx_full = np.linalg.inv(big2small) if left_big else big2small
    if return_config:
        # the winning tile's config, which when the tiles searched for themselves
        # is the strongest evidence about the pair in the whole run -- a tile that
        # actually contains the small scan matches on real tissue, where the
        # whole-image search is mostly comparing background. Learned from a
        # (small, tile) call, so undo the side swap `tile_config` applied.
        return mx_full[:2], (
            _swap_config_sides(best_config) if left_big else best_config
        )
    return mx_full[:2]


def _plot_windowed_qc(
    small, big, r0, c0, win, score, n_match, which_big,
    max_size, n_keypoints, auto_mask, config=None,
):
    """QC figure for the windowed route: the winning tile's feature-match plot, with a
    locator panel showing where that tile sits in `big`."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # standard crop-based feature-match figure for the winning tile
    search_then_register(
        small,
        big[r0 : r0 + win, c0 : c0 + win],
        max_size=max_size,
        n_keypoints=n_keypoints,
        auto_mask=auto_mask,
        plot_match_result=True,
        config=config,
    )
    fig = plt.gcf()
    main = fig.axes[0]
    main.set_title(f"{which_big} tile @ (r{r0}, c{c0})  ncc={score:.2f}  n={n_match}",
                   fontsize=6)
    # Lay both panels out in inches. The match panel is sized to its montage so
    # the aspect="equal" image fills its box exactly -- a taller box letterboxes
    # the image and opens a gap under the title. TOP_IN is left free for the axes
    # titles plus a suptitle the caller may add (`cli.align_he`).
    LEFT_IN, BOTTOM_IN, TOP_IN, GAP_IN, LOC_IN = 0.8, 0.4, 0.6, 0.15, 1.4
    mh, mw = main.images[0].get_array().shape[:2]
    # the match panel gets the same inches-per-image-pixel as a single-panel
    # coarse plot (`plot_util.size_axes_to_image`, which already sized this
    # figure), and the locator column is added around it -- rather than reading
    # back the figure width and dividing it up, which silently rescales the
    # match panel whenever that upstream sizing changes
    main_w = mw / plot_util.IMAGE_PX_PER_INCH
    main_h = main_w * mh / mw
    figw = LEFT_IN + main_w + GAP_IN + LOC_IN + 0.1
    figh = BOTTOM_IN + main_h + TOP_IN
    fig.set_size_inches(figw, figh)
    main.set_position(
        [LEFT_IN / figw, BOTTOM_IN / figh, main_w / figw, main_h / figh]
    )
    ax_loc = fig.add_axes([
        (LEFT_IN + main_w + GAP_IN) / figw, BOTTOM_IN / figh,
        LOC_IN / figw, main_h / figh,
    ])

    f = max(1, round(max(big.shape) / 500))
    bt = img_util.cv2_to_uint8(img_util.cv2_downscale_local_mean(big, f))
    # integer downscaled coords so the crop and its `extent` agree exactly
    rr0, cc0, wwin = r0 // f, c0 // f, max(1, win // f)
    crop = bt[rr0 : rr0 + wwin, cc0 : cc0 + wwin]
    # both panels share 0-255 limits; `bt` is uint8, so per-image autoscaling
    # would contrast-stretch the window differently from the background
    ax_loc.imshow(bt, cmap="gray", alpha=0.5, vmin=0, vmax=255)  # whole slide faded
    ax_loc.imshow(  # winning window at full opacity, placed at its location
        crop,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[cc0, cc0 + crop.shape[1], rr0 + crop.shape[0], rr0],
    )
    # `deepskyblue` matches the match-line color of the panel next to it; red is
    # reserved for status/error cues and loses contrast on grayscale tissue for
    # red-green color vision deficiency
    ax_loc.add_patch(
        Rectangle(
            (cc0, rr0), crop.shape[1], crop.shape[0],
            ec="deepskyblue", fc="none", lw=1,
        )
    )
    ax_loc.set_xlim(0, bt.shape[1])
    ax_loc.set_ylim(bt.shape[0], 0)
    ax_loc.set_title(f"window in {which_big}", fontsize=6)
    ax_loc.set_axis_off()
    ax_loc.set_anchor("N")  # keep it under its title instead of centered
    return fig


def _fignums():
    import matplotlib.pyplot as plt
    return tuple(plt.get_fignums())


def _close_figs_since(before):
    """Close figures opened since `before`, so an abandoned route leaves none.

    QC figures are collected by whoever saves them -- `cli.align_he.save_all_figs`
    sweeps every open figure, `MultiObjAligner._finish_new_figs` takes whatever
    appeared during the call -- so a figure for a route that lost has to be closed
    rather than merely ignored. `before=None` means nothing was plotted.
    """
    if before is None:
        return
    import matplotlib.pyplot as plt
    for num in set(plt.get_fignums()) - set(before):
        plt.close(num)


def coarse_register(
    img_left,
    img_right,
    pixel_size_left=None,
    pixel_size_right=None,
    matched_area_ratio=None,
    small_portion_area_ratio=0.25,
    min_match_count=15,
    min_ncc=0.10,
    window=None,
    window_margin=1.0,
    plot_match_result=False,
    n_workers=1,
    return_config=False,
    **search_kwargs,
):
    """Dispatch coarse alignment between the whole-image and windowed small-portion
    routes. Returns a 2x3 affine mapping `img_right` (moving) -> `img_left` (ref), or
    with `return_config`, that matrix and the intensity/orientation config the
    committed route matched with (see `search_then_register`).

    Decision (thresholds are hardcoded defaults, to be refined on more pairs):
    - If `matched_area_ratio` (smaller / larger physical-footprint area, in [0, 1]) is
      below `small_portion_area_ratio`, go straight to the windowed route (skip a
      whole-image attempt that would likely fail). When not given, it is computed from
      the nominal pixel sizes (`pixel_size_*`, um/pixel), which also set the tile size.
    - Otherwise try whole-image `search_then_register`, then verify confidence with the
      matched-keypoint count (`min_match_count`) and the whitened-intensity overlap
      correlation (`register_util.score_overlap`, `min_ncc`); if either is below
      threshold, fall back to the windowed route.

    When `plot_match_result` is True, a QC figure is drawn for the committed route
    only (the whole-image match plot, or the windowed match + locator plot)."""
    if (
        matched_area_ratio is None
        and pixel_size_left is not None
        and pixel_size_right is not None
    ):
        matched_area_ratio, _left_big, _footprint = matched_geometry(
            img_left.shape, img_right.shape, pixel_size_left, pixel_size_right
        )

    win_kwargs = dict(
        window=window,
        window_margin=window_margin,
        pixel_size_left=pixel_size_left,
        pixel_size_right=pixel_size_right,
        plot_match_result=plot_match_result,
        n_workers=n_workers,
        return_config=return_config,
    )
    if matched_area_ratio is not None and matched_area_ratio < small_portion_area_ratio:
        logger.info(
            f"matched_area_ratio {matched_area_ratio:.2f} < "
            f"{small_portion_area_ratio}; using windowed route"
        )
        return windowed_search_then_register(
            img_left, img_right, **win_kwargs, **search_kwargs
        )

    # Plot on the first pass and discard the figure if we end up not taking this
    # route, rather than re-running the match to draw it. The match is the
    # expensive part (ORB + RANSAC at `n_keypoints`), it runs on every coarse
    # registration in a run, and RANSAC is randomized -- so the re-run also drew
    # a figure that need not agree with the matrix actually returned.
    fignums_before = _fignums() if plot_match_result else None
    mx, n_match, cfg = search_then_register(
        img_left,
        img_right,
        plot_match_result=plot_match_result,
        return_match_count=True,
        return_config=True,
        **search_kwargs,
    )
    ncc = register_util.score_overlap(img_left, img_right, np.vstack([mx, [0, 0, 1]]))
    if n_match >= min_match_count and ncc >= min_ncc:
        return (mx, cfg) if return_config else mx
    _close_figs_since(fignums_before)
    logger.info(
        f"whole-image coarse weak (matches={n_match}, ncc={ncc:.3f}); "
        "falling back to windowed route"
    )
    return windowed_search_then_register(
        img_left, img_right, **win_kwargs, **search_kwargs
    )


if __name__ == "__main__":
    # `_swap_config_sides` must keep the histogram matching on the same image when
    # the argument order flips -- the invariant the windowed route relies on when it
    # reuses a pinned config for its (small, tile) calls.
    rng = np.random.default_rng(0)
    i1 = rng.random((32, 32), dtype="float32")
    i2 = rng.random((32, 32), dtype="float32") * 2
    m1, m2 = np.ones((32, 32), "bool"), np.ones((32, 32), "bool")

    for scalar in (1.0, -1.0):
        cfg = ("right", scalar, np.array)
        a1, a2 = match_img_with_config(i1, i2, m1, m2, *cfg)
        b1, b2 = match_img_with_config(i2, i1, m2, m1, *_swap_config_sides(cfg))
        # same images, swapped positions: i2 stays the adjusted one in both
        assert np.allclose(a1, b2), scalar
        assert np.allclose(a2, b1), scalar

    assert _swap_config_sides(("left", -1.0, np.flipud)) == ("right", -1.0, np.flipud)
    assert _swap_config_sides(_swap_config_sides(cfg)) == cfg
    print("register_coarse self-check OK")
