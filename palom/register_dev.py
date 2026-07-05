import itertools

import cv2
import numpy as np
from loguru import logger

from . import img_util, register_util, register


def masked_match_histograms(img, ref_img, mask=None, ref_mask=None):
    import skimage.exposure.histogram_matching

    if mask is None:
        mask = img_util.entropy_mask(img)
    if ref_mask is None:
        ref_mask = img_util.entropy_mask(ref_img)

    matched_img = np.zeros_like(img)
    matched_img[mask] = skimage.exposure.histogram_matching._match_cumulative_cdf(
        img[mask], ref_img[ref_mask]
    )
    matched_img[~mask] = ref_img[~ref_mask].mean()
    return matched_img


def match_img_with_config(img1, img2, mask1, mask2, adjust_which, scalar, func):
    assert adjust_which in ("left", "right")
    if adjust_which == "right":
        return img1, func(masked_match_histograms(scalar * img2, img1, mask2, mask1))
    else:
        return masked_match_histograms(scalar * img1, img2, mask1, mask2), func(img2)


def search_best_match_config(
    img_left,
    img_right,
    max_size=500,
    auto_mask=True,
    n_keypoints=2000,
    min_fold_increase=5,
):
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
        return results[idx]
    if downsize_factor == 1:
        return results[idx]

    return search_best_match_config(
        img_left=img_left,
        img_right=img_right,
        max_size=2 * max_size,
        auto_mask=auto_mask,
        n_keypoints=n_keypoints,
        min_fold_increase=min_fold_increase,
    )


def search_then_register(
    img_left,
    img_right,
    max_size=2000,
    n_keypoints=5000,
    auto_mask=True,
    plot_match_result=True,
    search_kwargs=None,
    return_match_count=False,
):
    search_kwargs = search_kwargs or {}
    img1 = img_left.astype("float32")
    img2 = img_right.astype("float32")

    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.ceil(shape_max / max_size))

    img1 = img_util.cv2_downscale_local_mean(img1, downsize_factor)
    img2 = img_util.cv2_downscale_local_mean(img2, downsize_factor)

    _, config = search_best_match_config(img1, img2, **search_kwargs)
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
    if return_match_count:
        return mx_full_res[:2], int(match.sum())
    return mx_full_res[:2]


def _tile_origins(length, window, step):
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
    window_margin=1.4,
    max_size=2000,
    n_keypoints=5000,
    auto_mask=True,
    plot_match_result=False,
    n_workers=1,
):
    """Coarse alignment for small-portion pairs (one scan images only a fraction of
    the other), where whole-image feature matching struggles.

    Slides a `window`x`window` tile over the *larger* image, runs
    `search_then_register(smaller, tile)` on each (feature matching handles the
    per-tile flip/rotation/scale), composes with the tile offset to get a full-image
    affine, warps the larger image into the smaller one's frame, and keeps the tile
    whose affine maximizes an intensity-permuted overlap NCC (tie-broken by matched-
    keypoint count). Returns a 2x3 affine mapping `img_right` (moving) -> `img_left`
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

    def _eval_tile(r0, c0):
        # NOTE: cropping to a tile is a form of masked feature matching (it
        # localizes ORB detection to a subregion so the small portion's
        # correspondences aren't diluted by the whole image). An equivalent
        # variant would keep the full image and pass a mask to ORB
        # (`detect`/`detectAndCompute` accept one) via
        # register.cv2_feature_detect_and_match, which would also allow arbitrary
        # (e.g. segmentation-derived) region shapes. Cropping is used here for
        # simplicity and to preserve local resolution.
        tile = big[r0 : r0 + win, c0 : c0 + win]
        mx, n_match = search_then_register(
            small,
            tile,
            max_size=max_size,
            n_keypoints=n_keypoints,
            auto_mask=auto_mask,
            plot_match_result=False,
            return_match_count=True,
        )
        # tile->small composed with big->tile gives big->small
        big2small = np.vstack([mx, [0, 0, 1]]) @ register_util.translate_mx(-c0, -r0)
        score = register_util.score_overlap(small, big, big2small, 1.0)
        logger.debug(f"tile(r{r0},c{c0}) ncc={score:.3f} matches={n_match}")
        return (score, n_match, big2small, r0, c0)

    origins = [
        (r0, c0)
        for r0 in _tile_origins(big.shape[0], win, step)
        for c0 in _tile_origins(big.shape[1], win, step)
    ]
    if n_workers == 1:
        results = [_eval_tile(r0, c0) for r0, c0 in origins]
    else:
        # The per-tile work is CPU-bound OpenCV (ORB, BFMatcher, RANSAC) that
        # releases the GIL, and reads the shared `big`/`small` arrays read-only,
        # so a thread pool parallelizes without copying the (large) images.
        # OpenCV's own TBB/OpenMP threads are deliberately left at their default:
        # benchmarking showed pinning them to 1 is slower at every worker count
        # (they usefully fill the cores left idle while other tile threads are
        # blocked on the GIL during the numpy/skimage phases).
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers
        ) as executor:
            results = list(executor.map(lambda o: _eval_tile(*o), origins))

    finite = [r for r in results if np.isfinite(r[0])]
    if not finite:
        logger.warning("Windowed registration found no valid tile; returning identity")
        return np.eye(3)[:2]

    score, n_match, big2small, r0, c0 = max(finite, key=lambda r: (r[0], r[1]))

    if plot_match_result:
        # re-run the winning tile with the standard feature-match plot, then insert a
        # locator panel showing where the tile came from in the (agnostically chosen)
        # `big` image
        _plot_windowed_qc(
            small, big, r0, c0, win, score, n_match,
            which_big="img_left" if left_big else "img_right",
            max_size=max_size, n_keypoints=n_keypoints, auto_mask=auto_mask,
        )

    # big->small back to moving(img_right)->ref(img_left)
    mx_full = np.linalg.inv(big2small) if left_big else big2small
    return mx_full[:2]


def _plot_windowed_qc(
    small, big, r0, c0, win, score, n_match, which_big,
    max_size, n_keypoints, auto_mask,
):
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
    )
    fig = plt.gcf()
    main = fig.axes[0]
    main.set_title(f"{which_big} tile @ (r{r0}, c{c0})  ncc={score:.2f}  n={n_match}",
                   fontsize=6)
    # shrink the match axes to make room, then insert the locator on the right
    p = main.get_position()
    main.set_position([p.x0, p.y0, p.width * 0.72, p.height])
    ax_loc = fig.add_axes([p.x0 + p.width * 0.76, p.y0, p.width * 0.24, p.height])

    f = max(1, round(max(big.shape) / 500))
    bt = img_util.cv2_to_uint8(img_util.cv2_downscale_local_mean(big, f))
    rr0, cc0, wwin = r0 / f, c0 / f, win / f
    ax_loc.imshow(bt, cmap="gray", alpha=0.5)  # whole slide faded
    ax_loc.imshow(  # winning window at full opacity, placed at its location
        bt[int(rr0) : int(rr0 + wwin), int(cc0) : int(cc0 + wwin)],
        cmap="gray",
        extent=[cc0, cc0 + wwin, rr0 + wwin, rr0],
    )
    ax_loc.add_patch(Rectangle((cc0, rr0), wwin, wwin, ec="r", fc="none", lw=1))
    ax_loc.set_xlim(0, bt.shape[1])
    ax_loc.set_ylim(bt.shape[0], 0)
    ax_loc.set_title(f"window in {which_big}", fontsize=6)
    ax_loc.set_axis_off()
    return fig


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
    **search_kwargs,
):
    """Dispatch coarse alignment between the whole-image and windowed small-portion
    routes. Returns a 2x3 affine mapping `img_right` (moving) -> `img_left` (ref).

    Decision (thresholds are hardcoded defaults, to be refined on more pairs):
    - If `matched_area_ratio` (smaller / larger physical-footprint area, in [0, 1]) is
      below `small_portion_area_ratio`, go straight to the windowed route (skip a
      whole-image attempt that would likely fail). When not given, it is computed from
      the nominal pixel sizes (`pixel_size_*`, um/pixel), which also set the tile size.
    - Otherwise try whole-image `search_then_register`, then verify confidence with
      the matched-keypoint count and an intensity-permuted overlap NCC; if either is
      below threshold, fall back to the windowed route.

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
    )
    if matched_area_ratio is not None and matched_area_ratio < small_portion_area_ratio:
        logger.info(
            f"matched_area_ratio {matched_area_ratio:.2f} < "
            f"{small_portion_area_ratio}; using windowed route"
        )
        return windowed_search_then_register(
            img_left, img_right, **win_kwargs, **search_kwargs
        )

    mx, n_match = search_then_register(
        img_left,
        img_right,
        plot_match_result=False,
        return_match_count=True,
        **search_kwargs,
    )
    ncc = register_util.score_overlap(img_left, img_right, np.vstack([mx, [0, 0, 1]]))
    if n_match >= min_match_count and ncc >= min_ncc:
        if plot_match_result:
            # re-run the committed whole-image result for its QC figure
            search_then_register(
                img_left, img_right, plot_match_result=True, **search_kwargs
            )
        return mx
    logger.info(
        f"whole-image coarse weak (matches={n_match}, ncc={ncc:.3f}); "
        "falling back to windowed route"
    )
    return windowed_search_then_register(
        img_left, img_right, **win_kwargs, **search_kwargs
    )
