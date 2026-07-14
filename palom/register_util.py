import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters
import skimage.transform

from . import img_util


IS_BF_IMG = img_util.is_brightfield_img


def match_quantized(source, template, levels=65536):
    """Fast drop-in replacement for `skimage.exposure.match_histograms`.

    Quantizes `source` and `template` onto `levels` bins spanning each image's
    own [min, max] with `cv2.normalize`, builds the CDFs with `cv2.calcHist`
    (both multi-threaded, no full sort), and applies the value mapping by direct
    integer indexing. Handles integer, [0, 1]-normalized and continuous float
    inputs alike; ~7-9x faster than skimage with <0.01% intensity error on the
    WSI-scale inputs used for coarse registration.
    """
    source = np.asarray(source)
    template = np.asarray(template)
    sf = np.ascontiguousarray(source.reshape(-1, 1), dtype=np.float32)
    tf = np.ascontiguousarray(template.reshape(-1, 1), dtype=np.float32)
    tlo, thi, _, _ = cv2.minMaxLoc(tf)
    # scale each image onto [0, levels-1] with cv2 (multi-threaded); calcHist
    # floor-bins, so reuse the same floor for the apply index -> consistent
    # quantizer between the CDF and the lookup
    sq = cv2.normalize(sf, None, 0, levels - 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    tq = cv2.normalize(tf, None, 0, levels - 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    src_hist = cv2.calcHist([sq], [0], None, [levels], [0, levels]).ravel()
    tmpl_hist = cv2.calcHist([tq], [0], None, [levels], [0, levels]).ravel()
    src_cdf = np.cumsum(src_hist) / sf.size
    tmpl_cdf = np.cumsum(tmpl_hist) / tf.size
    tmpl_vals = tlo + np.arange(levels) * (thi - tlo) / (levels - 1)
    lut = np.interp(src_cdf, tmpl_cdf, tmpl_vals).astype(np.float32)
    si = sq.astype(np.int32)
    np.clip(si, 0, levels - 1, out=si)
    return lut[si.ravel()].reshape(source.shape)


def _cv2_image_histogram(img, nbins=256):
    lo, hi = float(img.min()), float(img.max())
    # cv2's upper range is exclusive; nudge it so the max value lands in the last
    # bin, matching skimage's inclusive `np.histogram`
    hi_ex = float(np.nextafter(np.float32(hi), np.float32(hi + 1)))
    hist = cv2.calcHist(
        [np.ascontiguousarray(img.reshape(-1, 1), dtype=np.float32)],
        [0], None, [nbins], [lo, hi_ex]
    ).ravel()
    edges = np.linspace(lo, hi, nbins + 1)
    return hist, (edges[:-1] + edges[1:]) / 2


def _threshold_triangle_from_hist(hist, bin_centers):
    # `skimage.filters.threshold_triangle` reimplemented on a precomputed
    # histogram so it can share `_cv2_image_histogram` with the otsu threshold
    nbins = len(hist)
    arg_peak = int(np.argmax(hist))
    peak_h = float(hist[arg_peak])
    nz = np.flatnonzero(hist)
    arg_low, arg_high = int(nz[0]), int(nz[-1])
    flip = arg_peak - arg_low < arg_high - arg_peak
    if flip:
        hist = hist[::-1]
        arg_low = nbins - arg_high - 1
        arg_peak = nbins - arg_peak - 1
    width = arg_peak - arg_low
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low]
    norm = np.sqrt(peak_h ** 2 + width ** 2)
    length = (peak_h / norm) * x1 - (width / norm) * y1
    arg_level = int(np.argmax(length)) + arg_low
    if flip:
        arg_level = nbins - arg_level - 1
    return bin_centers[arg_level]


def otsu_triangle_thresholds(img, nbins=256):
    """Otsu and triangle thresholds from a single cv2-built histogram.

    Numerically identical to `skimage.filters.threshold_otsu`/`threshold_triangle`
    but ~7x faster: the histogram (the expensive full-image pass) is built once
    with the multi-threaded `cv2.calcHist` and shared by both thresholds.
    """
    hist, centers = _cv2_image_histogram(img, nbins)
    otsu = skimage.filters.threshold_otsu(hist=(hist.astype(np.int64), centers))
    triangle = _threshold_triangle_from_hist(hist, centers)
    return otsu, triangle


def make_img_pairs(img1, img2, auto_invert_intensity=True, auto_mask=False):
    img1 = np.asarray(img1).astype(np.float32)
    img2 = np.asarray(img2).astype(np.float32)
    compare_funcs = [
        np.less if IS_BF_IMG(i) else np.greater
        for i in (img1, img2)
    ]
    if not auto_invert_intensity:
        compare_funcs = [compare_funcs[0]]*2
    thresholds = [otsu_triangle_thresholds(i) for i in (img1, img2)]
    imgs_otsu = [
        f(i, otsu).astype(np.uint8)
        for (i, f, (otsu, _tri)) in zip((img1, img2), compare_funcs, thresholds)
    ]
    imgs_tri = [
        f(i, tri).astype(np.uint8)
        for (i, f, (_otsu, tri)) in zip((img1, img2), compare_funcs, thresholds)
    ]
    if auto_invert_intensity:
        img1, img2 = match_bf_fl_histogram(img1, img2, auto_mask)
    else:
        match_func = match_quantized
        if auto_mask:
            match_func = masked_match_histograms
        img2 = match_func(img2, img1)

    imgs_whiten = [
        img_util.whiten(i, 1)
        for i in (img1, img2)
    ]
    return [
        imgs_otsu,
        imgs_tri,
        (img1, img2),
        imgs_whiten
    ]


def match_bf_fl_histogram(img1, img2, auto_mask=False):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    # TODO does it make a difference to min/max rescale before histogram
    # matching?
    is_bf_img1, is_bf_img2 = [
        IS_BF_IMG(i)
        for i in (img1, img2)
    ]
    match_func = match_quantized
    if auto_mask:
        match_func = masked_match_histograms
    if is_bf_img1 == is_bf_img2:
        return img1, match_func(img2, img1)
    elif is_bf_img1:
        return img1, match_func(-img2, img1)
    elif is_bf_img2:
        return match_func(-img1, img2), img2


def masked_match_histograms(img, ref_img):
    # Distinct from register_coarse.masked_match_histograms (see the note there):
    # this variant takes no explicit masks, derives them from a ~500px
    # downsample, and matches with `match_quantized`. Kept separate on purpose.
    if IS_BF_IMG(img) != IS_BF_IMG(ref_img):
        bg = ['dark', 'light']
        print('`img` and `ref_img` detected as different types:')
        print(f"    `img` detected as {bg[IS_BF_IMG(img)]}-background image")
        print(f"    `ref_img` detected as {bg[IS_BF_IMG(ref_img)]}-background image")

    # downsize images to ~1000 px for speed
    shape_max = max(*img.shape, *ref_img.shape)
    downsize_factor = int(np.floor(shape_max / 500))
    if downsize_factor < 1:
        downsize_factor = 1
    mask = img_util.entropy_mask(
        img_util.cv2_downscale_local_mean(img, downsize_factor)
    )
    ref_mask = img_util.entropy_mask(
        img_util.cv2_downscale_local_mean(ref_img, downsize_factor)
    )
    repeats = (downsize_factor, downsize_factor)
    shape = img.shape
    ref_shape = ref_img.shape
    mask = img_util.repeat_2d(mask, repeats)[:shape[0], :shape[1]]
    ref_mask = img_util.repeat_2d(ref_mask, repeats)[:ref_shape[0], :ref_shape[1]]
    matched_img = np.zeros_like(img)
    # NOTE this does not handle inverted matching, both image must be the same
    # type. E.g. dark background, light signal
    matched_img[mask] = match_quantized(
        img[mask], ref_img[ref_mask]
    )
    matched_img[~mask] = ref_img[~ref_mask].mean()
    # matched_img[~mask] = skimage.exposure.histogram_matching._match_cumulative_cdf(
    #     img[~mask], ref_img[~ref_mask]
    # )
    return matched_img


def plot_img_keypoints(imgs, keypoints):
    fig, axs = plt.subplots(1, len(imgs))
    for i, k, a in zip(imgs, keypoints, axs):
        a.imshow(cv2.drawKeypoints(
            i, k, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        ))
        a.set_title(len(k))
    return


def get_flip_mx(img_shape, flip_axis):
    assert flip_axis in [0, 1, (0, 1), (1, 0)]
    mx = np.eye(3)
    offset_xy = np.array(img_shape)[::-1] - 1
    if type(flip_axis) == int:
        index = int(not flip_axis)
        mx[index, index] = -1
        mx[index, 2] = offset_xy[index]
        return mx
    mx[:2, :2] *= -1
    mx[:2, 2] = offset_xy
    return mx


def get_rot90_mx(img_shape, k):
    assert k in range(4)
    degree = -k*90
    h, w = img_shape
    translation = {
        0: (0, 0),
        1: (0, w-1),
        2: (w-1, h-1),
        3: (h-1, 0)
    }
    return skimage.transform.AffineTransform(
        rotation=np.deg2rad(degree),
        translation=translation[k]
    ).params


def translate_mx(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)


def score_overlap(ref, moving, mx, sigma=1.0):
    """Whitened-intensity correlation between `ref` and `moving` warped by `mx`
    (mov -> ref), over their overlap. Sign-tolerant (abs) for cross-modality.
    A modality-robust coarse-alignment confidence metric shared by the feature-based
    and Fourier-Mellin coarse routes."""
    warped = skimage.transform.warp(
        moving,
        skimage.transform.AffineTransform(matrix=mx).inverse,
        output_shape=ref.shape,
        preserve_range=True,
    ).astype("float32")
    mask = warped != 0
    if mask.sum() < 0.02 * mask.size:
        return -np.inf
    rw = img_util.whiten(ref, sigma)[mask]
    ww = img_util.whiten(warped, sigma)[mask]
    if rw.std() == 0 or ww.std() == 0:
        return -np.inf
    return float(abs(np.corrcoef(rw, ww)[0, 1]))
