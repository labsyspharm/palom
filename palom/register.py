import functools
import inspect
import itertools
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import skimage.exposure
import skimage.feature
import skimage.registration
from loguru import logger

from . import img_util, plot_util, register_util

if hasattr(skimage.registration, 'phase_cross_correlation'):
    register_translation = skimage.registration.phase_cross_correlation
else:
    register_translation = skimage.feature.register_translation


#
# Image-based registration
#
def phase_cross_correlation(img1, img2, sigma, upsample=10, module='skimage'):
    assert module in ['cv2', 'skimage']

    if (np.unique(img1).size == 1) | (np.unique(img2).size == 1):
        # FIXME does this make sense?
        return (np.inf, np.inf), np.inf
    
    img1w = img_util.whiten(img1, sigma)
    img2w = img_util.whiten(img2, sigma)
   
    if module == 'skimage':
        with warnings.catch_warnings():
            warnings.filterwarnings(
                # two patterns observed
                # 1. invalid value encountered in true_divide
                # 2. invalid value encountered in divide
                'ignore', 'invalid value encountered in',
                RuntimeWarning,
            )
            kwargs = dict(upsample_factor=upsample)
            # `normalization` kwarg was introduced in skimage v0.19
            if 'normalization' in inspect.signature(register_translation).parameters:
                kwargs.update(normalization=None)
            shift, _error, _phasediff = register_translation(
                img1w, img2w, **kwargs
            )
   
    elif module == 'cv2':
        shift_xy, _response = cv2.phaseCorrelate(img1w, img2w)
        shift = shift_xy[::-1]

    # At this point we may have a shift in the wrong quadrant since the FFT
    # assumes the signal is periodic. We test all four possibilities and return
    # the shift that gives the highest direct correlation (sum of products).
    shape = np.array(img1.shape)
    shift_pos = (shift + shape) % shape
    shift_neg = shift_pos - shape
    shifts = list(itertools.product(*zip(shift_pos, shift_neg)))
    correlations = [
        np.abs(np.sum(img1w * cv2_translate(img2w, s)))
        for s in shifts
    ]
    idx = np.argmax(correlations)
    shift = shifts[idx]
    correlation = correlations[idx]
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        error = -np.log(correlation / total_amplitude)
    else:
        error = np.inf
    return shift, error


def cv2_translate(img, shift):
    assert img.ndim == len(shift) == 2
    sy, sx = shift
    return cv2.warpAffine(
        img,   
        np.array([[1, 0, sx], [0, 1, sy]], dtype=float),
        img.shape[::-1]
    )


def normalized_phase_correlation(img1, img2, sigma):
    w1 = img_util.whiten(img1, sigma)
    w2 = img_util.whiten(img2, sigma)
    corr = scipy.fft.fftshift(np.abs(scipy.fft.ifft2(
        scipy.fft.fft2(w1) * scipy.fft.fft2(w2).conj()
    )))
    corr /= (np.linalg.norm(w1) * np.linalg.norm(w2))
    return corr


#
# Feature-based registration
#
# `cv2.estimateAffine2D` needs three non-degenerate correspondences. Measured
# against opencv 4.11: an empty array RAISES; 1-2 points return a None matrix
# alongside a real mask; collinear or coincident points return a None matrix
# and an all-zero mask. So the matrix -- never the mask -- is the failure
# signal, and the empty case has to be kept away from cv2 entirely.
MIN_AFFINE_POINTS = 3


def _no_point_pairs():
    """An empty correspondence set, shaped and typed so callers can `vstack` it.

    Returned wherever a match cannot be attempted. The previous sentinel was
    `np.empty((1, 2))` -- uninitialized memory -- so a crop that came back
    without keypoints fed two random coordinates into `ensambled_match`'s
    pooled RANSAC, and made `len(p_src)` truthy in
    `register_coarse.search_best_match_config`. Both are routine: a masked
    per-object thumbnail is mostly constant fill, and the windowed coarse route
    scores tiles that land entirely on background.
    """
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    )


def ensambled_match(
    img_left, img_right,
    n_keypoints=1000, plot_match_result=False,
    plot_individual_result=False, ransacReprojThreshold=5,
    return_match_mask=False,
    # try to infer imaging modalities from the images and perform intensity
    # inversion when needed; turn this off if the imaging modalities are known
    # to be the same
    auto_invert_intensity=True,
    # apply entropy mask during histogram matching
    # turn this off if the image is not a WSI
    auto_mask=False
):
    img_pairs = register_util.make_img_pairs(
        img_left, img_right,
        auto_invert_intensity=auto_invert_intensity,
        auto_mask=auto_mask
    )
    img_left, img_right = img_pairs[2]

    all_found = [
        cv2_feature_detect_and_match(
            *img_pair, n_keypoints=n_keypoints,
            plot_match_result=plot_individual_result
        )
        for img_pair in img_pairs
    ]
    all_src = np.vstack([i[0] for i in all_found])
    all_dst = np.vstack([i[1] for i in all_found])

    # `mask` is always an array with one row per pooled point, the shape cv2
    # itself returns, so `match.sum()` and `mask.flatten() > 0` are safe for
    # every caller. `t_matrix is None` stays the single failure signal --
    # `register_coarse.search_then_register` already tests exactly that.
    mask = np.zeros((len(all_src), 1), dtype=np.uint8)
    t_matrix = None
    if len(all_src) < MIN_AFFINE_POINTS:
        # `estimateAffine2D` RAISES on an empty array (it only returns
        # (None, None) for 1-2 points), and every representation coming back
        # empty is now reachable -- it used to be masked by the garbage
        # sentinel that `_no_point_pairs` replaced
        logger.debug(
            f"Only {len(all_src)} pooled correspondence(s) across the"
            f" {len(img_pairs)} representations; skipping the affine fit"
        )
    else:
        t_matrix, _mask = cv2.estimateAffine2D(
            all_dst, all_src,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransacReprojThreshold,
            maxIters=5000
        )
        if t_matrix is None:
            logger.debug(
                f"RANSAC found no consistent affine among {len(all_src)}"
                " pooled correspondences"
            )
        else:
            mask = _mask
    # nothing to draw without a fit, and `plot_matches` on an empty keypoint
    # set is not worth the special case -- callers bracket their plotting calls
    # by a fignum diff (`align_multi_obj._finish_new_figs`), so drawing no
    # figure is expected
    if plot_match_result and (t_matrix is not None):
        _, ax = plt.subplots()

        def _rescale_img(img):
            return np.log(
                skimage.exposure.rescale_intensity(
                    img,
                    in_range=tuple(np.percentile(img, [1, 99])),
                    out_range=(500, 5000),
                )
            )

        pimg_left = _rescale_img(img_left)
        pimg_right = _rescale_img(img_right)
        skimage.feature.plot_matches(
            ax,
            pimg_left,
            pimg_right,
            np.fliplr(all_src),
            np.fliplr(all_dst),
            np.arange(len(all_src)).repeat(2).reshape(-1, 2)[mask.flatten() > 0],
            keypoints_color=np.divide([255, 215, 0, 50], 255),
            matches_color="deepskyblue",
            only_matches=False,
        )
        ax.images[0].set_clim(min(pimg_left.min(), pimg_right.min()))
        for line in ax.get_lines():
            line.set_alpha(0.5)
            line.set_linewidth(.5)
        # sized here, at the one place this figure is created, so every coarse
        # match plot comes out at the same scale -- whole-slide and per-object
        # alike. Callers only add titles.
        plot_util.size_axes_to_image(ax)

    return (t_matrix, mask) if return_match_mask else t_matrix


def cv2_feature_detect_and_match(
    img_left, img_right, n_keypoints=1000,
    plot_keypoint_result=False, plot_match_result=False
):
    img_left, img_right = [
        img_util.cv2_to_uint8(i)
        for i in (img_left, img_right)
    ]
    descriptor_extractor = cv2.ORB_create(n_keypoints, edgeThreshold=0)

    keypoints_left, descriptors_left = descriptor_extractor.detectAndCompute(
        np.dstack(3*(img_left,)), None
    )
    keypoints_right, descriptors_right = descriptor_extractor.detectAndCompute(
        np.dstack(3*(img_right,)), None
    )
    if plot_keypoint_result == True:
        register_util.plot_img_keypoints(
            [img_left, img_right], [keypoints_left, keypoints_right]
        )
    logger.debug(f"keypts L:{len(keypoints_left)}, keypts R:{len(keypoints_right)}")
    if len(keypoints_left) == 0 or len(keypoints_right) == 0:
        return _no_point_pairs()

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(descriptors_left, descriptors_right)
    if len(matches) < MIN_AFFINE_POINTS:
        # `np.float32([])` below would be shape (0,) rather than (0, 2), and
        # there is nothing to fit anyway
        logger.debug(f"Only {len(matches)} cross-checked match(es); no fit")
        return _no_point_pairs()

    src_pts = np.float32(
        [keypoints_left[m.queryIdx].pt for m in matches]
    )
    dst_pts = np.float32(
        [keypoints_right[m.trainIdx].pt for m in matches]
    )
    t_matrix, mask = cv2.estimateAffine2D(
        dst_pts, src_pts,
        method=cv2.RANSAC, ransacReprojThreshold=30, maxIters=5000
    )
    if t_matrix is None or mask is None:
        # No consistent subset -- collinear or coincident correspondences.
        # cv2 4.11 returns an all-zero mask here rather than None, so the old
        # code already fell through to an empty selection; the `mask is None`
        # half is defensive, since the binding maps an empty Mat to None and
        # pyproject pins no upper bound on opencv. Returning early also keeps
        # `plot_match_result` from drawing a figure with nothing in it.
        logger.debug(
            f"RANSAC found no consistent affine among {len(matches)} matches"
        )
        return _no_point_pairs()
    if plot_match_result == True:
        plt.figure()
        imgmatch_ransac = cv2.drawMatches(
            img_left, keypoints_left,
            img_right, keypoints_right,
            matches, None,
            matchColor=(0, 255, 0), singlePointColor=None,
            matchesMask=mask.flatten(),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )
        plt.gca().imshow(imgmatch_ransac)
    return src_pts[mask.flatten()>0], dst_pts[mask.flatten()>0]


def match_test_flip_rotate(img_left, img_right, auto_mask=False):

    flip_funcs = [np.array] + [
        functools.partial(np.flip, axis=aa)
        for aa in (0, 1, (0, 1))
    ]
    rotate_funcs = [
        functools.partial(np.rot90, k=i)
        for i in range(4)
    ]
    flip_mxs = [np.eye(3)] + [
        register_util.get_flip_mx(img_right.shape, aa)
        for aa in (0, 1, (0, 1))
    ]
    rotate_mxs = [
        register_util.get_rot90_mx(img_right.shape, i)
        for i in range(4)
    ]

    # downsize images to ~500 px for speed
    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.floor(shape_max / 500))
    if downsize_factor > 1:
        img_left = img_util.cv2_downscale_local_mean(img_left, downsize_factor)
        img_right = img_util.cv2_downscale_local_mean(img_right, downsize_factor)

    n_matches = [
        ensambled_match(
            img_left, rr(ff(img_right)), return_match_mask=True, auto_mask=auto_mask
        )[1].sum()
        # only need half of the 4x4 combinations
        for ff, rr in itertools.product(flip_funcs[:2], rotate_funcs)
    ]
    best_flip, best_rotate = np.unravel_index(
        np.argmax(n_matches), (2, 4)
    )
    print(np.array(n_matches, int).reshape(2, 4))
    print(best_flip, best_rotate)

    # construct best flip and rotate func
    ff, rr = flip_funcs[best_flip], rotate_funcs[best_rotate]
    def flip_rotate_func(target_img):
        return rr(ff(target_img))

    return flip_rotate_func, rotate_mxs[best_rotate] @ flip_mxs[best_flip]
