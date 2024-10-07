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
    n_keypoints=5000,
    auto_mask=True,
    plot_match_result=True,
    search_kwargs=None,
):
    search_kwargs = search_kwargs or {}
    img1 = img_left.astype("float32")
    img2 = img_right.astype("float32")
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
        mx_flip = register_util.get_flip_mx(img_right.shape, 0)
    if mx is None:
        logger.warning(
            "Feature matching failed. Returning identity matrix as placeholder"
        )
        mx = np.eye(3)[:2]
        match = np.zeros(1, "bool")
    mx = (np.vstack([mx, [0, 0, 1]]) @ mx_flip)[:2, :]
    logger.debug(
        f"{match.sum():6} matches; {n_keypoints:6} keypoints; mask: {auto_mask}"
    )
    return mx
