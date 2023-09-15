import functools
import inspect
import itertools
import logging
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import skimage.feature
import skimage.registration
import skimage.exposure

from . import img_util, register_util

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
def feature_based_registration(
    img_left, img_right,
    test_flip=False, test_intensity_invert=False,
    n_keypoints=1000, plot_match_result=False,
    plot_individual_result=False, ransacReprojThreshold=5,
    # apply entropy mask during histogram matching
    # turn this off if the image is not a WSI
    auto_mask=False,
    kwarg_test_flip_and_intensity_invert=None
):
    _kwarg_test_flip_and_intensity_invert = {'auto_mask': auto_mask}
    if kwarg_test_flip_and_intensity_invert is None:
        kwarg_test_flip_and_intensity_invert = {}

    if test_flip == test_intensity_invert == True:
        kwarg_test_flip_and_intensity_invert['test_only'] = None
    if test_flip != test_intensity_invert:
        test_only = 'flip'
        if test_intensity_invert:
            test_only = 'intensity_invert'
        kwarg_test_flip_and_intensity_invert['test_only'] = test_only
    
    flip_func = np.array
    intensity_scalar = 1
    auto_invert_intensity = True
    mx_fr = np.eye(3)

    if test_flip | test_intensity_invert:
        test_result = test_flip_and_intensity_invert(
            img_left, img_right,
            **{
                **_kwarg_test_flip_and_intensity_invert,
                **kwarg_test_flip_and_intensity_invert
            }
        )
        flip_func = test_result['flip_func']
        intensity_scalar = test_result['intensity_scalar']
        if test_intensity_invert:
            auto_invert_intensity = False
        if flip_func != np.array:
            mx_fr = register_util.get_flip_mx(img_right.shape, 0)

    img_right = flip_func(img_right * intensity_scalar)
    mx_affine = ensambled_match(
        img_left, img_right,
        n_keypoints, plot_match_result,
        plot_individual_result, ransacReprojThreshold,
        auto_invert_intensity=auto_invert_intensity,
        auto_mask=auto_mask,
    )
    mx_affine = (np.vstack([mx_affine, [0, 0, 1]]) @ mx_fr)[:2, :]
    return mx_affine


def test_flip_and_intensity_invert(
        img_left, img_right,
        max_size=500,
        auto_mask=False,
        test_only=None
    ):
    assert test_only in [None, 'flip', 'intensity_invert']
    flip_funcs = [np.array, np.flipud]
    intensity_scalars = [1, -1]
    
    if test_only == 'flip':
        intensity_scalars.pop(1)
    
    if test_only == 'intensity_invert':
        flip_funcs.pop(1)
    
    # downsize images to ~500 px for speed
    shape_max = max(*img_left.shape, *img_right.shape)
    downsize_factor = int(np.floor(shape_max / max_size))
    if downsize_factor > 1:
        img_left = img_util.cv2_downscale_local_mean(img_left, downsize_factor)
        img_right = img_util.cv2_downscale_local_mean(img_right, downsize_factor)
    
    match_results = []
    for flip_func, scalar in itertools.product(
            flip_funcs, intensity_scalars
    ):
        n_matches = ensambled_match(
            img_left,
            flip_func(scalar * img_right),
            return_match_mask=True,
            auto_invert_intensity=False,
            auto_mask=auto_mask
        )[1].sum()
        match_results.append((n_matches, flip_func, scalar))
    for rr in match_results: print(rr)
    best = sorted(match_results, key=lambda x: x[0], reverse=True)[0]
    return {'flip_func': best[1], 'intensity_scalar': best[2]}


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

    t_matrix, mask = cv2.estimateAffine2D(
        all_dst, all_src,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransacReprojThreshold,
        maxIters=5000
    )
    if plot_match_result == True:
        _, ax = plt.subplots()
        skimage.feature.plot_matches(
            ax,
            np.log(skimage.exposure.rescale_intensity(img_left, out_range=(1, 4095))),
            np.log(skimage.exposure.rescale_intensity(img_right, out_range=(1, 4095))),
            np.fliplr(all_src), np.fliplr(all_dst),
            np.arange(len(all_src)).repeat(2).reshape(-1, 2)[mask.flatten()>0],
            keypoints_color=np.divide([255, 215, 0, 50], 255),
            matches_color='deepskyblue',
            only_matches=False
        )
        for line in ax.get_lines():
            line.set_alpha(0.5)
            line.set_linewidth(.5)

    return (t_matrix, mask) if return_match_mask else t_matrix


def cv2_feature_detect_and_match(
    img_left, img_right, n_keypoints=1000,
    plot_keypoint_result=False, plot_match_result=False
):
    img_left, img_right = [
        img_util.cv2_to_uint8(i)
        for i in (img_left, img_right)
    ]
    descriptor_extractor = cv2.ORB_create(n_keypoints)

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
    logging.info(f"keypts L:{len(keypoints_left)}, keypts R:{len(keypoints_right)}")
    if len(keypoints_left) == 0 or len(keypoints_right) == 0:
        return np.empty((1, 2)), np.empty((1, 2))

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(descriptors_left, descriptors_right)

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
