import cv2
import numpy as np
import scipy.fft
from . import img_util

import skimage.feature
import skimage.registration
import itertools
import warnings
import logging

if hasattr(skimage.registration, 'phase_cross_correlation'):
    register_translation = skimage.registration.phase_cross_correlation
else:
    register_translation = skimage.feature.register_translation


# 
# Image-based registration
# 
def phase_cross_correlation(img1, img2, sigma, upsample=10, module='skimage'):
    assert module in ['cv2', 'skimage']
    
    img1w = img_util.whiten(img1, sigma)
    img2w = img_util.whiten(img2, sigma)
    
    if module == 'skimage':
        img1_f = scipy.fft.fft2(img1w)
        img2_f = scipy.fft.fft2(img2w)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', 'invalid value encountered in true_divide',
                RuntimeWarning,
            )
            shift, _error, _phasediff = register_translation(
                img1_f, img2_f, upsample_factor=upsample, space='fourier'
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
import skimage.filters
import matplotlib.pyplot as plt
import skimage.exposure


def feature_based_registration(
    img_left, img_right,
    n_keypoints=1000, plot_match_result=False,
    plot_individual_result=False, ransacReprojThreshold=5
):
    compare_funcs = [
        np.less if img_util.is_brightfield_img(i) else np.greater
        for i in (img_left, img_right)
    ]
    imgs_otsu = [
        f(i, skimage.filters.threshold_otsu(i)).astype(np.uint8)
        for (i, f) in zip((img_left, img_right), compare_funcs)
    ]
    imgs_tri = [
        f(i, skimage.filters.threshold_triangle(i)).astype(np.uint8)
        for (i, f) in zip((img_left, img_right), compare_funcs)
    ]
    img_left, img_right = match_bf_fl_histogram(img_left, img_right)
    imgs_whiten = [
        img_util.whiten(i, 1)
        for i in (img_left, img_right)
    ]

    all_found = [
        cv2_feature_detect_and_match(
            *img_pair, n_keypoints=n_keypoints,
            plot_match_result=plot_individual_result
        )
        for img_pair in [
            imgs_otsu,
            imgs_tri,
            (img_left, img_right),
            imgs_whiten
        ]
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
        plt.figure()
        plt.gray()
        skimage.feature.plot_matches(
            plt.gca(), img_left, img_right,
            np.fliplr(all_src), np.fliplr(all_dst),
            np.arange(len(all_src)).repeat(2).reshape(-1, 2)[mask.flatten()>0],
            only_matches=False
        )
    return t_matrix


def match_bf_fl_histogram(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    bf1, bf2 = [
        img_util.is_brightfield_img(i) 
        for i in (img1, img2)
    ]
    if bf1 == bf2:
        return img1, skimage.exposure.match_histograms(img2, img1)
    elif bf1 is True:
        return img1, skimage.exposure.match_histograms(-img2, img1)
    elif bf2 is True:
        return skimage.exposure.match_histograms(-img1, img2), img2


def plot_img_keypoints(imgs, keypoints):
    fig, axs = plt.subplots(1, len(imgs))
    for i, k, a in zip(imgs, keypoints, axs):
        a.imshow(cv2.drawKeypoints(
            i, k, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        ))
        a.set_title(len(k))
    return 


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
        plot_img_keypoints(
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
        method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=5000
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






