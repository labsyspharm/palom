import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
import skimage.filters
import skimage.transform

from . import img_util


IS_BF_IMG = img_util.is_brightfield_img


def make_img_pairs(img1, img2, auto_invert_intensity=True, auto_mask=False):
    img1 = np.asarray(img1).astype(np.float32)
    img2 = np.asarray(img2).astype(np.float32)
    compare_funcs = [
        np.less if IS_BF_IMG(i) else np.greater
        for i in (img1, img2)
    ]
    if not auto_invert_intensity:
        compare_funcs = [compare_funcs[0]]*2
    imgs_otsu = [
        f(i, skimage.filters.threshold_otsu(i)).astype(np.uint8)
        for (i, f) in zip((img1, img2), compare_funcs)
    ]
    imgs_tri = [
        f(i, skimage.filters.threshold_triangle(i)).astype(np.uint8)
        for (i, f) in zip((img1, img2), compare_funcs)
    ]
    if auto_invert_intensity:
        img1, img2 = match_bf_fl_histogram(img1, img2, auto_mask)
    else:
        match_func = skimage.exposure.match_histograms
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
    match_func = skimage.exposure.match_histograms
    if auto_mask:
        match_func = masked_match_histograms
    if is_bf_img1 == is_bf_img2:
        return img1, match_func(img2, img1)
    elif is_bf_img1:
        return img1, match_func(-img2, img1)
    elif is_bf_img2:
        return match_func(-img1, img2), img2


def masked_match_histograms(img, ref_img):
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
    matched_img[mask] = skimage.exposure.histogram_matching._match_cumulative_cdf(
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
