import numpy as np
import cv2
import skimage.transform
import skimage.filters

def cv2_pyramid(img, max_size=1024):
    size_max = np.array(img.shape).max()
    pyramid = [img]
    while size_max > max_size:
        img = cv2.pyrDown(img)
        pyramid.append(img)
        size_max //= 2
    return pyramid


def whiten(img, sigma=1):
    border_mode = cv2.BORDER_REFLECT
    g_img = img if sigma == 0 else cv2.GaussianBlur(
        img, (0, 0), sigma, 
        borderType=border_mode
    ).astype(np.float32)
    log_img = cv2.Laplacian(
        g_img, cv2.CV_32F, ksize=1,
        borderType=border_mode
    )
    # log_img = cv2.convertScaleAbs(log_img)
    return log_img


def cv2_to_uint8(img):
    return cv2.normalize(
        src=img, dst=None,
        alpha=255, beta=0,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )


def is_single_channel(img):
    is_2d = img.ndim == 2
    is_flat_3d = img.ndim == 3 and img.shape[0] == 1
    return is_2d or is_flat_3d


def entropy_mask(img, kernel_size=9):
    img = skimage.exposure.rescale_intensity(
        img, out_range=np.uint8
    ).astype(np.uint8)
    entropy = skimage.filters.rank.entropy(img, np.ones((kernel_size, kernel_size)))
    return entropy > skimage.filters.threshold_otsu(entropy)


def is_brightfield_img(img, max_size=100):
    downscale_factor = int(max(img.shape) / max_size)
    thumbnail = skimage.transform.downscale_local_mean(img, (downscale_factor, downscale_factor))
    mask = entropy_mask(thumbnail)
    # is using mean better?
    return np.median(thumbnail[mask]) < np.median(thumbnail[~mask])