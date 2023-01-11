import numpy as np
import dask.array as da
import cv2
import skimage.transform
import skimage.filters
import skimage.morphology

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
    img = np.array(img)
    downscale_factor = int(max(img.shape) / max_size)
    thumbnail = skimage.transform.downscale_local_mean(img, (downscale_factor, downscale_factor))
    mask = entropy_mask(thumbnail)
    # is using mean better?
    return np.median(thumbnail[mask]) < np.median(thumbnail[~mask])


def block_labeled_mask(img_shape, block_shape, out_chunks=None):
    assert len(img_shape) == 2
    if len(block_shape) == 1:
        block_shape = (block_shape[0], block_shape[0])
    da_template = da.zeros(img_shape, chunks=block_shape)
    unit_mask = np.indices(da_template.numblocks).sum(axis=0) % 2
    unit_mask = skimage.morphology.label(
        unit_mask, connectivity=1, background=-1
    ).astype(np.int32)
    full_mask = np.repeat(
        np.repeat(unit_mask, block_shape[0], axis=0),
        block_shape[1],
        axis=1
    )[:img_shape[0], :img_shape[1]]
    if out_chunks is None:
        out_chunks = block_shape
    return da.from_array(full_mask, chunks=out_chunks)


def to_napari_affine(mx): 
    ul = np.flip(mx[:2, :2], (0, 1))
    rows = np.hstack([ul, np.flipud(mx[:2, 2:3])])
    return np.vstack((rows, [0, 0, 1]))