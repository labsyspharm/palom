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
    if np.issubdtype(img.dtype, bool):
        img = img.astype(np.int8)
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
        # more bins is useful for IF images
        img, out_range=(0, 1000)
    ).astype(np.uint16)
    entropy = skimage.filters.rank.entropy(img, np.ones((kernel_size, kernel_size)))
    # threshold_otsu is needed for bright-field images
    return entropy > skimage.filters.threshold_otsu(entropy)


def is_brightfield_img(img, max_size=100):
    img = np.array(img)
    downscale_factor = int(max(img.shape) / max_size)
    if downscale_factor > 1:
        img = cv2_downscale_local_mean(img, downscale_factor)
    mask = entropy_mask(img)
    # is using mean better?
    return np.median(img[mask]) < np.median(img[~mask])


def repeat_2d(arr, repeats):
    assert arr.ndim == 2
    assert len(repeats) == 2
    r0, r1 = repeats
    return np.repeat(
        np.repeat(arr, r0, axis=0), r1, axis=1
    )


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


# orders of magnitute faster than skimage.transform.downscale_local_mean
# also the gives sensible values of pixels on the edge
def cv2_downscale_local_mean(img, factor):
    assert img.ndim in [2, 3]
    img = np.asarray(img)
    axis_moved = False
    channel_ax = np.argmin(img.shape)
    if (img.ndim == 3) & (channel_ax != 2):
        img = np.moveaxis(img, channel_ax, 2)
        axis_moved = True
    simg = cv2.blur(
        img, ksize=(factor, factor), anchor=(0, 0)
    )[::factor, ::factor]
    if axis_moved:
        simg = np.moveaxis(simg, 2, channel_ax)
    return simg