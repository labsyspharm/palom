import skimage.util
import skimage.filters
import skimage.exposure
import numpy as np
import warnings

def imagej_rgb2cmyk(rgb_img):
    shape = rgb_img.shape
    assert 3 in shape, (
        'Image of shape {} is not an RGB image'.format(shape)
    )
    channel_idx = shape.index(3)
    rgb_img = skimage.util.img_as_float(rgb_img)
    rgb = np.moveaxis(rgb_img, channel_idx, 0)

    cmy = 1-rgb
    k = cmy.min(axis=0)
   
    s = 1-k

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', 'invalid value encountered in true_divide', RuntimeWarning,
        )
        f_cmy = (cmy-k) / s
    f_cmy *= ~(k >= 1)

    return np.append(f_cmy, k.reshape(1, *k.shape), axis=0)

def cmyk2marker_int(cmyk_img, min, max):
    marker_int = cmyk_img[1] + cmyk_img[2]
    # `selem` kwarg renamed to `footprint` in skimage v0.19
    marker_int = skimage.filters.median(marker_int, np.ones((3,3)))

    marker_int = skimage.exposure.rescale_intensity(
        marker_int, in_range=(min, max), out_range=(0, 1)
    )
    return marker_int

def ohsu_cmyk2marker_int(cmyk_img):
    marker_int = cmyk_img[1] + cmyk_img[2]
    # the latest workflow applys median filter here while earlier
    # workflow seems to apply median filter after rescale_intensity
    marker_int = skimage.filters.median(marker_int, np.ones((3,3)))

    max_int = marker_int.max()
    marker_int = skimage.exposure.rescale_intensity(
        marker_int.astype(float),
        in_range=(0.05*max_int, 0.95*max_int),
        out_range=(0, 1)
    )
    return skimage.util.img_as_ubyte(marker_int)

def rgb2aec(rgb_img):
    cmyk_img = imagej_rgb2cmyk(rgb_img)
    aec = cmyk_img[1] + cmyk_img[2]
    return skimage.filters.median(aec, np.ones((3,3)))