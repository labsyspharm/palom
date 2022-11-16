import palom
import numpy as np

# from . import register
import palom.register as register


def pc(img1, img2, sigma, mask=None):
    if mask is not None:
        mask = mask.astype(bool)
        if mask.mean() < 0.1:
            return np.array([0] * 3, dtype=np.float32).reshape(3, 1, 1)
    shifts, error = register.phase_cross_correlation(img1, img2, sigma)
    return np.array([*shifts, error], dtype=np.float32).reshape(3, 1, 1)


def shifts_to_lab(shifts, max_radius=None, l_factor=50, ab_factor=100):
    lab = np.zeros((3, *shifts[0].shape))
    distances = np.linalg.norm(shifts[:2], axis=0)
    if max_radius is None:
        max_radius = distances.max()
    lab[1:3] = np.where(
        distances < max_radius,
        shifts[:2] / max_radius,
        shifts[:2] / distances
    )[::-1]
    lab[0] = np.linalg.norm(lab[1:3], axis=0)
    # expected lab has range 0-100
    lab *= np.array([l_factor, ab_factor, ab_factor]).reshape(3, 1, 1)
    return lab




import skimage.color
import cv2
import skimage.transform
import skimage.filters


def entropy_mask(img, kernel_size=9):
    img = skimage.exposure.rescale_intensity(
        img, out_range=np.uint8
    ).astype(np.uint8)
    entropy = skimage.filters.rank.entropy(img, np.ones((kernel_size, kernel_size)))
    return entropy > skimage.filters.threshold_otsu(entropy)


def tissue_mask(img, downscale_factor=50):
    thumbnail = skimage.transform.downscale_local_mean(img, (downscale_factor, downscale_factor))
    mask = entropy_mask(thumbnail)
    return cv2.resize(mask.astype(np.uint8), dsize=img.shape[::-1]) > 0


def is_brightfield_img(img, max_size=100):
    downscale_factor = int(max(img.shape) / max_size)
    thumbnail = skimage.transform.downscale_local_mean(img, (downscale_factor, downscale_factor))
    mask = entropy_mask(thumbnail)
    return np.mean(thumbnail[mask]) < np.mean(thumbnail[~mask])


def plot_legend(shifts, max_radius, plot_scatter, plot_kde, pad_plot=False, plot_flow=False):

    import matplotlib.pylab as plt
    import scipy.ndimage
    import functools

    array2rgb = functools.partial(np.moveaxis, source=0, destination=2)

    # illuminant : {"A", "D50", "D55", "D65", "D75", "E"}
    lower, upper = -max_radius, max_radius
    fig, ax = plt.subplots(1, 1)
    legend_raster = np.mgrid[-1:1:500j, -1:1:500j]
    legend_raster[:, np.linalg.norm(legend_raster, axis=0) > 1] = 0

    ax.imshow(
        skimage.color.lab2rgb(
            array2rgb(shifts_to_lab(legend_raster, l_factor=50, ab_factor=100))
        ),
        # extent=(lower-.5, upper+.5, upper+.5, lower-.5)
        extent=(lower, upper, upper, lower)
    )
    if pad_plot:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(xlim[0]-.5, xlim[1]+.5)
        ax.set_ylim(ylim[0]+.5, ylim[1]-.5)
    if not (plot_scatter or plot_kde or plot_flow):
        return [fig]
    lab = shifts_to_lab(shifts, max_radius, l_factor=50, ab_factor=100)
    x_shifts, y_shifts = [
        # 100 is from `ab_factor`
        s[shifts[2] != 0].flatten() / 100 * max_radius
        for s in lab[1:3]
    ]
    if plot_scatter:
        ax.scatter(
            x_shifts, y_shifts,
            c=skimage.color.lab2rgb(
                array2rgb(shifts_to_lab(shifts, max_radius, l_factor=100, ab_factor=100)),
            ).reshape(-1, 3)[shifts[2].flatten() != 0],
            linewidths=0, s=4, alpha=.5
        )
    if plot_kde:
        # composite flow legend with kde
        density, _, _ = np.histogram2d(
            y_shifts, x_shifts,
            bins=50, density=True
        )
        density = scipy.ndimage.gaussian_filter(density, 1)
        ax.contour(
            density, levels=np.linspace(0, density.max(), 6)[1:],
            # note that y axis is inverted
            extent=(lower, upper, lower, upper),
            cmap='viridis'
        )
    if plot_flow:
        fig_flow, ax_flow = plt.subplots(1, 1)
        ax_flow.imshow(skimage.color.lab2rgb(array2rgb(lab)))
        return [fig, fig_flow]
    return [fig]




import tifffile
import dask.array as da
import dask.diagnostics


img1 = tifffile.imread('HEM.ome.tif', key=0)
img2 = tifffile.imread('coarse_aligned-PANCK_HEM.ome.tif', key=0)
img3 = tifffile.imread('palom-PANCK_HEM.ome.tif', key=0)


BLOCK_SIZE = 200

img1da = da.from_array(img1, chunks=(BLOCK_SIZE, BLOCK_SIZE))
img2da = da.from_array(img2, chunks=(BLOCK_SIZE, BLOCK_SIZE))

mask = img1da.map_blocks(skimage.transform.downscale_local_mean, (50, 50))
mask = entropy_mask(mask)
maskda = da.from_array(mask, chunks=(int(BLOCK_SIZE/50),)*2)


with dask.diagnostics.ProgressBar():
    s21 = da.map_blocks(pc, img1da, img2da, 0, maskda, new_axis=0).compute()
# invert shift direction to show the vector from img1 to img2
s21[:2] *= -1

with dask.diagnostics.ProgressBar():
    img3da = da.from_array(img3, chunks=(BLOCK_SIZE, BLOCK_SIZE))
    s31 = da.map_blocks(pc, img1da, img3da, 0, maskda, new_axis=0).compute()
# invert shift direction to show the vector from img1 to img2
s31[:2] *= -1


import napari

max_radius = 5
timg1 = skimage.transform.downscale_local_mean(img1, (10, 10))
flow_img = skimage.transform.resize(
    skimage.color.lab2rgb(
        np.moveaxis(shifts_to_lab(s21, max_radius), 0, 2)
    ),
    (*timg1.shape, 3),
    order=0
)
v = napari.Viewer()
v.add_image((0.5-is_brightfield_img(timg1))*timg1*2, blending='additive')
v.add_image(flow_img, blending='additive')
timg2 = skimage.transform.downscale_local_mean(img2, (10, 10))
v.add_image((0.5-is_brightfield_img(timg2))*timg2*2, blending='additive')

