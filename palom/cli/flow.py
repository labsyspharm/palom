import numpy as np

# from . import register
import palom.register as register


def pc(img1, img2, sigma, mask=None):
    if mask is not None:
        if not mask:
            return np.array([0] * 3, dtype=np.float32).reshape(3, 1, 1)
    (r_shift, c_shift), error = register.phase_cross_correlation(
        img1, img2, sigma, 10
    )
    # NOTE invert shift direction from the registration func
    return np.array(
        [-r_shift, -c_shift, error], dtype=np.float32
    ).reshape(3, 1, 1)


def shifts_to_lab(shifts, max_radius=None, l_factor=50, ab_factor=100):
    if max_radius is not None:
        assert max_radius > 0
    # shifts is [np.inf, np.inf, np.inf] for blank block
    valid = np.all(np.isfinite(shifts), axis=0)
    # FIXME workaround to exclude masked blocks which are [0, 0, 0]
    valid &= ~np.all(shifts == 0, axis=0)
    
    distances = np.where(valid, np.linalg.norm(shifts[:2], axis=0), np.nan)
    lab = np.zeros((3, *shifts[0].shape))
    
    if max_radius is None:
        max_radius = np.nanmax(distances)
    if max_radius == 0:
        return lab
    
    # normalize the shift vectors
    distances = np.clip(distances, max_radius, None)
    lab[1:3, valid] = (shifts[:2, valid] / distances[valid])[::-1]
    lab[0] = np.linalg.norm(lab[1:3], axis=0)
    lab *= np.array([l_factor, ab_factor, ab_factor]).reshape(3, 1, 1)
    return lab


import dask.array as da
import dask.diagnostics
import palom


def optical_flow(
    p1, p2, ch1, ch2,
    block_size,
    mask=None,
    num_workers=4,
    chunk_size=4096
):
    # Use process parallelization
    chunk_size = int(np.ceil(chunk_size / block_size) * block_size)
    Reader = palom.reader.OmePyramidReader
    shape = Reader(p1).pyramid[0].shape[1:]
    ref = da.ones((1, *shape), chunks=chunk_size)

    mask_shape = da.ones(shape, chunks=block_size).numblocks
    if mask is None:
        mask = np.ones(mask_shape, dtype=bool)
    assert mask.shape == mask_shape
   
    def func(
        p1, p2, ch1, ch2, mask,
        block_info=None
    ):
        _, rloc, cloc = block_info[None]['array-location']
        chunk1 = Reader(p1).pyramid[0][ch1, slice(*rloc), slice(*cloc)]
        chunk2 = Reader(p2).pyramid[0][ch2, slice(*rloc), slice(*cloc)]
        chunk_mask = mask[
            slice(*np.ceil(np.divide(rloc, block_size)).astype(int)),
            slice(*np.ceil(np.divide(cloc, block_size)).astype(int))
        ]
        return block_optical_flow(chunk1, chunk2, block_size, mask=chunk_mask)
   
    shifts = da.map_blocks(
        func,
        p1=p1, p2=p2, ch1=ch1, ch2=ch2, mask=mask,
        dtype=float,
        chunks=ref.chunks,
    )

    with dask.diagnostics.ProgressBar():
        return shifts.compute(
            scheduler='processes', num_workers=num_workers
        )


import itertools
import tqdm.contrib.itertools
import skimage.util
import skimage.transform
def block_optical_flow(
    img1, img2, block_size,
    sigma=0, mask=None,
    show_progress=False
):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    block_shape = (block_size, block_size)
    block_shape = np.min(np.vstack([img1.shape, block_shape]), axis=0)
    wv_img1 = skimage.util.view_as_windows(img1, block_shape, block_shape)
    wv_img2 = skimage.util.view_as_windows(img2, block_shape, block_shape)
    h, w = wv_img1.shape[:2]
   
    wv_mask = np.ones((h, w, 1, 1), dtype=bool)
    if mask is not None:
        wv_mask = mask[..., np.newaxis, np.newaxis]
        wv_mask = wv_mask[..., :h, :w]
   
    out = np.zeros((3, h, w), dtype=np.float32)
    product_func = itertools.product
    if show_progress:
        product_func = tqdm.contrib.itertools.product
    idxs = product_func(range(h), range(w))
    for rr, cc in idxs:
        out[:, rr, cc] = pc(
            wv_img1[rr, cc], wv_img2[rr, cc], sigma, mask=wv_mask[rr, cc]
        ).ravel()
    return out


def reader_block_mask(reader, block_size, level=-1, channel=0):
    mask = palom.img_util.entropy_mask(
        reader.pyramid[level][channel]
    ).astype(float)
    target_shape = da.ones(
        reader.pyramid[0].shape[1:], chunks=block_size
    ).numblocks
    resized_mask = skimage.transform.resize(
        mask, target_shape, order=2
    )
    return resized_mask > 0.1


import skimage.exposure
def compose_thumbnail(p1, p2, ch1, ch2, log_intensity=True):
    Reader = palom.reader.OmePyramidReader
    r1 = Reader(p1)
    r2 = Reader(p2)
    func = np.log1p if log_intensity else np.array
    img1 = np.asarray(func(r1.pyramid[-1][ch1]))
    img2 = np.asarray(func(r2.pyramid[-1][ch2]))

    mask = palom.img_util.entropy_mask(img1)
    rimg1 = skimage.exposure.rescale_intensity(
        img1,
        in_range=(np.mean(img1[~mask]), np.percentile(img1[mask], 99.9)),
        out_range=np.float32
    )
    rimg2 = skimage.exposure.rescale_intensity(
        img2,
        in_range=(np.mean(img2[~mask]), np.percentile(img2[mask], 99.9)),
        out_range=np.float32
    )
    rimg2 = skimage.exposure.match_histograms(rimg2, rimg1)
    out = np.array([rimg1, rimg2, rimg1])
    return out


def plot_legend(
    shifts, max_radius, plot_scatter, plot_kde,
    pad_plot=False, plot_flow=False, ax=None
):

    import matplotlib.pylab as plt
    import scipy.ndimage

    # illuminant : {"A", "D50", "D55", "D65", "D75", "E"}
    if max_radius is None:
        max_radius = np.linalg.norm(shifts[:2], axis=0).max()
    lower, upper = -max_radius, max_radius
    if ax is None:
        _, ax = plt.subplots(1, 1)
    fig = ax.get_figure()
    legend_raster = np.mgrid[-1:1:501j, -1:1:501j]
    legend_raster[:, np.linalg.norm(legend_raster, axis=0) > 1] = 0

    ax.imshow(
        skimage.color.lab2rgb(
            np.dstack(shifts_to_lab(legend_raster, l_factor=50, ab_factor=100))
        ),
        # extent=(lower-.5, upper+.5, upper+.5, lower-.5)
        extent=(lower, upper, upper, lower)
    )
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    if pad_plot:
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
                np.dstack(shifts_to_lab(shifts, max_radius, l_factor=100, ab_factor=100)),
            ).reshape(-1, 3)[shifts[2].flatten() != 0],
            linewidths=0, s=4, alpha=.5
        )
    if plot_kde:
        # composite flow legend with kde
        density, ybins, xbins = np.histogram2d(
            y_shifts, x_shifts,
            bins=50, density=True,
        )
        density = scipy.ndimage.gaussian_filter(density, 1)
        ax.contour(
            density, levels=np.linspace(0, density.max(), 6)[1:],
            extent=(*xbins[[0, -1]], *ybins[[-1, 0]]),
            origin='upper',
            cmap='viridis',
            linewidths=0.5
        )
    if plot_flow:
        fig_flow, ax_flow = plt.subplots(1, 1)
        ax_flow.imshow(skimage.color.lab2rgb(np.dstack(lab)))
        return [fig, fig_flow]
    return [fig]


def get_img_extent(shape, downscale_factor):
    h, w = np.multiply(shape, downscale_factor)
    return (-0.5, w-0.5, h-0.5, -0.5)


import skimage.color
import matplotlib.pyplot as plt
import pathlib
from typing import List


def process_img_channel_pair(
    img_path: str | pathlib.Path,
    ref_channel: int,
    other_channels: List[int],
    block_size: int,
    max_radius=5,
    compute_mask:bool = True,
    num_workers: int = 4,
    as_script=True
):
    img_path = pathlib.Path(img_path)
    reader = palom.reader.OmePyramidReader(img_path)

    mask = None
    if compute_mask:
        mask = reader_block_mask(reader, block_size)
   
    num_cpus = dask.system.cpu_count()
    if num_workers > num_cpus:
        num_workers = num_cpus

    print()
    print(f"Processing {img_path.name} using {num_workers} cores")
    print(f"Block size: {block_size}; image shape {reader.pyramid[0].shape[1:]}")
    print("Reference channel:", ref_channel)
    all_shifts = []
    figures = []
    for channel in other_channels:
        print("    Processing channel:", channel)

        shifts = optical_flow(
            img_path, img_path, ref_channel, channel, block_size, mask,
            num_workers=num_workers
        )
        all_shifts.append(shifts)
        lab = shifts_to_lab(shifts, max_radius=max_radius)
        rgb = skimage.color.lab2rgb(lab, channel_axis=0)
        thumbnail = compose_thumbnail(
            img_path, img_path, ref_channel, channel, log_intensity=False
        )

        thumbnail_extent = get_img_extent(
            thumbnail[0].shape, reader.level_downsamples[len(reader.pyramid)-1]
        )
        flow_extent = get_img_extent(
            shifts[0].shape, block_size
        )

        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f"Optical flow: {img_path.name}")
        gs = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.5))

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(np.dstack(thumbnail), extent=thumbnail_extent)
        ax0.set_title(f"Channel {ref_channel} (pink) and {channel} (green)")

        ax1 = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
        ax1.imshow(thumbnail[1], extent=thumbnail_extent, alpha=0.5, cmap='gray')
        ax1.imshow(np.dstack(rgb), extent=flow_extent, alpha=0.5)
        ax1.label_outer()
        ax1.set_title(f"Overlay on channel {channel}")

        ax2 = fig.add_subplot(gs[2], sharex=ax0, sharey=ax0)
        ax2.imshow(np.dstack(rgb), extent=flow_extent)
        ax2.label_outer()
        ax2.set_title(f"Flow field")

        _ = plot_legend(
            shifts, max_radius, True, True, plot_flow=False, ax=fig.add_subplot(gs[3])
        )

        for ax in fig.get_axes():
            ax.set_anchor('N')
       
        plt.tight_layout()
        out_dir = img_path.parent / 'qc'
        out_dir.mkdir(exist_ok=True)
        fig.savefig(
            out_dir / f"flow-field-{img_path.stem}-channel-{ref_channel}-{channel}.png",
            bbox_inches='tight', dpi=144
        )
        figures.append(fig)
   
    if as_script: return
    return all_shifts, figures


if __name__ == '__main__':
    import fire
    fire.Fire(process_img_channel_pair)

    """
    NAME
        flow.py

    SYNOPSIS
        flow.py IMG_PATH REF_CHANNEL OTHER_CHANNELS BLOCK_SIZE <flags>

    POSITIONAL ARGUMENTS
        IMG_PATH
            Type: str | pathlib.Path
        REF_CHANNEL
            Type: int
        OTHER_CHANNELS
            Type: List
        BLOCK_SIZE
            Type: int

    FLAGS
        -m, --max_radius=MAX_RADIUS
            Default: 5
        -c, --compute_mask=COMPUTE_MASK
            Type: bool Default: True
        -n, --num_workers=NUM_WORKERS
            Type: int Default: 4
        -a, --as_script=AS_SCRIPT
            Default: True

    ---
   
    NOTE: OTHER_CHANNELS is a list of integers, use [1] to indicate the
    second channel and [1,2,3] (do not include space) for second, third and
    fourth channels.

    Ref https://google.github.io/python-fire/guide/#argument-parsing
   
    Examples

    python flow.py \
        "Z:\RareCyte-S3\P54_CRCstudy_Bridge\S32-Tonsil-P54_Strip_P76.ome.tif" \
        0 [1,2,3] \
        128

    python flow.py \
        "Z:\RareCyte-S3\P54_CRCstudy_Bridge\S32-Tonsil-P54_Strip_P76.ome.tif" \
        0 1,2,3 \
        128

    python flow.py \
        "Z:\RareCyte-S3\P54_CRCstudy_Bridge\S32-Tonsil-P54_Strip_P76.ome.tif" \
        0 [1] \
        128
    """