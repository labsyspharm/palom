import numpy as np
import dask.array as da
import skimage.exposure
import sklearn.linear_model
from loguru import logger
import tqdm.dask

from . import register
from . import block_affine
from . import img_util


def block_affine_transformed_moving_img(ref_img, moving_img, mxs, is_mask=False):
    assert img_util.is_single_channel(ref_img)
    return_slice = slice(None)
    if img_util.is_single_channel(moving_img) and moving_img.ndim == 2:
        # return 2d image if ndim of moving_img is 2
        moving_img = moving_img[np.newaxis]
        return_slice = 0
    return da.array([
        da.map_blocks(
            block_affine.block_affine_dask,
            mxs,
            src_array=c,
            chunks=ref_img.chunks,
            dtype=moving_img.dtype,
            is_mask=is_mask
        )
        for c in moving_img
    ])[return_slice]


def _pc(img1, img2, mask, **pcc_kwargs):
    if not np.all(mask):
        return (np.inf, np.inf), np.inf
    return register.phase_cross_correlation(img1, img2, **pcc_kwargs)


def block_shifts(ref_img, moving_img, mask=True, pcc_kwargs=None):
    default_pcc_kwargs = dict(sigma=0, upsample=1)
    if pcc_kwargs is None:
        pcc_kwargs = {}
    return da.map_blocks(
        lambda a, b, m: np.atleast_2d(
            _pc(a, b, m, **{**default_pcc_kwargs, **pcc_kwargs})[0]
        ),
        ref_img,
        moving_img,
        mask,
        dtype=np.float32
    )


def constrain_block_shifts(shifts, grid_shape):
    distances = np.linalg.norm(shifts, axis=1)
    is_finite = np.isfinite(distances)
    # exclude np.inf when computing threshold
    threshold_distance = skimage.filters.threshold_triangle(
        distances[is_finite]
    )

    high_confidence_blocks = distances < threshold_distance

    lr = sklearn.linear_model.LinearRegression()
    block_coords = np.indices(grid_shape).reshape(2, -1).T
    lr.fit(
        block_coords[high_confidence_blocks],
        shifts[high_confidence_blocks]
    )
    predicted_shifts = lr.predict(block_coords)
    diffs = shifts - predicted_shifts
    distance_diffs = np.linalg.norm(diffs, axis=1)
    passed = (
        distance_diffs <
        # exclude np.inf when computing threshold
        skimage.filters.threshold_triangle(distance_diffs[is_finite])
    )
    fitted_shifts = shifts.copy()
    fitted_shifts[~passed] = predicted_shifts[~passed]
    return fitted_shifts


def viz_shifts(shifts, grid_shape, dcenter=None, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors
    distances = np.linalg.norm(shifts, axis=1)
    is_finite = np.isfinite(distances)
    if dcenter is None:
        # exclude np.inf when computing threshold
        dcenter = skimage.filters.threshold_triangle(distances[is_finite])
    # exclude np.inf when computing threshold
    dmin, dmax = np.percentile(distances[is_finite], (0, 100))
    divnorm = matplotlib.colors.TwoSlopeNorm(dcenter, dmin, dmax)
    colorbar_ticks = np.concatenate(
        [np.linspace(dmin, dcenter, 5), np.linspace(dcenter, dmax, 5)[1:]]
    )
    if ax is None:
        _, ax = plt.subplots()
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'cm_shift', np.vstack([
            plt.cm.plasma(np.linspace(0, 1, 256)),
            plt.cm.gray(np.linspace(0.5, 1, 256))
        ])
    )
    im = ax.imshow(distances.reshape(grid_shape), norm=divnorm, cmap=custom_cmap)
    cax = ax.inset_axes([1.04, 0.0, 0.02, 1])
    colorbar = plt.colorbar(im, cax=cax)
    colorbar.set_ticks(colorbar_ticks)
   
    return ax


def block_affine_matrices(mx, shifts):
       
    def shift_affine_mx(mx, shift):
        y, x = shift
        mx_shift = np.eye(3)
        mx_shift[:2, 2] = x, y
        return mx_shift @ mx

    mxs = [
        shift_affine_mx(mx, s)
        for s in shifts
    ]
    return np.array(mxs)


def block_affine_matrices_da(mxs, grid_shape):
    num_rows, num_cols = grid_shape
    grid = np.arange(num_rows * num_cols).reshape(grid_shape)
    mxs = np.vstack([
        np.hstack(mxs[r])
        for r in grid
    ])
    return da.from_array(mxs, chunks=3)


class Aligner:

    def __init__(
        self,
        ref_img,
        moving_img,
        ref_thumbnail,
        moving_thumbnail,
        ref_thumbnail_down_factor,
        moving_thumbnail_down_factor
    ) -> None:
        self.ref_img=ref_img
        self.moving_img=moving_img
        self.ref_thumbnail=ref_thumbnail
        self.moving_thumbnail=moving_thumbnail
        self.ref_thumbnail_down_factor=ref_thumbnail_down_factor
        self.moving_thumbnail_down_factor=moving_thumbnail_down_factor

    def coarse_register_affine(self, **kwargs):
        default_kwargs = {
            'n_keypoints': 2000,
            'plot_match_result': True
        }
        default_kwargs.update(kwargs)
        ref_img = self.ref_thumbnail
        moving_img = self.moving_thumbnail
        affine_matrix = register.feature_based_registration(
            ref_img, moving_img,
            **default_kwargs
        )
        self.coarse_affine_matrix = np.vstack(
            [affine_matrix, [0, 0, 1]]
        )
   
    @property
    def affine_matrix(self):
        if not hasattr(self, 'coarse_affine_matrix'):
            self.coarse_register_affine()
        affine = skimage.transform.AffineTransform
        mx_ref = affine(scale=1/self.ref_thumbnail_down_factor).params
        mx_moving = affine(scale=1/self.moving_thumbnail_down_factor).params
        affine_matrix = (
            np.linalg.inv(mx_ref) @
            self.coarse_affine_matrix.copy() @
            mx_moving
        )
        return affine_matrix
   
    @property
    def tform(self):
        return skimage.transform.AffineTransform(
            matrix=self.affine_matrix
        )
   
    def affine_transformed_moving_img(self, mxs=None):
        if mxs is None:
            mxs = self.affine_matrix
        ref_img = self.ref_img
        moving_img = self.moving_img

        return block_affine_transformed_moving_img(
            ref_img, moving_img, mxs
        )
   
    def compute_shifts(self, mask=True, pcc_kwargs=None):
        logger.info(f"Computing block-wise shifts")
        ref_img = self.ref_img
        moving_img = self.affine_transformed_moving_img(self.affine_matrix)
        shifts_da = block_shifts(ref_img, moving_img, mask, pcc_kwargs=pcc_kwargs)
        with tqdm.dask.TqdmCallback(
            ascii=True, desc='Computing shifts',
        ):
            shifts = shifts_da.compute()
        self.shifts = shifts.reshape(-1, 2)

    @property
    def grid_shape(self):
        return self.ref_img.numblocks

    @property
    def num_blocks(self):
        return self.ref_img.npartitions

    def constrain_shifts(self):
        if not hasattr(self, 'original_shifts'):
            self.original_shifts = self.shifts.copy()
        self.shifts = constrain_block_shifts(
            self.original_shifts,
            self.grid_shape
        )
   
    @property
    def block_affine_matrices(self):
        mx = self.affine_matrix
        shifts = self.shifts
        return block_affine_matrices(mx, shifts)

    @property
    def block_affine_matrices_da(self):
        return block_affine_matrices_da(
            self.block_affine_matrices,
            self.grid_shape
        )

    def overlay_grid(self, ax=None):
        import matplotlib.pyplot as plt
        img = self.ref_thumbnail
        img = skimage.exposure.rescale_intensity(img, out_range=np.uint16)
        shape = self.grid_shape
        grid = np.arange(np.multiply(*shape)).reshape(shape)
        h, w = np.divide(
            img.shape,
            np.divide(self.ref_img.chunksize, self.ref_thumbnail_down_factor)
        )
        cmap = 'gray_r' if img_util.is_brightfield_img(img) else 'gray'
        func = np.array if img_util.is_brightfield_img(img) else np.log1p

        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(
            func(img),
            cmap=cmap,
            extent=(-0.5, w-0.5, h-0.5, -0.5)
        )
        # checkerboard pattern
        checkerboard = np.indices(shape).sum(axis=0) % 2
        if hasattr(self, 'shifts'):
            shifts = getattr(self, 'original_shifts', self.shifts)
            checkerboard = checkerboard.astype(float)
            checkerboard.flat[~np.all(np.isfinite(shifts), axis=1)] = np.nan
        ax.imshow(checkerboard, cmap='cool', alpha=0.2)
        return grid
   
    def plot_shifts(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        self.overlay_grid(axs[0])
        shifts = getattr(self, 'original_shifts', self.shifts)
        viz_shifts(shifts, self.grid_shape, ax=axs[1])
        return fig


def get_aligner(
    reader1, reader2,
    level1=0, level2=0,
    channel1=0, channel2=0,
    thumbnail_level1=-1, thumbnail_level2=-1,
    thumbnail_channel1=None, thumbnail_channel2=None
):
    if None in [thumbnail_level1, thumbnail_level2]:
        thumbnail_level1, thumbnail_level2 = match_thumbnail_level(
            [reader1, reader2]
        )
    if thumbnail_level1 <= -1: thumbnail_level1 += len(reader1.pyramid)
    if thumbnail_level2 <= -1: thumbnail_level2 += len(reader2.pyramid)
    thumbnail_channel1 = thumbnail_channel1 or channel1
    thumbnail_channel2 = thumbnail_channel2 or channel2
    return Aligner(
        reader1.read_level_channels(level1, channel1), 
        reader2.read_level_channels(level2, channel2),
        reader1.read_level_channels(thumbnail_level1, thumbnail_channel1),
        reader2.read_level_channels(thumbnail_level2, thumbnail_channel2),
        reader1.level_downsamples[thumbnail_level1] / reader1.level_downsamples[level1],
        reader2.level_downsamples[thumbnail_level2] / reader2.level_downsamples[level2]
    )


def match_thumbnail_level(readers):
    assert len(readers) > 1
    level_px_sizes = [
        {
            rr.pixel_size*vv: kk
            for kk, vv in rr.level_downsamples.items()
        }
        for rr in readers
    ]
    px_sizes = [sorted(ss.keys()) for ss in level_px_sizes]
    target_px_size = min([max(ss) for ss in px_sizes])
    target_levels = [
        lps[ps[np.argmin(np.abs(np.array(ps) - target_px_size))]]
        for ps, lps in zip(px_sizes, level_px_sizes)
    ]
    return target_levels