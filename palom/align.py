import functools
import numpy as np
import dask.array as da
import skimage.exposure
import sklearn.linear_model
from loguru import logger
import tqdm.dask

from . import register
from . import block_affine
from . import img_util


def block_affine_transformed_moving_img(
    ref_img, moving_img, mxs, save_RAM=False
):
    assert img_util.is_single_channel(ref_img)
    _map_kwargs = dict(
        chunks=ref_img.chunks,
        dtype=moving_img.dtype
    )
    if img_util.is_single_channel(moving_img) and moving_img.ndim == 2:
        if save_RAM:
            # pass moving_img as dask array to save RAM usage in the cost of
            # longer runtime
            map_kwargs = {**_map_kwargs}
            map_func = functools.partial(
                block_affine.block_affine_dask,
                src_array=moving_img
            )
        else:
            # pass moving_img as a kwarg when calling da.map_blocks so that
            # moving_img will be persisted as numpy array when executing the
            # mapping function
            map_kwargs = {**_map_kwargs, **dict(src_array=moving_img)}
            map_func = block_affine.block_affine_dask
        return da.map_blocks(
            map_func,
            mxs,
            **map_kwargs
        )
    else:
        return da.array([
            block_affine_transformed_moving_img(
                ref_img, c, mxs, save_RAM=save_RAM
            )
            for c in moving_img
        ])


def block_shifts(ref_img, moving_img, pcc_kwargs=None):
    if pcc_kwargs is None:
        pcc_kwargs = dict(sigma=0, upsample=1)
    return da.map_blocks(
        lambda a, b: np.atleast_2d(
            register.phase_cross_correlation(a, b, **pcc_kwargs)[0]
        ),
        ref_img,
        moving_img,
        dtype=np.float32
    )


def constrain_block_shifts(shifts, grid_shape):
    num_rows, num_cols = grid_shape
    distances = np.linalg.norm(shifts, axis=1)
    threshold_distance = skimage.filters.threshold_triangle(
        distances
    )

    high_confidence_blocks = distances < threshold_distance

    lr = sklearn.linear_model.LinearRegression()
    block_coords = np.mgrid[:num_rows, :num_cols].reshape(2, -1).T
    lr.fit(
        block_coords[high_confidence_blocks],
        shifts[high_confidence_blocks]
    )
    predicted_shifts = lr.predict(block_coords)
    diffs = shifts - predicted_shifts
    distance_diffs = np.linalg.norm(diffs, axis=1)
    passed = (
        distance_diffs <
        skimage.filters.threshold_triangle(distance_diffs)
    )
    fitted_shifts = shifts.copy()
    fitted_shifts[~passed] = predicted_shifts[~passed]
    return fitted_shifts


def viz_shifts(shifts, grid_shape, vmax=None):
    import matplotlib.pyplot as plt
    distances = np.linalg.norm(shifts, axis=1)
    if vmax is None:
        vmax = skimage.filters.threshold_triangle(distances)
    plt.figure()
    plt.imshow(distances.reshape(grid_shape), vmax=vmax)


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

    def coarse_register_affine(self, n_keypoints=2000):
        ref_img = self.ref_thumbnail
        moving_img = self.moving_thumbnail
        affine_matrix = register.feature_based_registration(
            ref_img, moving_img,
            n_keypoints=n_keypoints,
            plot_match_result=True
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
    
    def affine_transformed_moving_img(self, mxs=None, save_RAM=False):
        if mxs is None:
            mxs = self.affine_matrix
        ref_img = self.ref_img
        moving_img = self.moving_img

        return block_affine_transformed_moving_img(
            ref_img, moving_img, mxs, save_RAM=save_RAM
        )
    
    def shifts_da(self, save_RAM=False, pcc_kwargs=None):
        ref_img = self.ref_img
        moving_img = self.affine_transformed_moving_img(
            self.affine_matrix, save_RAM=save_RAM
        )
        return block_shifts(ref_img, moving_img, pcc_kwargs=pcc_kwargs)

    def compute_shifts(self, pcc_kwargs=None):
        logger.info(f"Computing block-wise shifts")
        shifts_da = self.shifts_da(pcc_kwargs=pcc_kwargs)
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

    def debug(self, idx, pcc_kwargs=None):
        if type(idx) == int:
            idx = np.unravel_index(idx, self.grid_shape)
        row_idx, col_idx = idx
        shift = (
            self.shifts_da(pcc_kwargs=pcc_kwargs, save_RAM=True)
                .blocks[row_idx, col_idx]
                .compute()
        )
        affine_mx = block_affine_matrices(self.affine_matrix, shift)[0]
        
        img1 = self.ref_img
        img2 = self.affine_transformed_moving_img(save_RAM=True)
        img3 = self.affine_transformed_moving_img(mxs=affine_mx, save_RAM=True)

        return [
            img.blocks[row_idx, col_idx]
            for img in (img1, img2, img3)
        ]

    def overlay_grid(self):
        import matplotlib.pyplot as plt
        img = self.ref_thumbnail
        shape = self.grid_shape
        grid = np.arange(np.multiply(*shape)).reshape(shape)
        h, w = np.divide(
            img.shape,
            np.divide(self.ref_img.chunksize, self.ref_thumbnail_down_factor)
        )
        cmap = 'Greys' if img_util.is_brightfield_img(img) else 'Greys_r'
        plt.figure()
        plt.imshow(
            np.sqrt(np.abs(img)),
            cmap=cmap,
            extent=(-0.5, w-0.5, h-0.5, -0.5)
        )
        # checkerboard pattern
        plt.imshow(np.indices(shape).sum(axis=0) % 2, cmap='cool', alpha=0.2)
        return grid
