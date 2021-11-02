import numpy as np
import dask.array as da
import dask.diagnostics
import skimage.exposure
import sklearn.linear_model
from . import register
from . import block_affine


class ReaderAligner:
    # aligner only uses single-channel form to register readers

    def __init__(self, ref_reader, moving_reader, pyramid_level=0):
        self.ref_reader = ref_reader
        self.moving_reader = moving_reader
        self.pyramid_level = pyramid_level

    @property
    def ref_img(self):
        return self.ref_reader.get_processed_color(
            level=self.pyramid_level, mode='grayscale'
        )

    @property
    def grid_shape(self):
        return self.ref_img.numblocks

    @property
    def num_blocks(self):
        return self.ref_img.npartitions
    
    def coarse_register_affine(self, n_keypoints=2000):
        ref_img = self.ref_reader.get_processed_color(
            level=max(self.ref_reader.level_downsamples.keys())
        )
        moving_img = self.moving_reader.get_processed_color(
            level=max(self.moving_reader.level_downsamples.keys())
        )
        # `register.feature_based_registration` expects bright-field image to be
        # uint8 pixel dtype
        ref_img = skimage.exposure.rescale_intensity(
            ref_img, out_range=np.uint8
        )
        moving_img = skimage.exposure.rescale_intensity(
            moving_img, out_range=np.uint8
        )
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
        level_downsamples = self.moving_reader.level_downsamples
        coarse_factor = (
            level_downsamples[max(level_downsamples.keys())] /
            level_downsamples[self.pyramid_level]            
        )
        affine_matrix = self.coarse_affine_matrix.copy()
        affine_matrix[:2, 2] *= coarse_factor
        return affine_matrix

    @property
    def tform(self):
        return skimage.transform.AffineTransform(
            matrix=self.affine_matrix
        )

    def affine_transformed_moving_img(self, mode):
        ref_img = self.ref_img
        moving_img = self.moving_reader.get_processed_color(
            level=self.pyramid_level, mode=mode
        )
        return da.map_blocks(
            block_affine.block_affine_dask,
            self.affine_matrix,
            chunks=ref_img.chunks,
            dtype=ref_img.dtype,
            src_array=moving_img
        )

    def compute_shifts(self):
        ref_img = self.ref_img
        moving_img = self.affine_transformed_moving_img(mode='grayscale')
        shifts_da = da.map_blocks(
            lambda a, b: np.atleast_2d(
                register.phase_cross_correlation(a, b, sigma=0, upsample=1)[0]
            ),
            ref_img,
            moving_img,
            dtype=np.float32
        )
        with dask.diagnostics.ProgressBar():
            shifts = shifts_da.compute()
        self.shifts = shifts.reshape(-1, 2)

    def constrain_shifts(self):
        num_rows, num_cols = self.grid_shape
        self.distances = np.linalg.norm(self.shifts, axis=1)
        self.threshold_distance = skimage.filters.threshold_triangle(
            self.distances
        )

        high_confidence_blocks = self.distances < self.threshold_distance

        lr = sklearn.linear_model.LinearRegression()
        block_coords = np.mgrid[:num_rows, :num_cols].reshape(2, -1).T
        lr.fit(
            block_coords[high_confidence_blocks],
            self.shifts[high_confidence_blocks]
        )
        predicted_shifts = lr.predict(block_coords)
        diffs = self.shifts - predicted_shifts
        distance_diffs = np.linalg.norm(diffs, axis=1)
        passed = (
            distance_diffs <
            skimage.filters.threshold_triangle(distance_diffs)
        )
        self.shifts[~passed] = predicted_shifts[~passed]
    
    @property 
    def shifts_da(self):
        return da.from_array(
            self.shifts.reshape(self.grid_shape[0], -1),
            chunks=(1, 2)
        )
    
    @property
    def block_affine_matrices(self):
        mx = self.affine_matrix
        shifts = self.shifts
        
        def shift_affine_mx(mx, shift):
            y, x = shift
            mx_shift = np.eye(3)
            mx_shift[:2, 2] = x, y
            return mx @ mx_shift

        mxs = [
            shift_affine_mx(mx, s)
            for s in shifts
        ]
        return np.array(mxs)

    @property
    def block_affine_matrices_da(self):
        grid = np.arange(self.num_blocks).reshape(self.grid_shape)
        mxs = np.vstack([
            np.hstack(self.block_affine_matrices[r])
            for r in grid
        ])
        return da.from_array(mxs, chunks=3)
    
    def get_ref_mosaic(self, mode):
        level = self.pyramid_level
        ref_reader = self.ref_reader
        if mode == 'color':
            ref_img = da.moveaxis(
                ref_reader.pyramid_color[level], 2, 0
            )
        else:
            img_dtype = self.ref_reader.pixel_dtype
            _ref_img = ref_reader.get_processed_color(
                level=level, mode=mode
            )
            ref_img = da.map_blocks(
                skimage.exposure.rescale_intensity,
                _ref_img,
                chunks=_ref_img.chunks,
                dtype=img_dtype,
                in_range=(0, 1), 
                out_range=img_dtype.type
            )[np.newaxis, :]
        return ref_img

    def get_aligned_mosaic(self, mode):
        level = self.pyramid_level
        ref_reader = self.ref_reader
        moving_reader = self.moving_reader
        if mode == 'color':
            ref_img = da.moveaxis(
                ref_reader.pyramid_color[level], 2, 0
            )
            moving_img = da.moveaxis(
                moving_reader.pyramid_color[level], 2, 0
            )
            mxs = da.array([self.block_affine_matrices_da])
            multichannel = True
        else:
            ref_img = ref_reader.get_processed_color(
                level=level, mode=mode
            )
            moving_img = moving_reader.get_processed_color(
                level=level, mode=mode
            )
            mxs = self.block_affine_matrices_da
            multichannel = False
            img_dtype = self.ref_reader.pixel_dtype

        warped_img = da.map_blocks(
            block_affine.block_affine_dask,
            mxs,
            chunks=ref_img.chunks,
            dtype=ref_img.dtype,
            src_array=moving_img,
            multichannel=multichannel
        )

        if mode == 'color':
            return warped_img
        
        else:
            # output image in (C, Y, X) shape
            return da.map_blocks(
                skimage.exposure.rescale_intensity,
                warped_img,
                chunks=ref_img.chunks,
                dtype=img_dtype,
                in_range=(0, 1), 
                out_range=img_dtype.type
            )[np.newaxis, :]

