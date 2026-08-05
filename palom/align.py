import cv2
import numpy as np
import dask.array as da
import skimage.exposure
import skimage.transform
import sklearn.linear_model
from loguru import logger
import tqdm.dask

from . import register
from . import register_coarse
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


def block_displacement_transformed_moving_img(
    ref_img, moving_img, affine_matrix, shifts, grid_shape,
    sigma_blocks=0.0, field_order=1, is_mask=False, cval=0.0,
    interpolation="skimage",
):
    """Seam-free alternative to `block_affine_transformed_moving_img`.

    Instead of giving each block its own affine (a piecewise-constant
    translation that is discontinuous at block edges -> visible cracks), the
    per-block `shifts` are treated as a coarse displacement field sampled at
    block centers. A single global `affine_matrix` (moving -> ref) plus a
    continuous interpolation of that field warps the moving image with one
    resample per output chunk, so neighbouring chunks sample contiguous source
    coordinates and no seams appear.

    `sigma_blocks` controls how gradually the displacement blends between
    blocks: it is a Gaussian smoothing of the (tiny) block-shift grid, in
    block units (0 = pure interpolation; ~0.3-1.5 is a useful range -- larger
    values dissolve seams further but erase genuine local deformation).
    `field_order` is 1 (bilinear, C0 -- may show faint creases along block
    lines) or 3 (bicubic, smoother but can ring near sharp shift changes) for
    the coarse-grid upsampling. `interpolation` selects the final image
    resample backend: "skimage" (float-precision, more accurate) or "cv2"
    (fixed-point, ~1/32-px sub-pixel quantization, faster).

    RAM: the full-resolution displacement field is never materialized. Each
    output chunk computes its own displacement from the small `grid_shape`
    grid and reads only the source crop it needs, mirroring the memory
    profile of the existing per-block path.
    """
    assert img_util.is_single_channel(ref_img)
    out_shape = tuple(int(s) for s in ref_img.shape[-2:])
    cy, cx = ref_img.chunksize[-2:]

    return_slice = slice(None)
    if img_util.is_single_channel(moving_img) and moving_img.ndim == 2:
        moving_img = moving_img[np.newaxis]
        return_slice = 0

    grid = np.asarray(shifts, dtype="float32").reshape(*grid_shape, 2)
    # `block_affine_matrices` applies the shift in reference space, i.e.
    # source = inv(A) @ (ref - shift); we add the field as inv(A) @ (ref + d),
    # so the displacement is the negative of the per-block shift.
    grid = -grid
    if sigma_blocks and sigma_blocks > 0:
        import scipy.ndimage as ndi
        # smoothing happens on the small grid (cheap, RAM-free); no blur is
        # applied across the channel/component axis
        grid = ndi.gaussian_filter(
            grid, sigma=(sigma_blocks, sigma_blocks, 0), mode="nearest"
        )

    inv_affine = np.linalg.inv(np.asarray(affine_matrix, dtype="float64"))
    order = 0 if is_mask else 1
    field_interp = cv2.INTER_LINEAR if field_order == 1 else cv2.INTER_CUBIC

    template = da.zeros(out_shape, dtype="uint8", chunks=(cy, cx))
    warped = [
        template.map_blocks(
            _displacement_remap_block,
            src_array=c,
            shift_grid=grid,
            out_shape=out_shape,
            inv_affine=inv_affine,
            cval=float(cval),
            module=interpolation,
            order=order,
            field_interp=field_interp,
            dtype=moving_img.dtype,
        )
        for c in moving_img
    ]
    return da.array(warped)[return_slice]


def _sample_displacement(rows, cols, shift_grid, out_shape, field_interp):
    """Per-pixel (dy, dx) for `rows`/`cols`, interpolated from a small
    block-center shift grid (cell-center aligned)."""
    nr, nc = shift_grid.shape[:2]
    H, W = out_shape
    grow = ((rows + 0.5) * (nr / H) - 0.5).astype("float32")
    gcol = ((cols + 0.5) * (nc / W) - 0.5).astype("float32")
    samp_x, samp_y = np.meshgrid(gcol, grow)
    samp_x = np.ascontiguousarray(samp_x, dtype="float32")
    samp_y = np.ascontiguousarray(samp_y, dtype="float32")
    grid = np.asarray(shift_grid, dtype="float32")
    dy = cv2.remap(
        np.ascontiguousarray(grid[..., 0]), samp_x, samp_y, field_interp,
        borderMode=cv2.BORDER_REPLICATE,
    )
    dx = cv2.remap(
        np.ascontiguousarray(grid[..., 1]), samp_x, samp_y, field_interp,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return dy, dx


def _ref_to_moving_coords(rows, cols, dy, dx, inv_affine):
    """Reference pixel grid + displacement, mapped to moving coordinates by
    the inverse affine. Returns (map_y, map_x)."""
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    map_y = (yy + dy).astype("float32")
    map_x = (xx + dx).astype("float32")
    if inv_affine is not None and not np.allclose(inv_affine, np.eye(3)):
        h, w = map_y.shape
        # points are (x, y); cv2.transform applies the 2x3 affine
        pts = np.stack([map_x.ravel(), map_y.ravel()], axis=1)[:, np.newaxis, :]
        pts = cv2.transform(pts.astype("float32"), inv_affine[:2]).reshape(h, w, 2)
        map_x = np.ascontiguousarray(pts[..., 0])
        map_y = np.ascontiguousarray(pts[..., 1])
    return map_y, map_x


def _remap_crop(src_array, map_y, map_x, cval, module="skimage", order=1):
    """Resample only the source crop the mapping lands in (+1 px interpolation
    margin), so memory stays bounded per chunk.

    `module="skimage"` uses `skimage.transform.warp` (float-precision sampling,
    more accurate); `module="cv2"` uses `cv2.remap` (fixed-point, ~1/32-px
    sub-pixel quantization, but faster). `order` is the interpolation order
    (0 nearest, 1 linear/bilinear, 3 cubic)."""
    h, w = map_y.shape
    src_h, src_w = src_array.shape[-2:]
    rmin = int(np.floor(map_y.min())) - 1
    cmin = int(np.floor(map_x.min())) - 1
    rmax = int(np.ceil(map_y.max())) + 1
    cmax = int(np.ceil(map_x.max())) + 1
    if rmax <= 0 or cmax <= 0 or rmin >= src_h or cmin >= src_w:
        return np.full((h, w), cval, dtype=src_array.dtype)
    rmin, cmin = max(rmin, 0), max(cmin, 0)
    rmax, cmax = min(rmax, src_h), min(cmax, src_w)
    crop = np.asarray(src_array[rmin:rmax, cmin:cmax])
    if 0 in crop.shape:
        return np.full((h, w), cval, dtype=src_array.dtype)
    map_x = map_x - cmin
    map_y = map_y - rmin
    if module == "skimage":
        warped = skimage.transform.warp(
            crop, np.stack([map_y, map_x]), order=order,
            mode="constant", cval=cval, preserve_range=True,
        )
        if np.issubdtype(src_array.dtype, np.integer):
            warped = np.round(warped)
        return warped.astype(src_array.dtype)
    # cv2.remap requires both coordinate maps to be contiguous CV_32FC1
    cv2_interp = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 3: cv2.INTER_CUBIC}[order]
    map_x = np.ascontiguousarray(map_x, dtype="float32")
    map_y = np.ascontiguousarray(map_y, dtype="float32")
    return cv2.remap(crop, map_x, map_y, cv2_interp, borderValue=cval)


def _displacement_remap_block(
    _template, src_array=None, shift_grid=None, out_shape=None,
    inv_affine=None, cval=0.0, module="skimage", order=1,
    field_interp=cv2.INTER_LINEAR, block_info=None,
):
    (r0, r1), (c0, c1) = block_info[None]["array-location"]
    rows = np.arange(r0, r1, dtype="float32")
    cols = np.arange(c0, c1, dtype="float32")
    dy, dx = _sample_displacement(rows, cols, shift_grid, out_shape, field_interp)
    map_y, map_x = _ref_to_moving_coords(rows, cols, dy, dx, inv_affine)
    return _remap_crop(src_array, map_y, map_x, cval, module=module, order=order)


def _multiobj_displacement_block(
    _template, src_array=None, label_to_obj=None, base_inv_affine=None,
    label_mask=None, mask_scale=1.0, out_shape=None, cval=0.0,
    module="skimage", order=1, field_interp=cv2.INTER_LINEAR, block_info=None,
):
    """One output chunk of a multi-object displacement warp.

    Each pixel is owned by exactly one object according to the (reference
    space) labeled `label_mask`; that object's own affine + smoothed
    displacement field warps it. Pixels whose label has no object (background
    or excluded objects) fall back to the baseline affine. Only the objects
    whose labels appear in this chunk are warped, so interior chunks cost a
    single remap.
    """
    (r0, r1), (c0, c1) = block_info[None]["array-location"]
    h, w = r1 - r0, c1 - c0
    rows = np.arange(r0, r1, dtype="float32")
    cols = np.arange(c0, c1, dtype="float32")

    # per-pixel ownership: nearest sample of the labeled mask, which lives at
    # `mask_scale`x coarser (reference-thumbnail) resolution
    mr = np.clip((np.arange(r0, r1) / mask_scale).astype(int), 0, label_mask.shape[0] - 1)
    mc = np.clip((np.arange(c0, c1) / mask_scale).astype(int), 0, label_mask.shape[1] - 1)
    chunk_labels = label_mask[np.ix_(mr, mc)]

    result = np.full((h, w), cval, dtype=src_array.dtype)
    zeros = np.zeros((h, w), dtype="float32")
    for lbl in np.unique(chunk_labels):
        sel = chunk_labels == lbl
        obj = None if label_to_obj is None else label_to_obj.get(int(lbl))
        if obj is None:
            inv_affine, dy, dx = base_inv_affine, zeros, zeros
        else:
            inv_affine, grid = obj
            dy, dx = _sample_displacement(rows, cols, grid, out_shape, field_interp)
        map_y, map_x = _ref_to_moving_coords(rows, cols, dy, dx, inv_affine)
        warped = _remap_crop(src_array, map_y, map_x, cval, module=module, order=order)
        result[sel] = warped[sel]
    return result


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
    finite_distances = distances[is_finite]
    # Not enough finite shifts, or no spread among them (e.g. a masked region
    # with very few valid blocks, or blocks that all returned the same shift):
    # there is no trend to fit and `threshold_triangle` would choke on a
    # degenerate histogram, so leave the shifts as-is.
    if finite_distances.size < 3 or np.ptp(finite_distances) == 0:
        return shifts
    # exclude np.inf when computing threshold
    threshold_distance = skimage.filters.threshold_triangle(finite_distances)

    high_confidence_blocks = distances < threshold_distance
    if high_confidence_blocks.sum() < 3:
        return shifts

    lr = sklearn.linear_model.LinearRegression()
    block_coords = np.indices(grid_shape).reshape(2, -1).T
    lr.fit(
        block_coords[high_confidence_blocks],
        shifts[high_confidence_blocks]
    )
    predicted_shifts = lr.predict(block_coords)
    diffs = shifts - predicted_shifts
    distance_diffs = np.linalg.norm(diffs, axis=1)
    finite_diffs = distance_diffs[is_finite]
    if np.ptp(finite_diffs) == 0:
        return shifts
    passed = (
        distance_diffs <
        # exclude np.inf when computing threshold
        skimage.filters.threshold_triangle(finite_diffs)
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
        moving_thumbnail_down_factor,
        # physical size (um) of one thumbnail pixel; `coarse_register_affine`
        # uses the pair to tell whether the two scans image comparable tissue
        # areas. `None` (either one) means "unknown" -- the coarse route is then
        # picked from match confidence alone, never from a placeholder scale.
        ref_thumbnail_pixel_size=None,
        moving_thumbnail_pixel_size=None,
    ) -> None:
        self.ref_img=ref_img
        self.moving_img=moving_img
        self.ref_thumbnail=ref_thumbnail
        self.moving_thumbnail=moving_thumbnail
        self.ref_thumbnail_down_factor=ref_thumbnail_down_factor
        self.moving_thumbnail_down_factor=moving_thumbnail_down_factor
        self.ref_thumbnail_pixel_size=ref_thumbnail_pixel_size
        self.moving_thumbnail_pixel_size=moving_thumbnail_pixel_size
        self._coarse_affine_matrix = None

    @property
    def coarse_affine_matrix(self):
        """Coarse affine in the thumbnail frame, as a 3x3.

        The single place this matrix is stored: assign a matrix measured
        elsewhere (any of the `register_coarse` entry points, or
        `align_refine.refine_affine_by_block_translation`) and the setter
        normalizes it; read it without assigning and it is registered lazily.
        Multi-level / multi-object orchestrators hold an `Aligner` and use this
        property instead of caching their own copy.
        """
        if self._coarse_affine_matrix is None:
            self.coarse_register_affine()
        return self._coarse_affine_matrix

    @coarse_affine_matrix.setter
    def coarse_affine_matrix(self, mx):
        mx = np.asarray(mx, dtype=float)
        if mx.shape == (2, 3):
            mx = np.vstack([mx, [0, 0, 1]])
        assert mx.shape == (3, 3), f"affine matrix must be 2x3 or 3x3, got {mx.shape}"
        self._coarse_affine_matrix = mx

    @property
    def has_coarse_affine_matrix(self):
        """Whether a coarse affine is available without registering one.

        Lets an orchestrator run its *own* coarse registration (with its own
        engine and defaults) only when needed, instead of tripping the lazy
        `coarse_register_affine` in the getter.
        """
        return self._coarse_affine_matrix is not None

    def coarse_register_affine(self, **kwargs):
        """Register the thumbnails and store the result in
        `coarse_affine_matrix`.

        `register_coarse.coarse_register` is the engine: it searches the
        intensity/orientation configs itself (so cross-modality and mirrored
        pairs need no explicit flip/invert flags) and falls back to a windowed
        route when one scan images only a portion of the other. `kwargs` are
        forwarded to it.

        The keypoint budget is the engine's `N_KEYPOINTS` -- deliberately not
        overridden here, so a bare `Aligner`, the multi-object baseline and the
        CLI all register the same way.
        """
        default_kwargs = {
            'plot_match_result': True
        }
        default_kwargs.update(kwargs)
        self.coarse_affine_matrix = register_coarse.coarse_register(
            np.asarray(self.ref_thumbnail),
            np.asarray(self.moving_thumbnail),
            pixel_size_left=self.ref_thumbnail_pixel_size,
            pixel_size_right=self.moving_thumbnail_pixel_size,
            **default_kwargs
        )

    @property
    def affine_matrix(self):
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
    level1=0,
    channel1=0, channel2=0,
    thumbnail_level1=-1, thumbnail_level2=-1,
    thumbnail_channel1=None, thumbnail_channel2=None,
    thumbnails_pixel_size=None,
):
    thumbnail_channel1 = thumbnail_channel1 or channel1
    thumbnail_channel2 = thumbnail_channel2 or channel2
    # `thumbnails_pixel_size` is the authoritative request for thumbnail
    # resolution: when given it wins outright over the thumbnail-level args
    # (both readers get thumbnails at the same physical pixel size).

    # a thumbnail pixel size is only meaningful when BOTH readers know their
    # own: the coarse route is chosen from the ratio of the two, so one real
    # size against the other's 1 um placeholder is worse than no size at all
    known_px_size = reader1.has_pixel_size and reader2.has_pixel_size

    # reader2's working level is not the caller's choice: it is the coarsest
    # level whose pixels are still no coarser than reader1's `level1`, so the
    # moving image is read at the resolution the reference grid can use and
    # never below it. Comparing the two physically requires both real pixel
    # sizes -- against a placeholder, level 0 is the safe answer.
    level2 = 0
    if known_px_size:
        level2 = level_at_px_size(
            reader2, reader1.pixel_size * reader1.level_downsamples[level1]
        )

    if thumbnails_pixel_size is not None:
        px = thumbnails_pixel_size
        thumbnail1 = make_thumbnail_at_px_size(reader1, px, thumbnail_channel1)
        thumbnail2 = make_thumbnail_at_px_size(reader2, px, thumbnail_channel2)

        aligner = Aligner(
            reader1.read_level_channels(level1, channel1),
            reader2.read_level_channels(level2, channel2),
            thumbnail1,
            thumbnail2,
            px / reader1.pixel_size / reader1.level_downsamples[level1],
            px / reader2.pixel_size / reader2.level_downsamples[level2],
            px if known_px_size else None,
            px if known_px_size else None,
        )
    else:
        if None in [thumbnail_level1, thumbnail_level2]:
            thumbnail_level1, thumbnail_level2 = match_thumbnail_level(
                [reader1, reader2]
            )
        if thumbnail_level1 <= -1: thumbnail_level1 += len(reader1.pyramid)
        if thumbnail_level2 <= -1: thumbnail_level2 += len(reader2.pyramid)
        thumbnail_px1 = thumbnail_px2 = None
        if known_px_size:
            thumbnail_px1 = reader1.pixel_size * reader1.level_downsamples[thumbnail_level1]
            thumbnail_px2 = reader2.pixel_size * reader2.level_downsamples[thumbnail_level2]
        aligner = Aligner(
            reader1.read_level_channels(level1, channel1),
            reader2.read_level_channels(level2, channel2),
            reader1.read_level_channels(thumbnail_level1, thumbnail_channel1),
            reader2.read_level_channels(thumbnail_level2, thumbnail_channel2),
            reader1.level_downsamples[thumbnail_level1] / reader1.level_downsamples[level1],
            reader2.level_downsamples[thumbnail_level2] / reader2.level_downsamples[level2],
            thumbnail_px1,
            thumbnail_px2,
        )
    # the levels the affine was fit at: anything warping a whole pyramid level
    # (rather than `aligner.moving_img`) must read `reader2.pyramid[level2]`,
    # or the affine and the pixels it is applied to are a power of two apart
    aligner.level1, aligner.level2 = level1, level2
    return aligner


def level_at_px_size(reader, px_size, rtol=1e-3):
    """The coarsest pyramid level whose pixel size is still <= `px_size`.

    The one place a physical pixel size becomes a pyramid level. Taking the
    coarsest level that is not coarser than the target reads the fewest pixels
    without giving up resolution the target asked for; when every level is
    coarser than the target, level 0 is the closest available.

    `rtol` absorbs the float drift in `level_downsamples` (derived from level
    shapes that round up *or* down), so a level that is nominally an exact
    match counts as one instead of being rejected by a strict comparison and
    silently costing a 4x-larger read.
    """
    levels = sorted(reader.level_downsamples)
    px_sizes = np.array([
        reader.pixel_size * reader.level_downsamples[ll] for ll in levels
    ])
    not_coarser = np.nonzero(px_sizes <= px_size * (1 + rtol))[0]
    return int(levels[not_coarser[-1]]) if len(not_coarser) else 0


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


def make_thumbnail_at_px_size(reader, px_size, channel):
    if not reader.has_pixel_size:
        logger.warning(
            f"Requesting a {px_size:g} µm thumbnail from an image with no pixel"
            f" size metadata; the 1 µm placeholder makes the resulting scale"
            f" arbitrary. Pass `pixel_size=` to the reader"
        )
    # only ever downsample to reach `px_size` (never upsample), except when the
    # target is finer than level 0 and there is nothing to downsample from
    level = level_at_px_size(reader, px_size)
    level_px_size = reader.pixel_size * reader.level_downsamples[level]
    factor = level_px_size / px_size

    # `level_at_px_size`'s tolerance can accept a level a hair coarser than the
    # target; that is a match, not an upsample, so warn only past the tolerance
    if factor > 1.001:
        logger.warning(
            f"Requested thumbnail pixel size {px_size:g} µm is finer than the"
            f" finest available level ({level_px_size:g} µm); upsampling"
            f" {factor:.2f}x."
        )

    ori = reader.read_level_channels(level, channel)

    if max(ori.shape) > 20000:
        logger.warning(
            f"Reading level {level} at shape {tuple(ori.shape)} (> 20000 px) to build"
            f" a {px_size:g} µm thumbnail; this may be slow or memory-heavy."
        )

    ori = np.asarray(ori)
    return cv2.resize(
        ori, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
    )
