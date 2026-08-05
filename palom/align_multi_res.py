import dask.array as da
import numpy as np

from . import align


def map_to_finest_grid(arr, scale, out_shape):
    """Resample a per-block array onto the finest aligner's block grid.

    `scale` is the (row, col) ratio of this level's block *footprint* -- the
    level-0 area one block covers -- to the finest level's. Every level's block
    grid starts at pixel 0 of the same image, so a fine block maps exactly to
    the coarse block containing its center. That reduces to a plain tile-repeat
    when the ratio is an integer and stays correct when it is not.
    """
    (nr, nc), (h, w) = arr.shape, out_shape
    rr = np.minimum(((np.arange(h) + 0.5) / scale[0]).astype(int), nr - 1)
    cc = np.minimum(((np.arange(w) + 0.5) / scale[1]).astype(int), nc - 1)
    return arr[np.ix_(rr, cc)]


class MultiResAligner:

    def __init__(
        self,
        reader1, reader2,
        level1=0,
        channel1=0, channel2=0,
        thumbnail_channel1=None, thumbnail_channel2=None,
        thumbnail_level1=-1,
        thumbnails_pixel_size=None,
        min_num_blocks=4
    ) -> None:
        self.reader1 = reader1
        self.reader2 = reader2
        self.level1 = level1

        self.channel1 = channel1
        self.channel2 = channel2
        self.thumbnail_channel1 = thumbnail_channel1 or channel1
        self.thumbnail_channel2 = thumbnail_channel2 or channel2
        self.thumbnail_level1 = thumbnail_level1
        self.thumbnails_pixel_size = thumbnails_pixel_size

        self.min_num_blocks = min_num_blocks
        
        self._make_aligners()

    @property
    def downsample_factors(self):
        return [
            self.reader1.level_downsamples[ll]
            for ll in self.levels
        ]

    @property
    def block_footprints(self):
        """(row, col) level-0 pixels covered by one block, per aligner level.

        Not the same as `downsample_factors`: a block's footprint is chunk size
        *times* downsample, and chunk size is not guaranteed constant across
        pyramid levels (`reader.auto_format_pyramid` halves it at every level,
        which keeps the footprint -- and hence the grid -- identical at all
        levels). Only the footprint ratio maps one level's grid onto another's.
        """
        return [
            np.multiply(al.ref_img.chunksize[-2:], dd)
            for al, dd in zip(self.aligners, self.downsample_factors)
        ]

    @property
    def coarse_affine_matrix(self):
        # no separate copy is kept here: the finest aligner owns the matrix and
        # `align` propagates it to the coarser levels. `MultiObjAligner` always
        # assigns one (per object); reading it unassigned falls back to the base
        # `Aligner.coarse_register_affine` and its smaller keypoint budget.
        return self.aligners[0].coarse_affine_matrix

    @coarse_affine_matrix.setter
    def coarse_affine_matrix(self, mx):
        for aligner in self.aligners:
            aligner.coarse_affine_matrix = mx

    def _make_aligners(self):
        self.aligners = []
        # `levels` must be exactly the levels that produced `aligners` -- the
        # two are zipped in `constrain_shifts` and `plot_shifts`. Deriving it
        # from `reader1.pyramid[x].numblocks` instead double-counts the channel
        # axis (readers chunk it to 1), which keeps levels that get no aligner.
        self.levels = []
        for l1 in range(self.level1, len(self.reader1.pyramid)):
            c21l = align.get_aligner(
                self.reader1, self.reader2,
                channel1=self.channel1, channel2=self.channel2,
                level1=l1,
                thumbnail_channel1=self.thumbnail_channel1,
                thumbnail_channel2=self.thumbnail_channel2,
                thumbnail_level1=self.thumbnail_level1,
                # FIXME handle user selected thumbnail level
                thumbnail_level2=None,
                thumbnails_pixel_size=self.thumbnails_pixel_size,
            )
            if c21l.num_blocks < self.min_num_blocks:
                continue
            # `constrain_shifts` maps grids across levels by block footprint,
            # which assumes every block of a level covers the same area -- only
            # the trailing chunk may be short. Regular for any zarr/tiff-backed
            # pyramid; a rechunked or sliced input could break it.
            for chunks in c21l.ref_img.chunks:
                assert len(set(chunks[:-1])) <= 1, (
                    f"level {l1} has irregular chunks {chunks}; cross-level"
                    " block mapping requires uniformly sized blocks"
                )
            self.aligners.append(c21l)
            self.levels.append(l1)

    def align(self, mask_fn=None):
        # `mask_fn(grid_shape) -> bool array` optionally restricts block-shift
        # computation to a region (e.g. one tissue object) at each level; it
        # must return a non-empty mask for every level so each aligner keeps
        # at least one finite block (an all-False mask crashes constrain).
        self._aligner_shifts = []
        # read once: the getter registers lazily when nobody assigned a matrix
        coarse_affine_matrix = self.coarse_affine_matrix
        for aligner in self.aligners:
            aligner.coarse_affine_matrix = coarse_affine_matrix
            if mask_fn is None:
                aligner.compute_shifts()
            else:
                mask = da.from_array(
                    np.asarray(mask_fn(aligner.grid_shape), dtype=bool), chunks=1
                )
                aligner.compute_shifts(mask=mask)
            self._aligner_shifts.append(aligner.shifts)

    def constrain_shifts(self, exclude_result_levels=None):
        aligners = self.aligners
        for aligner in aligners:
            # FIXME workaround to manually exclude computed shifts from certain levels
            if not hasattr(aligner, 'original_shifts'):
                aligner.constrain_shifts()
        _valid_masks = [
            # a block is valid where constrain left it unchanged AND it is
            # finite -- masked-out / unconstrained blocks are inf and must not
            # win the cross-level argmax below
            np.all(al.original_shifts == al.shifts, axis=1)
            & np.all(np.isfinite(al.shifts), axis=1)
            for al in aligners
        ]
        h, w = aligners[0].grid_shape
        # Two distinct ratios, easily conflated: `grid_scales` maps a level's
        # block grid onto the finest one (ratio of block footprints), while
        # `pixel_scales` converts a shift measured in that level's pixels to
        # the finest level's pixels (ratio of downsample factors). They agree
        # only when every level shares a chunk size, and neither is guaranteed
        # to be an integer (a non-integer downsample factor is a warning in
        # `reader.level_downsamples`, not an error).
        footprints = self.block_footprints
        grid_scales = [np.divide(ff, footprints[0]) for ff in footprints]
        pixel_scales = [
            dd / self.downsample_factors[0] for dd in self.downsample_factors
        ]
        valid_masks = [
            map_to_finest_grid(mm.reshape(aa.grid_shape), ss, (h, w))
            for ss, mm, aa in zip(grid_scales, _valid_masks, aligners)
        ]
        idxs = [
            map_to_finest_grid(
                np.arange(aa.shifts.shape[0]).reshape(aa.grid_shape), ss, (h, w)
            )
            for aa, ss in zip(aligners, grid_scales)
        ]
        exclude_result_levels = exclude_result_levels or []
        for level in exclude_result_levels:
            valid_masks[level][:] = False
        mask = np.argmax(valid_masks, axis=0)
        out = np.zeros((2, *aligners[0].grid_shape))
        for ii, (aa, idx, ss) in enumerate(
            zip(aligners, idxs, pixel_scales)
        ):
            out[np.array([mask == ii]*2)] = (
                ss * aa.shifts[idx[mask == ii]].T.flatten()
            )
        # Block shifts are residuals on top of the affine, so `shift == 0` means
        # "defer to the affine". Residual inf only survives at outside-object
        # blocks of a degenerate object (level-0 constrain early-returned); zero
        # them so the field is finite by contract -- inside-object blocks are
        # always finite -- and downstream warps don't hit inf * 0 = NaN or bleed
        # inf through cv2 interpolation at the object boundary.
        out = np.where(np.isfinite(out), out, 0.0)
        self.shifts = out.reshape(2, -1).T
        self.valid_masks = valid_masks
        self.idxs = idxs

    def plot_shifts(self, max_radius=None):
        import matplotlib.colors
        import matplotlib.figure
        import matplotlib.pyplot as plt
        import skimage.color
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .cli import flow
        
        shape = self.aligners[0].grid_shape
        shifts = self.shifts.T.reshape(2, *shape)

        # `valid_masks` already encodes finiteness (constrain's isfinite guard)
        # and `self.shifts` is finite by contract, so the valid region is exactly
        # where any level won.
        mask = np.max(self.valid_masks, axis=0)

        if max_radius is None:
            max_radius = np.percentile(np.linalg.norm(shifts, axis=0)[mask], 99.5)

        lab = flow.shifts_to_lab(
            shifts,
            max_radius=max_radius
        )
        rgb = skimage.color.lab2rgb(lab, channel_axis=0)
        
        thumbnail = self.reader1.pyramid[-1][self.thumbnail_channel1].compute()
        thumbnail_extent = flow.get_img_extent(
            thumbnail.shape,
            self.reader1.level_downsamples[len(self.reader1.pyramid)-1]
        )
        
        flow_extent = flow.get_img_extent(shape, self.block_footprints[0][0])

        w, h = matplotlib.figure.figaspect(shape[0] / (shape[1] * 3))
        fig = plt.figure(figsize=(w, h))
        gs = fig.add_gridspec(1, 3, width_ratios=(.5, 1, 1))

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(np.log1p(thumbnail), alpha=1, extent=thumbnail_extent, cmap='gray')
        ax1.imshow(np.dstack([*rgb, mask]), extent=flow_extent, alpha=0.8)
        _cax = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
        _cax.axis('off')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
        ax2.imshow(np.log1p(thumbnail), alpha=1, extent=thumbnail_extent, cmap='gray')
        # Use the first N of Set3's 12 fixed colors so a given level keeps the
        # same color regardless of total level count, while the colorbar only
        # spans the levels that actually exist (argmax indexes 0..N-1).
        n_masks = len(self.valid_masks)
        cmap = matplotlib.colors.ListedColormap(
            matplotlib.colormaps['Set3'].colors[:n_masks]
        )
        im = ax2.imshow(
            np.argmax(self.valid_masks, axis=0),
            extent=flow_extent, alpha=0.5 * mask,
            cmap=cmap, vmin=-.5, vmax=n_masks-.5,
        )
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(im, cax=cax, ticks=range(len(self.valid_masks)))
        cbar.set_ticklabels(self.levels)

        _ = flow.plot_legend(
            np.array([*shifts, mask]),
            max_radius, True, True, plot_flow=False, ax=fig.add_subplot(gs[0])
        )
        return fig
