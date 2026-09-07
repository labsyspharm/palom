import copy

import dask.array as da
import numpy as np

from . import align, img_util


def coarsen_2x(arr):
    """One rung of the multi-res ladder: a 2x area mean, computed chunk-locally.

    `img_util.cv2_downscale_local_mean` is palom's downsampler everywhere else
    (`register_coarse`, `register_util`, `align_multi_obj`), so the ladder and
    the coarse fit see pixels reduced the same way. It is ~6x faster than a
    reshape-mean and keeps `ceil(n/2)`, where `da.coarsen(trim_excess=True)`
    floors and drops the trailing pixel -- which would erode the right/bottom
    edge once per rung and stop the shape ratios from being exactly 2.

    No halo is needed: with an even chunk size, output pixel `i` averages input
    `[2i, 2i+1]`, both inside the chunk, so the per-block result is identical to
    running the whole array through at once. Only a trailing odd chunk reaches
    cv2's border handling, and there a reflected edge value beats dropping the
    pixel.
    """
    return da.map_blocks(
        img_util.cv2_downscale_local_mean, arr, 2, dtype=arr.dtype,
        chunks=tuple(tuple(-(-c // 2) for c in ax) for ax in arr.chunks),
    )


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


def moving_coarsen_exponent(ref_px, moving_px, n_rungs):
    """How many 2x reductions each rung's *moving* image gets.

    The reference ladder is 2x per rung by construction, so rung `k`'s reference
    pixel is `ref_px * 2**k`. The moving image starts at `moving_px`, which is
    set by `level2` -- the coarsest level of reader2 that is still no coarser
    than `level1`. Because a file pyramid steps by 4x as readily as by 2x, that
    starting point can be most of a factor of two finer than it needs to be
    (melanoma pair: 0.263 um against a 0.65 um reference), and coarsening the
    moving side in lockstep with the reference carries that mismatch all the way
    up: rung 2 asks for 2.6 um and gets 1.05 um, four times the pixels it can
    use.

    So each rung takes as many 2x reductions as fit under its own reference
    pixel -- `floor(log2(ref_px_k / moving_px))` -- which makes the exponents
    non-uniform (0, 2, 3 on the melanoma pair, i.e. a 4x step then a 2x step).
    Two clamps keep the result buildable and safe:

    - rung 0 is pinned to 0. Its `moving_img` is the array `level2` names, and
      `Aligner.level2` is what a caller warps a whole pyramid level with
      (`cli.align_he` reads `reader2.pyramid[level2]`); coarsening it here would
      put the finest affine in a frame no level corresponds to.
    - the sequence is forced non-decreasing, so each rung is reachable from the
      one below by whole 2x steps and the ladder can chain (and share) its
      coarsening instead of re-reducing the base every time.

    A moving image *coarser* than the reference clamps to 0 throughout: there is
    nothing to gain by upsampling, and the reference-side ladder still coarsens.
    """
    exponents, prev = [], 0
    for k in range(n_rungs):
        want = int(np.floor(np.log2(ref_px * 2**k / moving_px)))
        prev = 0 if k == 0 else max(prev, want, 0)
        exponents.append(prev)
    return exponents


class MultiResAligner:

    def __init__(
        self,
        reader1, reader2,
        level1=0,
        channel1=0, channel2=0,
        thumbnail_channel1=None, thumbnail_channel2=None,
        thumbnail_level1=-1,
        thumbnails_pixel_size=None,
        min_num_blocks=4,
    ) -> None:
        self.reader1 = reader1
        self.reader2 = reader2
        self.level1 = level1

        self.channel1 = channel1
        self.channel2 = channel2
        # `is None`, not `or`: channel 0 is a legitimate request
        self.thumbnail_channel1 = (
            channel1 if thumbnail_channel1 is None else thumbnail_channel1
        )
        self.thumbnail_channel2 = (
            channel2 if thumbnail_channel2 is None else thumbnail_channel2
        )
        self.thumbnail_level1 = thumbnail_level1
        self.thumbnails_pixel_size = thumbnails_pixel_size

        self.min_num_blocks = min_num_blocks

        self._make_aligners()

    @property
    def downsample_factors(self):
        # Exact powers of two off `level1`'s factor. Every rung is `level1`
        # coarsened 2x per step, so these are known by construction rather than
        # measured from level shapes -- none of `reader.level_downsamples`'
        # rounding drift reaches the cross-rung mapping.
        base = self.reader1.level_downsamples[self.level1]
        return [base * 2**k for k in range(len(self.aligners))]

    @property
    def moving_downsample_factors(self):
        """Per-rung moving downsample, in reader2 level-0 pixels -- the moving
        counterpart of `downsample_factors`.

        Deliberately *not* `base * 2**k`: the moving image starts at whatever
        pixel size `level2` is, which is generally not a power of two away from
        `level1`'s, so locking it to the reference's 2x steps carries that
        starting mismatch into every rung. See `moving_coarsen_exponent`. The
        two lists are the thing to read side by side when a rung's affine looks
        scaled wrong.
        """
        base = self.reader2.level_downsamples[self.aligners[0].level2]
        return [base * 2**e for e in self._moving_exponents]

    @property
    def block_footprints(self):
        """(row, col) level-0 pixels covered by one block, per aligner rung.

        Not the same as `downsample_factors`: a block's footprint is chunk size
        *times* downsample. Rungs 1+ are rechunked to the base's chunk size, so
        their footprints double exactly; the base carries whatever chunking the
        reader handed over. Only the footprint ratio maps one rung's grid onto
        another's.
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

    # Rungs above the finest are materialized so each is reduced from the one
    # before it rather than by re-walking the base. Capped because rung 1 is a
    # quarter of the base: a whole-slide level-0 base would otherwise put
    # gigabytes in memory. Over budget, a rung stays lazy -- cheaply, as long as
    # its parent was persisted; see `_persist_rung`.
    PERSIST_BUDGET = 2 << 30

    def _make_aligners(self):
        """Build the resolution ladder from `level1` alone.

        The coarser rungs are coarsened here rather than read from
        `reader1.pyramid[level1 + k]`, so nothing downstream depends on how the
        file's own pyramid was built: every rung is reduced from the caller's
        starting level by the same filter. The reference's downsample factors
        are exactly 2**k; the moving side's are not (see
        `moving_coarsen_exponent`), because its starting level is generally not
        a power of two away from `level1`. The coarse fit still uses the file's
        thumbnail levels -- it is the block-shift refinement that is meant to
        correct a poorly built pyramid.
        """
        if not 0 <= self.level1 < len(self.reader1.pyramid):
            raise ValueError(
                f"level1={self.level1} is not a valid level of"
                f" {type(self.reader1).__name__} (pyramid has"
                f" {len(self.reader1.pyramid)} level(s))"
            )
        # The one reader touch. `level1` is also the frame every caller reads
        # results back in (`aligners[0].affine_matrix`, and the grid
        # `MultiObjAligner` builds its block masks on), so the base rung is kept
        # unconditionally -- `min_num_blocks` only decides how far the ladder
        # extends above it, and an empty `aligners` would crash with a bare
        # IndexError from the `coarse_affine_matrix` getter.
        base = align.get_aligner(
            self.reader1, self.reader2,
            channel1=self.channel1, channel2=self.channel2,
            level1=self.level1,
            thumbnail_channel1=self.thumbnail_channel1,
            thumbnail_channel2=self.thumbnail_channel2,
            thumbnail_level1=self.thumbnail_level1,
            # FIXME handle user selected thumbnail level
            thumbnail_level2=None,
            thumbnails_pixel_size=self.thumbnails_pixel_size,
        )
        # `constrain_shifts` maps grids across rungs by block footprint, which
        # assumes every block of a rung covers the same area -- only the trailing
        # chunk may be short. Rungs 1+ are rechunked to the base's chunk size, so
        # only the base can violate it.
        for chunks in base.ref_img.chunks:
            assert len(set(chunks[:-1])) <= 1, (
                f"level {self.level1} has irregular chunks {chunks}; cross-rung"
                " block mapping requires uniformly sized blocks"
            )
        self.aligners = [base]
        # spent by `_persist_rung` as the ladder is built
        self._persist_budget = self.PERSIST_BUDGET
        # 2x reductions applied to each rung's moving image, relative to
        # `level2`. Tracked as it is built because the loop chains from `prev`
        # and needs to know how many steps are still owed.
        self._moving_exponents = [0]
        # Nominal level per rung: the pixels all come from `level1`, coarsened
        # 2**k. Only `MultiObjAligner`'s finest-rung assert and the `plot_shifts`
        # tick labels read this -- nothing indexes `reader1.pyramid` with it.
        self.levels = [self.level1]
        ref_chunks = base.ref_img.chunksize
        moving_chunks = base.moving_img.chunksize
        # The two sides are coarsened independently: the reference by 2x per
        # rung (so `downsample_factors` stays exactly 2**k and the cross-rung
        # shift scaling is exact), the moving side by whatever whole number of
        # 2x steps fits under that rung's reference pixel. Both are still
        # reduced from the caller's starting levels by the same filter -- the
        # ladder never reads a level of the file's own pyramid.
        known_px = self.reader1.has_pixel_size and self.reader2.has_pixel_size
        ref_px = self.reader1.pixel_size * self.reader1.level_downsamples[self.level1]
        moving_px = (
            self.reader2.pixel_size * self.reader2.level_downsamples[base.level2]
        )
        while True:
            prev = self.aligners[-1]
            if min(*prev.ref_img.shape, *prev.moving_img.shape) < 2:
                break
            k = len(self.aligners)
            # without both pixel sizes the ratio is meaningless (one reader's
            # placeholder 1 um against the other's real size would pick a wildly
            # wrong factor), so fall back to the reference's lockstep 2x
            exponent = (
                moving_coarsen_exponent(ref_px, moving_px, k + 1)[-1]
                if known_px
                else k
            )
            rung = copy.copy(prev)
            rung.ref_img = coarsen_2x(prev.ref_img).rechunk(ref_chunks)
            moving = prev.moving_img
            for _ in range(exponent - self._moving_exponents[-1]):
                moving = coarsen_2x(moving)
            rung.moving_img = moving.rechunk(moving_chunks)
            if (
                rung.ref_img.npartitions < self.min_num_blocks
                or min(*rung.moving_img.shape) < 2
            ):
                break
            # The reference halves every rung; the moving side may have taken
            # more than one step, so its factor comes off the base rather than
            # from halving `prev` -- `2**exponent` is exact where a chain of
            # divisions is only nominally so.
            rung.ref_thumbnail_down_factor = prev.ref_thumbnail_down_factor / 2
            rung.moving_thumbnail_down_factor = (
                base.moving_thumbnail_down_factor / 2**exponent
            )
            self.aligners.append(rung)
            self.levels.append(self.levels[-1] + 1)
            self._moving_exponents.append(exponent)
            # before the next iteration coarsens from it -- see `_persist_rung`
            self._persist_rung(rung)

    def _persist_rung(self, rung):
        """Materialize a rung, in place, as soon as it is built.

        Staying lazy is far more expensive than it looks. Every block of
        `compute_shifts` takes the rung's whole moving array as a `src_array`
        kwarg (`block_affine_transformed_moving_img`), so the rung's entire
        graph -- not the slice a block reads -- is a dependency of every block:
        a lazy rung re-reads and re-coarsens the full moving slide once per
        `compute_shifts` call, which on a real pair is minutes for a rung of a
        few dozen blocks. Measured on the twopiece pair at level1=1: a lazy rung
        carries ~47 000 tasks whatever its size (4 blocks or 660), against a few
        thousand once persisted.

        This runs per rung, inside the build loop, rather than once over the
        finished ladder. Persisting rung by rung does NOT re-walk the base --
        the coarser rungs chain off `prev`, which is already in memory by then
        -- and it is what keeps a rung that misses the budget cheap: it is then
        a coarsening of the last persisted rung instead of a second full pass
        over the base. Persisting the finished ladder in one call left the
        over-budget rungs pointing at the base, because their graphs were built
        before the parents were replaced.

        The budget is spent finest-rung-first, which is the order they are
        built; each rung is a quarter of the one below, so in practice the whole
        ladder above the first rung fits, and if it does not, the rungs that do
        fit still give the ones above them a cheap parent. The base itself is
        never persisted -- it is too big to hold and its blocks stream fine.
        """
        for name in ("ref_img", "moving_img"):
            arr = getattr(rung, name)
            if arr.nbytes > self._persist_budget:
                continue
            self._persist_budget -= arr.nbytes
            setattr(rung, name, arr.persist())

    def align(self, mask_fn=None):
        # `mask_fn(grid_shape) -> bool array` optionally restricts block-shift
        # computation to a region (e.g. one tissue object) at each level; it
        # must return a non-empty mask for every level so each aligner keeps
        # at least one finite block (an all-False mask crashes constrain).
        # read once, then fan out through the setter: the getter registers
        # lazily when nobody assigned a matrix, and doing that per level would
        # register once per level
        self.coarse_affine_matrix = self.coarse_affine_matrix
        for aligner in self.aligners:
            # `constrain_shifts` reads `original_shifts` as the pristine shifts
            # this run produced. Leaving a previous run's behind makes its
            # `original_shifts == shifts` validity test compare across runs and
            # silently mismark which blocks constrain moved.
            if hasattr(aligner, 'original_shifts'):
                del aligner.original_shifts
            if mask_fn is None:
                aligner.compute_shifts()
            else:
                mask = da.from_array(
                    np.asarray(mask_fn(aligner.grid_shape), dtype=bool), chunks=1
                )
                aligner.compute_shifts(mask=mask)

    def constrain_shifts(self, domain_tol=None):
        aligners = self.aligners
        for aligner in aligners:
            # `Aligner.constrain_shifts` is idempotent (it re-constrains from
            # `original_shifts`), and `align` clears that attribute, so this is
            # safe to call unconditionally. To suppress a level's result, use
            # withholding the constrain.
            # each rung partitions its own field: the rungs measure
            # independently, so they are entitled to disagree about where
            # the domains are
            aligner.constrain_shifts(domain_tol=domain_tol)
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
        # Which level supplied each block, -1 where no level was valid: argmax
        # of an all-False column returns 0, so level 0's extrapolated value is
        # written but is not a measurement and must not be reported as one.
        self.result_levels = np.where(
            np.any(valid_masks, axis=0), np.take(self.levels, mask), -1
        )
        # The phase-correlation error of whichever rung supplied each block, on
        # the finest grid. Not rescaled: it is a correlation confidence, not a
        # length. `inf` where the winning rung had nothing to measure.
        errors = np.full((h, w), np.inf)
        for ii, (aa, idx) in enumerate(zip(aligners, idxs)):
            # `result_levels == -1` marks the blocks argmax handed to rung 0
            # without any rung passing; their shift is rung 0's extrapolation,
            # so rung 0's measured error would misdescribe it. Leave those inf.
            sel = (mask == ii) & (self.result_levels >= 0)
            if sel.any():
                errors[sel] = np.asarray(aa.shift_errors)[idx[sel]]
        self.shift_errors = errors.ravel()

    def level_histogram(self, mask=None):
        """Block counts keyed by the level that supplied the shift, -1 = none.

        A field mostly filled from coarse levels is a field mostly extrapolated
        from elsewhere; without this it reads the same as a measured one.
        """
        levels = self.result_levels.ravel()
        if mask is not None:
            levels = levels[np.asarray(mask).ravel()]
        return {
            int(k): int(v)
            for k, v in zip(*np.unique(levels, return_counts=True))
        }

    def plot_shifts(self, max_radius=None, domain_labels=None):
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
            # `np.percentile` of an empty selection raises; an all-invalid grid
            # (every level's constrain early-returned) is a legitimate, if
            # useless, thing to plot
            valid_magnitudes = np.linalg.norm(shifts, axis=0)[mask]
            max_radius = (
                np.percentile(valid_magnitudes, 99.5) if valid_magnitudes.size else 1.0
            )
        # a degenerate all-zero shift field would otherwise divide by zero in
        # `shifts_to_lab`
        max_radius = max(float(max_radius), np.finfo("float32").eps)

        lab = flow.shifts_to_lab(
            shifts,
            max_radius=max_radius
        )
        rgb = skimage.color.lab2rgb(lab, channel_axis=0)
        
        # backdrop only -- the coarsest level, independent of `thumbnail_level1`
        # (which selects the *registration* thumbnail, and may be a fixed
        # physical pixel size rather than a level at all)
        thumbnail_level = len(self.reader1.pyramid) - 1
        thumbnail = self.reader1.pyramid[thumbnail_level][
            self.thumbnail_channel1
        ].compute()
        thumbnail_extent = flow.get_img_extent(
            thumbnail.shape, self.reader1.level_downsamples[thumbnail_level]
        )
        
        flow_extent = flow.get_img_extent(shape, self.block_footprints[0][0])

        # A fourth panel only when there is a partition to show. Without it
        # the domains are visible as log text alone, which cannot answer the
        # question the figure exists for: *where* are they.
        n_panels = 3 + (domain_labels is not None)
        widths = (.5, 1, 1) + ((1,) if domain_labels is not None else ())
        w, h = matplotlib.figure.figaspect(shape[0] / (shape[1] * n_panels))
        fig = plt.figure(figsize=(w, h))
        gs = fig.add_gridspec(1, n_panels, width_ratios=widths)

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

        if domain_labels is not None:
            labels = np.asarray(domain_labels)
            ax3 = fig.add_subplot(gs[3], sharex=ax1, sharey=ax1)
            ax3.imshow(
                np.log1p(thumbnail), alpha=1, extent=thumbnail_extent, cmap="gray"
            )
            n_dom = int(labels.max()) + 1
            if n_dom > 0:
                # tab20 rather than Set3: adjacent domains have to be told
                # apart, and Set3's pastels wash out over the backdrop
                dcmap = matplotlib.colors.ListedColormap(
                    matplotlib.colormaps["tab20"].colors[: max(n_dom, 1)]
                )
                # loose blocks are transparent, not a colour -- they are the
                # absence of a domain, and colouring them invents one
                dim = ax3.imshow(
                    np.where(labels >= 0, labels, 0),
                    extent=flow_extent, alpha=0.55 * (labels >= 0),
                    cmap=dcmap, vmin=-0.5, vmax=n_dom - 0.5,
                )
                dcax = make_axes_locatable(ax3).append_axes(
                    "right", size="5%", pad=0.05
                )
                cb = plt.colorbar(dim, cax=dcax, ticks=range(n_dom))
                cb.set_label("domain", fontsize=6)
            else:
                make_axes_locatable(ax3).append_axes(
                    "right", size="5%", pad=0.05
                ).axis("off")
            # Deliberately no loose count: `plot_shifts` does not know which
            # blocks belong to the object, so it can only count the whole grid
            # -- 650 "loose" for an object of 1698 blocks that is 100% covered.
            # Loose blocks are visible here as transparency, and the log line
            # carries the count bounded to the object.
            ax3.set_title(f"{n_dom} domain(s)", fontsize=6)
            ax3.axis("off")

        _ = flow.plot_legend(
            np.array([*shifts, mask]),
            max_radius, True, True, plot_flow=False, ax=fig.add_subplot(gs[0])
        )
        return fig
