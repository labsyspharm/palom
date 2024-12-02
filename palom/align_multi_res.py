import numpy as np

from . import align
from . import img_util


def match_levels(r1, r2):
    px_sizes_1 = [
        r1.pixel_size * r1.level_downsamples[i]
        for i in range(len(r1.pyramid))
    ]
    px_sizes_2 = [
        r2.pixel_size * r2.level_downsamples[i]
        for i in range(len(r2.pyramid))
    ]
    return [
        (idx, np.where(np.array(px_sizes_2) < px1)[0].max())
        if np.min(px_sizes_2) < px1
        else (idx, 0)
        for idx, px1 in enumerate(px_sizes_1)
    ]


class MultiResAligner:

    def __init__(
        self,
        reader1, reader2,
        level1=0,
        channel1=0, channel2=0,
        thumbnail_channel1=None, thumbnail_channel2=None,
        thumbnail_level1=-1,
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

        self.min_num_blocks = min_num_blocks
        
        self._make_aligners()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        props = ['aligners', 'base_aligners']
        for pp in props:
            if pp in state: del state[pp]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(
            state['reader1'], state['reader2'],
            level1=state['level1'],
            channel1=state['channel1'], channel2=state['channel2'],
            thumbnail_channel1=state['thumbnail_channel1'],
            thumbnail_channel2=state['thumbnail_channel2'],
            thumbnail_level1=state['thumbnail_level1'],
            min_num_blocks=state['min_num_blocks']
        )
        if '_coarse_affine_matrix' in state:
            for aligner in self.aligners:
                aligner.coarse_affine_matrix = state['_coarse_affine_matrix']
        if '_aligner_shifts' in state:
            if len(state['_aligner_shifts']) == len(self.aligners):
                for aa, ss in zip(self.aligners, state['_aligner_shifts']):
                    aa.shifts = ss
        # FIXME this does not fully recover the state before pickling, running
        # `constrain_shifts()` is required after loading the pickled

    @property
    def levels(self):
        levels = filter(
            lambda x: x >= self.level1,
            self.reader1.level_downsamples.keys()
        )
        levels = filter(
            lambda x: np.prod(self.reader1.pyramid[x].numblocks) >= self.min_num_blocks,
            levels
        )
        return sorted(levels)

    @property
    def downsample_factors(self):
        return [
            self.reader1.level_downsamples[ll]
            for ll in self.levels
        ]

    @property
    def level_pairs(self):
        levels = match_levels(self.reader1, self.reader2)
        filtered = filter(lambda x: x[0] >= self.level1, levels)
        return list(filtered)
    
    @property
    def base_aligner(self):
        import copy
        aligner = copy.deepcopy(self.aligners[0])
        if hasattr(self, 'shifts'):
            aligner.shifts = self.shifts
        return aligner

    @property
    def coarse_affine_matrix(self):
        if not hasattr(self, '_coarse_affine_matrix'):
            self._coarse_align()
        return self._coarse_affine_matrix
    
    def _make_aligners(self):
        self.level2 = self.level_pairs[0][1]
        self.aligners = []
        for l1, l2 in self.level_pairs:
            c21l = align.get_aligner(
                self.reader1, self.reader2, 
                channel1=self.channel1, channel2=self.channel2,
                level1=l1, level2=l2,
                thumbnail_channel1=self.thumbnail_channel1,
                thumbnail_channel2=self.thumbnail_channel2,
                thumbnail_level1=self.thumbnail_level1,
                # FIXME handle user selected thumbnail level
                thumbnail_level2=None
            )
            if c21l.num_blocks < self.min_num_blocks:
                continue
            self.aligners.append(c21l)

    def _coarse_align(self, **kwargs):
        l1, l2 = self.level_pairs[0]
        aligner = align.get_aligner(
            self.reader1, self.reader2,
            thumbnail_channel1=self.thumbnail_channel1,
            thumbnail_channel2=self.thumbnail_channel2,
            thumbnail_level1=self.thumbnail_level1,
            # FIXME handle user selected thumbnail level
            thumbnail_level2=None,
            channel1=self.channel1,
            channel2=self.channel2,
            level1=l1, level2=l2,
        )
        default_kwargs = {
            'n_keypoints': 20_000,
            'plot_match_result': True,
            'test_flip': True,
            'test_intensity_invert': True,
        }
        aligner.coarse_register_affine(**{**default_kwargs, **kwargs})
        self._coarse_affine_matrix = aligner.coarse_affine_matrix

    def align(self):
        self._aligner_shifts = []
        for aligner in self.aligners:
            aligner.coarse_affine_matrix = self.coarse_affine_matrix
            aligner.compute_shifts()
            self._aligner_shifts.append(aligner.shifts)

    def constrain_shifts(self, exclude_result_levels=None):
        aligners = self.aligners
        for aligner in aligners:
            # FIXME workaround to manually exclude computed shifts from certain levels
            if not hasattr(aligner, 'original_shifts'):
                aligner.constrain_shifts()
        _valid_masks = [
            np.all(al.original_shifts == al.shifts, axis=1)
            for al in aligners
        ]
        h, w = aligners[0].grid_shape
        downsample_factors = [
            int(dd / self.downsample_factors[0])
            for dd in self.downsample_factors
        ]
        valid_masks = [
            img_util.repeat_2d(mm.reshape(aa.grid_shape), (dd, dd))[:h, :w]
            for dd, mm, aa in zip(
                downsample_factors, _valid_masks, aligners
            ) 
        ]
        idxs = [
            img_util.repeat_2d(
                np.arange(aa.shifts.shape[0]).reshape(aa.grid_shape), 
                (dd, dd)
            )[:h, :w]
            for aa, dd in zip(aligners, downsample_factors)
        ]
        exclude_result_levels = exclude_result_levels or []
        for level in exclude_result_levels:
            valid_masks[level][:] = False
        mask = np.argmax(valid_masks, axis=0)
        out = np.zeros((2, *aligners[0].grid_shape))
        for ii, (aa, idx, dd) in enumerate(
            zip(aligners, idxs, downsample_factors)
        ):
            out[np.array([mask == ii]*2)] = (
                dd * aa.shifts[idx[mask == ii]].T.flatten()
            )
        self.shifts = out.reshape(2, -1).T
        self.valid_masks = valid_masks
        self.idxs = idxs

    def plot_shifts(self, max_radius=None):
        import matplotlib.pyplot as plt
        import matplotlib.figure
        import skimage.color
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        from .cli import flow
        
        shape = self.base_aligner.grid_shape
        shifts = self.shifts.T.reshape(2, *shape)

        shift_mask = shifts
        if hasattr(self.base_aligner, 'original_shifts'):
            shift_mask = self.base_aligner.original_shifts.T.reshape(2, *shape)
        shift_mask = np.all(np.isfinite(shift_mask), axis=0)
        mask = np.max(self.valid_masks, axis=0) & shift_mask

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
        
        flow_extent = flow.get_img_extent(
            shape,
            self.downsample_factors[0] * self.reader1.pyramid[self.levels[0]].chunksize[1]
        )

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
        im = ax2.imshow(
            np.argmax(self.valid_masks, axis=0),
            extent=flow_extent, alpha=0.5,
            cmap='Set3', vmin=-.5, vmax=12-.5,
        )
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, ticks=self.levels, values=self.levels)

        _ = flow.plot_legend(
            np.array([*shifts, mask]),
            max_radius, True, True, plot_flow=False, ax=fig.add_subplot(gs[0])
        )
        return fig
