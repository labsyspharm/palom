import itertools
from functools import cached_property

import cv2
import dask.array as da
import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.transform

from . import align, align_multi_res, align_refine, img_util, register_dev


def transform_bbox(bbox, affine_mx):
    tform_bbox = []
    tform = skimage.transform.AffineTransform(affine_mx)
    for rs, re, cs, ce in bbox:
        xx, yy = tform.inverse(
            list(itertools.product([cs, ce], [rs, re]))
        ).T
        rs2, cs2 = np.clip(np.floor([yy.min(), xx.min()]).astype(int), 0, None)
        re2, ce2 = np.ceil([yy.max(), xx.max()]).astype(int)
        tform_bbox.append([rs2, re2, cs2, ce2])
    return tform_bbox


class MultiObjAligner:

    def __init__(
        self,
        reader1, reader2,
        level1=0, level2=None,
        channel1=0, channel2=0,
        thumbnail_channel1=None, thumbnail_channel2=None,
        thumbnail_level1=-1,
    ) -> None:
        self.reader1 = reader1
        self.reader2 = reader2
        self.level1 = level1
        self.level2 = level2 or self._set_level2()

        self.channel1 = channel1
        self.channel2 = channel2
        self.thumbnail_channel1 = thumbnail_channel1 or channel1
        self.thumbnail_channel2 = thumbnail_channel2 or channel2
        self.thumbnail_level1 = thumbnail_level1

    def run(self, downscale_factor=8, exclude_objects=None, refine=True,
            multi_res=True, min_num_blocks=25):
        self.segment_objects(downscale_factor=downscale_factor, plot_segmentation=True)
        self.align_all_objects(
            plot_shift=True, refine=refine, multi_res=multi_res,
            min_num_blocks=min_num_blocks,
        )
        self.combine_object_results(exclude_objects=exclude_objects)

    @cached_property
    def aligner(self):
        return self.make_aligner()

    @cached_property
    def ref_thumbnail(self):
        return np.array(self.aligner.ref_thumbnail)
    
    @cached_property
    def moving_thumbnail(self):
        return np.array(self.aligner.moving_thumbnail)
    
    @cached_property
    def fill_value_ref_thumbnail(self):
        return np.mean(
            self.ref_thumbnail[~img_util.entropy_mask(self.ref_thumbnail)]
        )
    
    @cached_property
    def fill_value_moving_thumbnail(self):
        return np.mean(
            self.moving_thumbnail[~img_util.entropy_mask(self.moving_thumbnail)]
        )
    
    @property
    def baseline_affine_matrix(self):
        if not hasattr(self, '_affine_matrix'):
            self._coarse_align()
        return self._affine_matrix

    @property
    def baseline_coarse_affine_matrix(self):
        if not hasattr(self, '_coarse_affine_matrix'):
            self._coarse_align()
        return self._coarse_affine_matrix
    
    @property
    def bbox_ref_thumbnail(self):
        if not hasattr(self, '_bbox_ref_thumbnail'):
            self.segment_objects(plot_segmentation=True)
        return self._bbox_ref_thumbnail
    
    @cached_property
    def bbox_ref_img_block(self):
        bbox = self.bbox_ref_thumbnail.astype(float)
        c21l = self.aligner
        downsample_factor = c21l.ref_thumbnail_down_factor
        rchunk, cchunk = c21l.ref_img.chunksize
        bbox[:, 0:2] *= downsample_factor / rchunk
        bbox[:, 2:4] *= downsample_factor / cchunk
        bbox[:, [0, 2]] = np.clip(np.floor(bbox[:, [0, 2]]), 0, None)
        bbox[:, [1, 3]] = np.ceil(np.ceil(bbox[:, [1, 3]]))
        return bbox.astype(int)

    def _coarse_align(self, **kwargs):
        c21l = self.make_aligner()
        default_kwargs = {
            'n_keypoints': 20_000,
            'plot_match_result': True,
            'test_flip': True,
            'test_intensity_invert': True,
            'auto_mask': True
        }
        c21l.coarse_register_affine(**{**default_kwargs, **kwargs})
        self._coarse_affine_matrix = c21l.coarse_affine_matrix
        self._affine_matrix = c21l.affine_matrix

    def segment_objects(self, downscale_factor=8, min_area=None, plot_segmentation=False):
        shape = self.ref_thumbnail.shape
        mask = img_util.entropy_mask(
            img_util.cv2_downscale_local_mean(
                self.ref_thumbnail, downscale_factor
            )
        )
        labeled = skimage.morphology.label(mask)
        labeled = skimage.segmentation.expand_labels(labeled, 4)

        regionprops = skimage.measure.regionprops_table(
            labeled, properties=['label', 'bbox', 'area']
        )
        area = np.array(regionprops['area'])
        # drop specks/debris; default threshold is 1% of the largest object
        if min_area is None:
            min_area = 0.01 * area.max() if area.size else 0
        keep = area >= min_area
        order = np.argsort(area[keep])[::-1]  # largest object first
        bbox_ref_thumbnail = downscale_factor * np.array([
            regionprops['bbox-0'],
            regionprops['bbox-2'],
            regionprops['bbox-1'],
            regionprops['bbox-3']
        ]).T
        self._bbox_ref_thumbnail = bbox_ref_thumbnail[keep][order]
        # label value of each (sorted) object, so per-object masks can be read
        # back from `segmentation_mask` (which keeps the original label values)
        self._object_labels = np.array(regionprops['label'])[keep][order]
        self.segmentation_mask = img_util.repeat_2d(
            labeled, (downscale_factor, downscale_factor)
        )[:shape[0], :shape[1]]
        if plot_segmentation:
            self.plot_segmentation()

    def object_block_mask(self, i, grid_shape=None, threshold=0.25):
        """Boolean mask over a block grid for object `i`, from its segmentation
        label (not its bounding box), so overlapping object bboxes don't collide.

        `grid_shape` defaults to the finest (level1) grid; pass a coarser level's
        `grid_shape` for the multi-res path. Always returns at least one True
        block (falls back to the object's centroid block when the label fills
        less than `threshold` of every block), so masked shift computation never
        produces an all-infinite -- and thus crashing -- level.
        """
        if grid_shape is None:
            grid_shape = self.aligner.grid_shape
        nbi, nbj = grid_shape
        obj = self.segmentation_mask == self._object_labels[i]
        grid = cv2.resize(obj.astype('float32'), (nbj, nbi), interpolation=cv2.INTER_AREA)
        mask = grid >= threshold
        if not mask.any():
            rr, cc = np.where(obj)
            bi = min(int(rr.mean() / obj.shape[0] * nbi), nbi - 1)
            bj = min(int(cc.mean() / obj.shape[1] * nbj), nbj - 1)
            mask = np.zeros((nbi, nbj), dtype=bool)
            mask[bi, bj] = True
        return mask

    def plot_segmentation(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches
        import matplotlib.cm
        colors = matplotlib.cm.Set3.colors
        fig, (ax1, ax2) = plt.subplots(1, 2)
        def _proc_img(img):
            if img_util.is_brightfield_img(img):
                return img
            return np.log1p(img)
        ax1.imshow(_proc_img(self.ref_thumbnail), cmap='gray')
        ax2.imshow(_proc_img(self.moving_thumbnail), cmap='gray')
        bounds = skimage.segmentation.find_boundaries(
            self.segmentation_mask, mode='thick'
        ).astype(float)
        ax1.imshow(np.where(bounds == 0, np.nan, bounds), cmap='cividis', vmin=0, vmax=1, interpolation='none')
        for idx, (rs, re, cs, ce) in enumerate(self.bbox_ref_thumbnail):
            color = colors[idx % len(colors)]
            mpatch = matplotlib.patches.Rectangle((cs, rs), ce-cs, re-rs, fill=False, edgecolor=color)
            ax1.add_patch(mpatch)

            corners = mpatch.get_corners()
            tform = skimage.transform.AffineTransform(self.baseline_coarse_affine_matrix)
            mpathc2 = matplotlib.patches.Polygon(tform.inverse(corners), fill=False, edgecolor=color)
            ax2.add_patch(mpathc2)
        return fig

    def _set_level2(self):
        lv_pairs = align_multi_res.match_levels(self.reader1, self.reader2)
        return lv_pairs[self.level1][1]
    
    def make_aligner(self):
        return align.get_aligner(
            self.reader1, self.reader2,
            level1=self.level1, level2=self.level2,
            channel1=self.channel1, channel2=self.channel2,
            thumbnail_level1=self.thumbnail_level1,
            thumbnail_level2=None,
            thumbnail_channel1=self.thumbnail_channel1,
            thumbnail_channel2=self.thumbnail_channel2,
        )
    
    def align_object(self, i, plot_shifts=True, refine=True, multi_res=True,
                     min_num_blocks=25, **kwargs):
        rs, re, cs, ce = np.array(self.bbox_ref_thumbnail[i]).astype(int)
        rsm, rem, csm, cem = transform_bbox(
            self.bbox_ref_thumbnail, self.baseline_coarse_affine_matrix
        )[i]

        masked_t_ref = np.ones_like(self.ref_thumbnail) * self.fill_value_ref_thumbnail
        masked_t_ref[rs:re, cs:ce] = self.ref_thumbnail[rs:re, cs:ce]

        masked_t_moving = np.ones_like(self.moving_thumbnail) * self.fill_value_moving_thumbnail
        masked_t_moving[rsm:rem, csm:cem] = self.moving_thumbnail[rsm:rem, csm:cem]

        c21l = self.make_aligner()
        c21l.ref_thumbnail = masked_t_ref
        c21l.moving_thumbnail = masked_t_moving
        # use the same coarse method as the CLI's first step
        # (`register_dev.search_then_register`); flip/intensity-invert and the
        # reference-order search are handled internally, so no explicit
        # `test_flip`/`test_intensity_invert` is needed here
        default_kwargs = {
            'n_keypoints': 10_000,
            'plot_match_result': True,
            'auto_mask': True,
        }
        _mx = register_dev.search_then_register(
            np.asarray(masked_t_ref),
            np.asarray(masked_t_moving),
            **{**default_kwargs, **kwargs}
        )
        c21l.coarse_affine_matrix = np.vstack([_mx, [0, 0, 1]])
        if plot_shifts:
            import matplotlib.pyplot as plt
            plt.gcf().suptitle(f"Object {i} (coarse alignment)")

        # per-object block region from the segmentation label (not bbox), so
        # overlapping object bounding boxes don't cross-assign blocks
        block_mask = self.object_block_mask(i)

        if refine:
            refined = align_refine.refine_affine_by_block_translation(
                c21l, block_mask=block_mask, plot=plot_shifts
            )
            if refined is not None:
                c21l.coarse_affine_matrix = refined
                if plot_shifts:
                    import matplotlib.pyplot as plt
                    plt.gcf().suptitle(f"Object {i} (coarse affine refinement)")

        shift_mask = da.from_array(block_mask, chunks=1)
        if multi_res:
            # coarse-to-fine block shifts within this object, using its refined
            # affine as the baseline; the per-level mask follows the object
            mr = align_multi_res.MultiResAligner(
                self.reader1, self.reader2, level1=self.level1,
                channel1=self.channel1, channel2=self.channel2,
                thumbnail_channel1=self.thumbnail_channel1,
                thumbnail_channel2=self.thumbnail_channel2,
                thumbnail_level1=self.thumbnail_level1,
                # match the standalone `multi_res` path so both use the same
                # number of pyramid levels (the class default of 4 would add
                # coarser levels)
                min_num_blocks=min_num_blocks,
            )
            mr._coarse_affine_matrix = c21l.coarse_affine_matrix
            mr.align(mask_fn=lambda gs: self.object_block_mask(i, gs))
            mr.constrain_shifts()
            block_aligner = mr.base_aligner
            # QC: show the multi-res per-level selection (not just the combined
            # field) for this object
            shift_plotter = mr
        else:
            c21l.compute_shifts(mask=shift_mask)
            c21l.constrain_shifts()
            block_aligner = c21l
            shift_plotter = c21l
        if plot_shifts:
            try:
                shift_plotter.plot_shifts()
                import matplotlib.pyplot as plt
                plt.gcf().suptitle(f"Object {i} (block shifts)")
            except Exception as e:
                print(f'\nFailed plotting shifts for object: {i}\n')
                print(e)
        return (block_aligner.shifts, block_aligner.block_affine_matrices, shift_mask)

    def align_all_objects(self, plot_shift=True, refine=True, multi_res=True,
                          min_num_blocks=25):
        block_mxs = []
        shift_masks = []
        for idx, _ in enumerate(self.bbox_ref_thumbnail):
            _, mx, mask = self.align_object(
                idx, plot_shifts=plot_shift, refine=refine, multi_res=multi_res,
                min_num_blocks=min_num_blocks,
            )
            block_mxs.append(mx)
            shift_masks.append(mask)
        self.block_mxs = np.array(block_mxs)
        self.shift_masks = np.array(shift_masks)

    def combine_object_results(self, exclude_objects=None):
        to_include = np.ones(len(self.shift_masks), dtype=bool)
        if exclude_objects is not None:
            for ii in exclude_objects:
                to_include[ii] = False
        assert to_include.sum() > 0
        masks = self.shift_masks[to_include]
        mxs = self.block_mxs[to_include]
        passed = np.argmax(
            masks.reshape(len(masks), -1), axis=0
        )
        mxs_final = np.zeros_like(mxs[0])
        for idx, bb in enumerate(mxs):
            mm = passed == idx
            mxs_final[mm] = bb[mm]
        mxs_final[
            ~masks.reshape(len(masks), -1).max(axis=0)
        ] = self.baseline_affine_matrix
        self.block_affine_matrices_da = align.block_affine_matrices_da(
            mxs_final, self.aligner.grid_shape
        )
