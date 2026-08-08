import itertools
import pathlib
import re
from functools import cached_property

import cv2
import dask.array as da
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform
from loguru import logger

from . import (
    align,
    align_multi_res,
    align_refine,
    img_util,
    register_coarse,
    register_util,
)


def transform_bbox(bbox, affine_mx, shape=None):
    """Map reference-frame bboxes through `affine_mx` into the moving frame.

    `shape` clips the result to the moving image's bounds. Without it a bad
    affine can hand back a box that lies entirely outside the image, whose slice
    is empty -- which silently turns the masked thumbnail into a constant, and
    the object's coarse fit into a registration against a blank image.
    """
    tform_bbox = []
    tform = skimage.transform.AffineTransform(affine_mx)
    hi = (None, None) if shape is None else (shape[0], shape[1])
    for rs, re, cs, ce in bbox:
        xx, yy = tform.inverse(list(itertools.product([cs, ce], [rs, re]))).T
        rs2, cs2 = np.floor([yy.min(), xx.min()]).astype(int)
        re2, ce2 = np.ceil([yy.max(), xx.max()]).astype(int)
        rs2, re2 = np.clip([rs2, re2], 0, hi[0])
        cs2, ce2 = np.clip([cs2, ce2], 0, hi[1])
        tform_bbox.append([rs2, re2, cs2, ce2])
    return tform_bbox


class MultiObjAligner:
    def __init__(
        self,
        reader1,
        reader2,
        level1=0,
        channel1=0,
        channel2=0,
        thumbnail_channel1=None,
        thumbnail_channel2=None,
        thumbnail_level1=-1,
        thumbnails_pixel_size=None,
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
        # per-object QC rows, appended by `align_object`; initialized here so
        # calling `align_object` on its own is not an AttributeError
        self.object_qc = []
        # set by `run`; the warp defaults to it so the two outputs cannot
        # disagree about which objects are excluded
        self.exclude_objects = None
        # set by `run`; None keeps every QC figure open for the caller to do as
        # it likes with (notebooks, `.dev/golden/capture.py`), which is what
        # calling `align_object` on its own should do
        self.qc_dir = None
        self._qc_count = 0

    def run(
        self,
        downscale_factor=8,
        merge_gap=500.0,
        segment=True,
        exclude_objects=None,
        refine=True,
        multi_res=True,
        min_num_blocks=25,
        windowed_coarse=True,
        coarse_kwargs=None,
        plot=True,
        qc_dir=None,
    ):
        # `plot=False` skips every QC figure; `qc_dir` writes each one as soon
        # as it is drawn (see `_finish_new_figs`) instead of leaving it open for
        # the caller to collect, so a run that fails partway still leaves the
        # QC for everything that came before it.
        self.qc_dir = qc_dir
        self.segment_objects(
            downscale_factor=downscale_factor,
            merge_gap=merge_gap,
            segment=segment,
            plot_segmentation=plot,
        )
        self.align_all_objects(
            plot_shift=plot,
            refine=refine,
            multi_res=multi_res,
            min_num_blocks=min_num_blocks,
            windowed_coarse=windowed_coarse,
            coarse_kwargs=coarse_kwargs,
        )
        # remembered so the warp does not have to be told again -- an object
        # excluded from the block-matrix combine but not from
        # `displacement_transformed_moving_img` is warped by its own affine in
        # one output and by the baseline in the other
        self.exclude_objects = exclude_objects
        self.combine_object_results(exclude_objects=exclude_objects)
        logger.info("Alignment QC summary\n" + self.qc_summary(exclude_objects))

    def seed_baseline_coarse(self, coarse_affine_matrix, match_config=None):
        """Seed the baseline (whole-image) coarse affine from outside, instead
        of letting `self.aligner` register it lazily.

        `coarse_affine_matrix` is in the thumbnail frame (2x3 or 3x3, as
        produced by `register_coarse.coarse_register`). The baseline is used for
        object bbox transforms, the background fill in
        `combine_object_results`, and the fallback affine in the multi-object
        displacement warp.

        Pass `match_config` (the seeding fit's `Aligner.coarse_match_config`)
        along with it -- a matrix alone carries no record of the configuration it
        was matched under, and `match_config` would have to search for one again.
        """
        self.aligner.coarse_affine_matrix = coarse_affine_matrix
        self.aligner.coarse_match_config = match_config
        # `bbox_moving_thumbnail` was derived from the previous baseline, and
        # `match_config` from the previous config
        self.__dict__.pop("_bbox_moving_thumbnail", None)
        self.__dict__.pop("match_config", None)

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
    def match_config(self):
        """The intensity/orientation config every object's coarse fit reuses.

        Inherited from the baseline fit -- the config that fit actually committed to,
        which on the windowed route is the winning tile's. The config encodes the
        modality relationship (which image is histogram-matched into which, and
        whether it is intensity-inverted) and whether the scans are mirrored, all
        properties of the slide pair rather than of one tissue piece. Re-searching per
        object costs 8 ORB+RANSAC runs each (x N tiles on the windowed route) on a
        thumbnail that is mostly background fill, where the search's `min_fold_increase`
        test is weak and can settle on a different config than the whole slide did.

        Inheriting rather than searching afresh matters for the same reason. A fresh
        whole-thumbnail search sees mostly background on a small-portion pair, so it
        routinely exhausts `search_best_match_config`'s recursion without any config
        standing clear and returns the argmax of a flat field -- while the baseline,
        on the windowed route, had a tile sitting on real tissue that found one
        decisively. Objects were being pinned to the weaker of the two answers, and
        to one the baseline they perturb never agreed to.

        ponytail: only a seeded baseline (`seed_baseline_coarse` without a config)
        still searches. A piece placed mirrored relative to the rest of the slide
        gets the wrong flip either way, its fit comes back near identity, and
        `_pick_object_affine` drops it back to the baseline affine (the object stays
        on the baseline, visible in its QC panel and scores). Upgrade when seen in
        practice: re-search per object when the pinned-config fit scores weak.
        """
        # reading the baseline runs `Aligner`'s lazy coarse fit if nothing has
        # seeded or triggered it yet, which is what leaves the config behind
        _ = self.baseline_coarse_affine_matrix
        config = self.aligner.coarse_match_config
        if config is not None:
            logger.info(
                f"Pinned coarse match config for all objects:"
                f" {register_coarse.format_config(config)}"
                f" (inherited from the baseline coarse fit)"
            )
            return config
        n_inliers, config = register_coarse.search_best_match_config(
            self.ref_thumbnail, self.moving_thumbnail
        )
        logger.info(
            f"Pinned coarse match config for all objects:"
            f" {register_coarse.format_config(config)} ({n_inliers} inliers;"
            f" searched, as the seeded baseline carried no config)"
        )
        return config

    @staticmethod
    def _background_fill_value(thumbnail):
        """Mean background intensity, used to fill everything outside an object.

        Falls back to the whole-image mean when `entropy_mask` claims the entire
        frame: `np.mean` of an empty selection is NaN, and that NaN would fill
        the masked thumbnail and poison every object's coarse fit rather than
        failing loudly.
        """
        background = thumbnail[~img_util.entropy_mask(thumbnail)]
        if background.size == 0:
            logger.warning(
                "Entropy mask found no background; filling outside each object"
                " with the whole-image mean instead"
            )
            return np.mean(thumbnail)
        return np.mean(background)

    @cached_property
    def fill_value_ref_thumbnail(self):
        return self._background_fill_value(self.ref_thumbnail)

    @cached_property
    def fill_value_moving_thumbnail(self):
        return self._background_fill_value(self.moving_thumbnail)

    # the baseline coarse/full-res affines live on `self.aligner` -- no copy is
    # kept here, and no local coarse defaults either: reading one before
    # `seed_baseline_coarse` falls through to `Aligner`'s lazy registration,
    # which now uses the same keypoint budget this class used to raise on its own
    @property
    def baseline_affine_matrix(self):
        return self.aligner.affine_matrix

    @property
    def baseline_coarse_affine_matrix(self):
        return self.aligner.coarse_affine_matrix

    @property
    def bbox_ref_thumbnail(self):
        if not hasattr(self, "_bbox_ref_thumbnail"):
            # Deliberately not a lazy `segment_objects()`: it would run with the
            # *default* `merge_gap`/`downscale_factor`, quietly ignoring what the
            # caller meant to segment with, and plot as a side effect.
            raise AttributeError(
                "no objects segmented yet; call `segment_objects` (or `run`,"
                " which calls it) first"
            )
        return self._bbox_ref_thumbnail

    @property
    def bbox_moving_thumbnail(self):
        """`bbox_ref_thumbnail` mapped through the baseline coarse affine.

        Computed for all objects at once and cached: `transform_bbox` transforms
        the whole list, so calling it per object was quadratic in object count.
        Invalidated by `segment_objects` (new boxes) and `seed_baseline_coarse`
        (new affine).
        """
        if not hasattr(self, "_bbox_moving_thumbnail"):
            self._bbox_moving_thumbnail = transform_bbox(
                self.bbox_ref_thumbnail,
                self.baseline_coarse_affine_matrix,
                shape=self.moving_thumbnail.shape,
            )
        return self._bbox_moving_thumbnail

    # a typical WSI is ~2 cm across, so the default 500 µm merge gap is ~2.5% of
    # the image width; used as the fallback when the physical scale is unknown
    MERGE_GAP_IMAGE_FRACTION = 0.025

    def _merge_radius(self, merge_gap, downscale_factor, mask_shape):
        """Closing radius (in `mask` pixels) that bridges a `merge_gap` µm gap.

        Falls back to a fraction of the image width when `reader1` has no real
        pixel size: the placeholder 1 µm would make the radius wrong by the true
        pixel size (3x too small on a 0.325 µm scan), and under-merging is the
        dangerous direction -- it splits one piece into several, which is a hard
        visible seam, where over-merging only gives the shift field more work.
        """
        if not self.reader1.has_pixel_size:
            gap_px = self.MERGE_GAP_IMAGE_FRACTION * max(mask_shape)
            logger.warning(
                f"{self.reader1.source_name} has no pixel size metadata; merging"
                f" tissue gaps up to {self.MERGE_GAP_IMAGE_FRACTION:.1%} of the"
                f" image ({2 * 0.5 * gap_px:.0f} px) instead of {merge_gap:g} µm."
                f" Pass `px_size1` for a physical merge gap"
            )
            return int(round(0.5 * gap_px))
        small_px_um = (
            self.reader1.pixel_size
            * self.reader1.level_downsamples[self.level1]
            * self.aligner.ref_thumbnail_down_factor
            * downscale_factor
        )
        return int(round(0.5 * merge_gap / small_px_um))

    def segment_objects(
        self,
        downscale_factor=8,
        min_area=None,
        merge_gap=500.0,
        segment=True,
        plot_segmentation=False,
    ):
        shape = self.ref_thumbnail.shape
        mask = img_util.entropy_mask(
            img_util.cv2_downscale_local_mean(self.ref_thumbnail, downscale_factor)
        )
        if not segment:
            # single global object: all tissue is one label (no splitting). It
            # still runs the same per-object coarse + multi-res block-shift path,
            # so "one object" is exactly the classic single-object multi-res
            # alignment (N=1 degenerate case of the multi-object orchestrator).
            labeled = mask.astype("int32")
        else:
            # Conservative merge (err toward one object): bridge gaps up to
            # `merge_gap` microns before labeling, so a single torn/folded piece
            # -- or one with a low-entropy interior (fat, lumen) -- stays one
            # object instead of splitting into several. Only genuinely
            # well-separated pieces survive as distinct objects; within-piece
            # motion is the block-shift field's job, not the segmenter's. A
            # wrong split is a hard, visible seam, whereas a missed split just
            # means the shift field does slightly more work -- so we lean toward
            # merging. `merge_gap` is in microns (resolution-independent); set 0
            # to disable. Closing with a radius-r disk fills gaps up to ~2*r.
            if merge_gap and merge_gap > 0:
                r = self._merge_radius(merge_gap, downscale_factor, mask.shape)
                if r >= 1:
                    mask = skimage.morphology.binary_closing(
                        mask, skimage.morphology.disk(r)
                    )
            labeled = skimage.measure.label(mask)
            labeled = skimage.segmentation.expand_labels(labeled, 4)

        regionprops = skimage.measure.regionprops_table(
            labeled, properties=["label", "bbox", "area"]
        )
        area = np.array(regionprops["area"])
        # drop specks/debris; default threshold is 1% of the largest object
        if min_area is None:
            min_area = 0.01 * area.max() if area.size else 0
        keep = area >= min_area
        order = np.argsort(area[keep])[::-1]  # largest object first
        bbox_ref_thumbnail = (
            downscale_factor
            * np.array(
                [
                    regionprops["bbox-0"],
                    regionprops["bbox-2"],
                    regionprops["bbox-1"],
                    regionprops["bbox-3"],
                ]
            ).T
        )
        self._bbox_ref_thumbnail = bbox_ref_thumbnail[keep][order]
        # derived from the boxes above; re-segmenting invalidates it
        self.__dict__.pop("_bbox_moving_thumbnail", None)
        # label value of each (sorted) object, so per-object masks can be read
        # back from `segmentation_mask` (which keeps the original label values)
        self._object_labels = np.array(regionprops["label"])[keep][order]
        self.segmentation_mask = img_util.repeat_2d(
            labeled, (downscale_factor, downscale_factor)
        )[: shape[0], : shape[1]]
        if plot_segmentation:
            # Take the baseline affine first, under its own name. Reading it
            # registers it lazily when nobody called `seed_baseline_coarse`, and
            # that registration draws a match figure of its own -- inside
            # `plot_segmentation`, where the bracket below would file it as a
            # second "object segmentation". Already seeded (every CLI run), this
            # finds nothing and writes nothing.
            figs_before = self._fignums()
            _ = self.baseline_coarse_affine_matrix
            self._finish_new_figs(figs_before, "baseline coarse alignment")
            # bracketed like every other plotting call so it is written and
            # closed on the same path, even though it hands back its figure
            figs_before = self._fignums()
            self.plot_segmentation()
            self._finish_new_figs(figs_before, "object segmentation")

    def object_block_mask(self, i, grid_shape=None, threshold=1.0 / 16):
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
        grid = cv2.resize(
            obj.astype("float32"), (nbj, nbi), interpolation=cv2.INTER_AREA
        )
        mask = grid >= threshold
        if not mask.any():
            rr, cc = np.where(obj)
            bi = min(int(rr.mean() / obj.shape[0] * nbi), nbi - 1)
            bj = min(int(cc.mean() / obj.shape[1] * nbj), nbj - 1)
            mask = np.zeros((nbi, nbj), dtype=bool)
            mask[bi, bj] = True
        return mask

    def plot_segmentation(self):
        import matplotlib.cm
        import matplotlib.patches
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt

        colors = matplotlib.cm.Set3.colors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("object segmentation")

        def _proc_img(img):
            if img_util.is_brightfield_img(img):
                return img
            return np.log1p(img)

        ax1.imshow(_proc_img(self.ref_thumbnail), cmap="gray")
        ax2.imshow(_proc_img(self.moving_thumbnail), cmap="gray")
        ax1.set_title("reference (object outline + bbox)", fontsize=8)
        ax2.set_title(
            "moving (bbox through the baseline coarse affine;\n"
            "dashed = the axis-aligned crop taken from it)",
            fontsize=8,
        )
        tform = skimage.transform.AffineTransform(self.baseline_coarse_affine_matrix)
        for idx, (label, (rs, re, cs, ce), (rs2, re2, cs2, ce2)) in enumerate(
            zip(
                self._object_labels,
                self.bbox_ref_thumbnail,
                self.bbox_moving_thumbnail,
            )
        ):
            color = colors[idx % len(colors)]
            # a contour, not `find_boundaries`: a raster outline at thumbnail
            # resolution aliases into dashes once the figure is rendered small,
            # and one shared color cannot say which object a boundary belongs to
            ax1.contour(
                self.segmentation_mask == label,
                levels=[0.5],
                colors=[color],
                linewidths=0.8,
            )
            mpatch = matplotlib.patches.Rectangle(
                (cs, rs), ce - cs, re - rs, fill=False, edgecolor=color
            )
            ax1.add_patch(mpatch)
            # bboxes of neighbouring pieces overlap heavily on a diagonal slide,
            # so label each one with the index the QC summary and the logs use
            for ax, (x, y) in [(ax1, (cs, rs)), (ax2, (cs2, rs2))]:
                ax.text(
                    x,
                    y,
                    f" {idx}",
                    color=color,
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                )

            corners = mpatch.get_corners()
            mpathc2 = matplotlib.patches.Polygon(
                tform.inverse(corners), fill=False, edgecolor=color
            )
            ax2.add_patch(mpathc2)
            # what is actually cropped for the object's coarse fit: the mapped
            # box is a parallelogram, the crop is its axis-aligned hull clipped
            # to the image, and a bad affine shows up as the two disagreeing
            ax2.add_patch(
                matplotlib.patches.Rectangle(
                    (cs2, rs2),
                    ce2 - cs2,
                    re2 - rs2,
                    fill=False,
                    edgecolor=color,
                    linestyle="--",
                    linewidth=0.8,
                )
            )
        for ax, img in [(ax1, self.ref_thumbnail), (ax2, self.moving_thumbnail)]:
            # patches drawn outside the image must not rescale the panel
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.set_axis_off()
            # the two thumbnails have different aspect ratios; hang both from
            # the top so each stays under its own title
            ax.set_anchor("N")
        return fig

    @property
    def level2(self):
        """The reader2 level the object affines were fit at.

        Picked by `get_aligner` from physical pixel size; exposed because a
        caller warping a whole pyramid level with these affines must read
        `reader2.pyramid[level2]` and not assume level 0.
        """
        return self.aligner.level2

    def make_aligner(self):
        return align.get_aligner(
            self.reader1,
            self.reader2,
            level1=self.level1,
            channel1=self.channel1,
            channel2=self.channel2,
            thumbnail_level1=self.thumbnail_level1,
            thumbnail_level2=None,
            thumbnails_pixel_size=self.thumbnails_pixel_size,
            thumbnail_channel1=self.thumbnail_channel1,
            thumbnail_channel2=self.thumbnail_channel2,
        )

    # Fall back only when the preferred affine's overlap score has collapsed to
    # this fraction of the best candidate's.
    #
    # `score_overlap` is a gross-failure detector, not a fine discriminator. It
    # is the same cross-modality confidence `coarse_register` uses to decide
    # whether an affine is plausible at all, and cross-modality it sits around
    # 0.25-0.35, where its differences are structural rather than positional.
    # Measured on the two-object pair: a refinement that is 0.46px accurate
    # scored 0.254 while the unrefined fit at 20.87px scored 0.308 -- ranking
    # near-neighbours by this score gets them backwards. A gross failure looks
    # nothing like that: on the melanoma pair a per-object fit that landed
    # nowhere scored 0.003 against 0.241. Only that collapse is detectable.
    FALLBACK_SCORE_RATIO = 0.5

    @classmethod
    def _choose_by_score(cls, scores, preferred):
        """Keep `preferred` unless its score has collapsed against the best.

        Kept separate from the measuring so the rule can be pinned against the
        scores actually observed, rather than against whatever a synthetic image
        happens to produce -- the metric's near-neighbour behaviour depends on
        the content and does not survive being mocked up.
        """
        best = max(scores, key=scores.get)
        if scores[preferred] < cls.FALLBACK_SCORE_RATIO * scores[best]:
            return best
        return preferred

    def _pick_object_affine(self, i, candidates, ref_crop, moving, to_crop=None):
        """Choose among coarse affine candidates by overlap score.

        `align_object` used to accept its per-object fit unconditionally. That
        fit is a fresh feature match on masked thumbnails, so it can land well
        off while the whole-image baseline -- which it is a perturbation of --
        was fine. One bad object is a hard visible seam, so the fit has to beat
        the fallback rather than merely exist.

        The comparison is deliberately blunt: keep the most refined candidate
        unless its score has collapsed (see `FALLBACK_SCORE_RATIO`). The score
        cannot rank near-neighbours, so a close contest is not evidence and must
        not move the choice -- only a fit that landed nowhere is detectable.

        Score on the object's *crop* of the real thumbnails, not on the masked
        ones the coarse fit was made from: masking fills everything outside the
        bbox with a constant, and `score_overlap` whitens, so the synthetic
        rectangle edge would carry weight that the tissue should.

        `to_crop` maps the reference thumbnail frame (which the candidates are
        in) into `ref_crop`'s frame.
        """
        if to_crop is None:
            to_crop = np.eye(3)
        scores = {
            name: register_util.score_overlap(ref_crop, moving, to_crop @ mx)
            for name, mx in candidates
        }
        preferred = candidates[-1][0]
        chosen = self._choose_by_score(scores, preferred)
        if chosen != preferred:
            logger.warning(
                f"Object {i}: '{preferred}' coarse affine scores"
                f" {scores[preferred]:.3f} against {scores[chosen]:.3f} for"
                f" '{chosen}'; falling back to '{chosen}'"
            )
        logger.info(
            f"Object {i}: coarse affine '{chosen}' ("
            + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
            + ")"
        )
        return chosen, scores

    def make_multi_res_aligner(self, multi_res=True, min_num_blocks=25):
        """The resolution ladder every object's block shifts are measured on.

        Nothing about it is per-object -- the object enters only through the
        masked thumbnails, the coarse affine and the block mask, all of which
        `align_object` reassigns on the aligner it is handed. So
        `align_all_objects` builds one and passes it to every object.

        That matters because `MultiResAligner.__init__` persists the ladder,
        which walks the whole moving image once (on the melanoma pair, ~40s to
        decode a 10.8 GB level-0 SVS). Per object, that is the single most
        expensive thing in a multi-piece run.

        `multi_res=False` is this same aligner truncated to its base rung rather
        than a separate code path: `MultiResAligner`'s base rung is the very
        `get_aligner` call `make_aligner()` makes, argument for argument, and an
        unreachable `min_num_blocks` stops the ladder there. Routing it through
        here also gets it `MultiResAligner.constrain_shifts`, which normalizes
        non-finite residuals to 0 -- `Aligner.constrain_shifts` alone leaves
        `inf` at outside-object blocks whenever `constrain_block_shifts` takes
        one of its degenerate early returns, and that `inf` reaches
        `displacement_transformed_moving_img` as `inf * 0 = NaN` under
        smoothing, or as a poisoned `cv2.remap` without it.
        """
        mr = align_multi_res.MultiResAligner(
            self.reader1,
            self.reader2,
            level1=self.level1,
            channel1=self.channel1,
            channel2=self.channel2,
            thumbnail_channel1=self.thumbnail_channel1,
            thumbnail_channel2=self.thumbnail_channel2,
            thumbnail_level1=self.thumbnail_level1,
            thumbnails_pixel_size=self.thumbnails_pixel_size,
            # match the standalone `multi_res` path so both use the same number
            # of rungs (the class default of 4 would add coarser ones)
            min_num_blocks=min_num_blocks if multi_res else np.inf,
        )
        # `block_mask`, `shift_mask` and everything `combine_object_results`
        # does are on `self.aligner`'s grid; `MultiResAligner` keeps level1
        # unconditionally, so its finest rung is the same grid. Pin it -- a
        # mismatch would surface far away, as an IndexError when `block_mask`
        # indexes `shifts` in `align_object`.
        assert mr.levels[0] == self.level1, (
            f"multi-res finest level {mr.levels[0]} != level1 {self.level1}"
        )
        return mr

    def align_object(
        self,
        i,
        plot_shifts=True,
        refine=True,
        multi_res=True,
        min_num_blocks=25,
        windowed_coarse=True,
        coarse_kwargs=None,
        mr=None,
    ):
        rs, re, cs, ce = np.array(self.bbox_ref_thumbnail[i]).astype(int)
        rsm, rem, csm, cem = self.bbox_moving_thumbnail[i]

        masked_t_ref = np.ones_like(self.ref_thumbnail) * self.fill_value_ref_thumbnail
        masked_t_ref[rs:re, cs:ce] = self.ref_thumbnail[rs:re, cs:ce]

        masked_t_moving = (
            np.ones_like(self.moving_thumbnail) * self.fill_value_moving_thumbnail
        )
        masked_t_moving[rsm:rem, csm:cem] = self.moving_thumbnail[rsm:rem, csm:cem]

        # Built up front so its finest rung can *be* `c21l`: it is at `level1`
        # with the same thumbnails, so a separate `make_aligner()` only
        # duplicated it -- and that duplicate cost a full thumbnail build per
        # object. `align_all_objects` hands the same ladder to every object (see
        # `make_multi_res_aligner`); building one here keeps this method
        # callable on its own.
        if mr is None:
            mr = self.make_multi_res_aligner(
                multi_res=multi_res, min_num_blocks=min_num_blocks
            )
        c21l = mr.aligners[0]
        # the coarse fit and the refinement both read these; the coarser levels
        # of `mr` never touch a thumbnail, so masking only the finest is enough.
        # Assigned per object, so a shared ladder never carries the previous
        # object's mask into this one's coarse fit.
        c21l.ref_thumbnail = masked_t_ref
        c21l.moving_thumbnail = masked_t_moving

        # candidates in increasing order of preference; the whole-image baseline
        # is the known-good fallback this object's own fit has to beat
        candidates = [
            ("baseline", np.asarray(self.baseline_coarse_affine_matrix)),
        ]
        # An empty moving crop means the baseline affine puts this object off the
        # moving image entirely, so `masked_t_moving` is a constant -- registering
        # against it is a guaranteed-useless feature match, and `score_overlap`
        # would reject its result anyway. Skip straight to the baseline.
        if rem <= rsm or cem <= csm:
            logger.warning(
                f"Object {i}: the baseline affine maps its bbox outside the"
                f" moving thumbnail; keeping the baseline coarse affine"
            )
            c21l.coarse_affine_matrix = self.baseline_coarse_affine_matrix
        else:
            # flip/intensity-invert and the reference-order search are handled
            # inside the engine, so no explicit `test_flip`/`test_intensity_invert`
            default_kwargs = {
                # follows the caller's plotting choice rather than forcing it on
                "plot_match_result": plot_shifts,
                # searched once for the slide pair, not per object -- see `match_config`
                "config": self.match_config,
            }
            coarse_kwargs = {**default_kwargs, **(coarse_kwargs or {})}
            figs_before = self._fignums()
            if windowed_coarse:
                # `coarse_register` adds a windowed retry when the whole-image match
                # comes back weak, which is an object's only chance of recovering
                # from a failed fit -- `search_then_register` just returns identity.
                # `matched_area_ratio=1.0` skips its physical-footprint test: both
                # masked thumbnails are full-size (the object's bbox is filled in and
                # the rest is background), so the test would compare the whole slides
                # and never see the small portion the object actually is.
                _mx = register_coarse.coarse_register(
                    np.asarray(masked_t_ref),
                    np.asarray(masked_t_moving),
                    matched_area_ratio=1.0,
                    **coarse_kwargs,
                )
            else:
                # no windowed tile search on this route, so nothing to parallelize
                coarse_kwargs.pop("n_workers", None)
                # ponytail: an object whose whole-image fit lands nowhere has no
                # second chance here -- `search_then_register` returns identity and
                # `_pick_object_affine` drops it to the baseline, so the object is
                # never better than the whole-slide affine. Observed on LSP74545
                # (2026-08-07): every object fit came back at 14 matches / score
                # 0.000 and fell back. The retry exists one branch up; the reason
                # it is not the default is cost -- N tiles x 8 configs per object,
                # on thumbnails that are mostly background fill. Upgrade path:
                # take the `windowed_coarse=True` branch per object when the
                # whole-image fit scores below `FALLBACK_SCORE_RATIO` of the
                # baseline, so the retry costs only the objects that need it,
                # rather than making `windowed_coarse` an all-or-nothing flag.
                _mx = register_coarse.search_then_register(
                    np.asarray(masked_t_ref),
                    np.asarray(masked_t_moving),
                    **coarse_kwargs,
                )
            c21l.coarse_affine_matrix = _mx
            self._finish_new_figs(figs_before, f"Object {i} (coarse alignment)")
            candidates.append(("object", c21l.coarse_affine_matrix))

        # per-object block region from the segmentation label (not bbox), so
        # overlapping object bounding boxes don't cross-assign blocks
        block_mask = self.object_block_mask(i)

        refine_stats = None
        if refine:
            figs_before = self._fignums()
            refined, refine_stats = align_refine.refine_affine_by_block_translation(
                c21l, block_mask=block_mask, plot=plot_shifts
            )
            self._finish_new_figs(
                figs_before, f"Object {i} (coarse affine refinement)"
            )
            if refined is not None:
                c21l.coarse_affine_matrix = refined
                candidates.append(("object+refine", c21l.coarse_affine_matrix))

        chosen, scores = self._pick_object_affine(
            i,
            candidates,
            self.ref_thumbnail[rs:re, cs:ce],
            self.moving_thumbnail,
            register_util.translate_mx(-cs, -rs),
        )
        c21l.coarse_affine_matrix = dict(candidates)[chosen]

        shift_mask = da.from_array(block_mask, chunks=1)
        # coarse-to-fine block shifts within this object, using its refined
        # affine as the baseline; the per-rung mask follows the object. `c21l` is
        # `mr.aligners[0]`, so this fans the chosen affine out to the coarser
        # rungs (the setter's job, not a re-assignment).
        mr.coarse_affine_matrix = c21l.coarse_affine_matrix
        mr.align(mask_fn=lambda gs: self.object_block_mask(i, gs))
        mr.constrain_shifts()
        # the finest rung carries this object's affine; the shifts are the
        # cross-rung pick made by `constrain_shifts`
        affine_matrix, shifts = mr.aligners[0].affine_matrix, mr.shifts
        # QC: show the per-rung selection, not just the combined field
        shift_plotter = mr
        plot_failed = False
        if plot_shifts:
            figs_before = self._fignums()
            try:
                shift_plotter.plot_shifts()
            except Exception as e:
                plot_failed = True
                logger.warning(f"Failed plotting shifts for object {i}: {e}")
            finally:
                # in `finally` so a call that raised partway still hands over
                # what it drew: with nothing sweeping the open figures at the
                # end of the run any more, a figure left behind here is one
                # nothing will ever write or close
                self._finish_new_figs(figs_before, f"Object {i} (block shifts)")

        in_object = np.asarray(block_mask).ravel()
        magnitudes = np.linalg.norm(np.asarray(shifts)[in_object], axis=1)
        self.object_qc.append(
            {
                "object": i,
                "label": int(self._object_labels[i]),
                "n_blocks": int(in_object.sum()),
                "affine": chosen,
                # what the object would have used absent a score collapse; recorded
                # rather than re-derived, since the candidate list is not fixed (an
                # object whose bbox lands off the moving image never gets an
                # "object" candidate at all)
                "preferred": candidates[-1][0],
                "scores": scores,
                "refine": refine_stats,
                "shift_median": float(np.median(magnitudes))
                if magnitudes.size
                else None,
                "shift_max": float(magnitudes.max()) if magnitudes.size else None,
                "plot_failed": plot_failed,
            }
        )
        return (
            affine_matrix,
            shifts,
            align.block_affine_matrices(affine_matrix, shifts),
            shift_mask,
        )

    def align_all_objects(
        self,
        plot_shift=True,
        refine=True,
        multi_res=True,
        min_num_blocks=25,
        windowed_coarse=True,
        coarse_kwargs=None,
    ):
        # `coarse_kwargs` reaches each object's coarse call -- notably
        # `n_workers`, which parallelizes the windowed retry's tile search and
        # otherwise never leaves the whole-slide baseline call
        block_mxs = []
        shift_masks = []
        object_affines = []
        object_shifts = []
        self.object_qc = []
        # one ladder for the whole slide, not one per piece -- building it
        # persists a coarsened copy of the moving image, which is the most
        # expensive step in the run
        mr = self.make_multi_res_aligner(
            multi_res=multi_res, min_num_blocks=min_num_blocks
        )
        for idx, _ in enumerate(self.bbox_ref_thumbnail):
            affine, shifts, mx, mask = self.align_object(
                idx,
                mr=mr,
                plot_shifts=plot_shift,
                refine=refine,
                multi_res=multi_res,
                min_num_blocks=min_num_blocks,
                windowed_coarse=windowed_coarse,
                # passed as a dict, not splatted: splatting let a coarse kwarg
                # named `refine`/`multi_res`/... bind to `align_object`'s own
                # parameter and never reach the registration
                coarse_kwargs=coarse_kwargs,
            )
            object_affines.append(affine)
            object_shifts.append(shifts)
            block_mxs.append(mx)
            shift_masks.append(mask)
        self.block_mxs = np.array(block_mxs)
        self.shift_masks = np.array(shift_masks)
        # per-object (global affine, per-block shift field) kept separately so
        # the displacement-field warp can build one continuous field per object
        self.object_affines = np.array(object_affines)
        self.object_shifts = np.array(object_shifts)

    @staticmethod
    def _fignums():
        import matplotlib.pyplot as plt

        return tuple(plt.get_fignums())

    def _finish_new_figs(self, before, title):
        """Title whatever figures a plotting call just created, and -- when a
        `qc_dir` is set -- write and close them right there.

        Writing per figure, rather than sweeping the open figures once at the
        end of the run, is what makes the QC survive a failure: a crash on
        object 5 leaves objects 0-4 *and* object 5's earlier stages on disk.
        It also bounds memory at one figure instead of ~3 per object, and drops
        the need to route the name through the figure's title and read it back
        out to build a filename.

        The figures come from a fignum diff rather than a return value because
        a plotting call may draw one figure, several, or none, and callers deep
        in `register_coarse` discard a losing route's figure before returning.
        `plt.gcf()` would *create* a figure when none is open, yielding a blank
        one carrying a real title (or stamping it onto an unrelated figure that
        happened to be current).
        """
        import matplotlib.pyplot as plt

        new = [n for n in plt.get_fignums() if n not in before]
        for num in new:
            fig = plt.figure(num)
            fig.suptitle(title)
            if self.qc_dir is not None:
                self._write_fig(fig, title)
                plt.close(fig)
        return bool(new)

    # QC figures are written at 144 dpi, which is legible for a whole-slide
    # thumbnail without the 2-3x file size of the print-oriented default
    QC_DPI = 144

    def _write_fig(self, fig, title):
        """Write one QC figure into `qc_dir`, numbered in creation order.

        The number is a plain counter, not the matplotlib figure number, which
        restarts at 1 whenever the open figures all close and so cannot order a
        run. The name comes from the caller's title -- several figures can share
        one (a plotting call that drew two), and the counter keeps those apart.
        """
        qc_dir = pathlib.Path(self.qc_dir)
        qc_dir.mkdir(exist_ok=True, parents=True)
        self._qc_count += 1
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        path = qc_dir / f"{self._qc_count:02d}-{slug}.png"
        fig.savefig(path, dpi=self.QC_DPI, bbox_inches="tight")
        logger.debug(f"Wrote QC figure {path}")

    @staticmethod
    def _format_refine(stats):
        if stats is None:
            return "not run"
        if stats["accepted"]:
            return (
                f"{stats['n_inliers']}/{stats['n_correspondences']} inliers,"
                f" {stats['residual']:.2f}px"
            )
        return f"rejected ({stats['reason']})"

    def qc_summary(self, exclude_objects=None):
        """One table saying what each object's alignment actually did.

        Every decision here is already logged, but as single lines among
        thousands: a rejected refinement, a per-object coarse that lost to the
        baseline, a shift plot that raised. Those are exactly the outcomes worth
        seeing, and a run that ends without a verdict is not hands-off, only
        quiet.
        """
        qc = getattr(self, "object_qc", None)
        if not qc:
            return "  (no objects aligned)"
        excluded = set(exclude_objects or [])
        head = (
            f"  {'obj':>3} {'label':>5} {'blocks':>6}  {'affine':<14}"
            f" {'score':>5}  {'refinement':<30} {'shift med/max':>13}  notes"
        )
        lines = [head, "  " + "-" * (len(head) - 2)]
        n_fallback = n_rejected = 0
        for r in qc:
            # `align_object` records what it actually preferred; fall back to
            # the candidate order for rows built by hand (the self-checks)
            preferred = r.get("preferred") or next(
                name
                for name in ("object+refine", "object", "baseline")
                if name in r["scores"]
            )
            notes = []
            if r["affine"] != preferred:
                notes.append(f"FELL BACK to {r['affine']}")
                n_fallback += 1
            if r["refine"] is not None and not r["refine"]["accepted"]:
                n_rejected += 1
            if r["plot_failed"]:
                notes.append("shift plot failed")
            if r["object"] in excluded:
                notes.append("EXCLUDED from output")
            shift = "-"
            if r["shift_median"] is not None:
                shift = f"{r['shift_median']:.1f} / {r['shift_max']:.1f}"
            score = r["scores"].get(r["affine"], float("nan"))
            lines.append(
                f"  {r['object']:>3} {r['label']:>5} {r['n_blocks']:>6} "
                f" {r['affine']:<14} {score:>5.3f}  {self._format_refine(r['refine']):<30}"
                f" {shift:>13}  {', '.join(notes)}"
            )
        tail = [
            f"  {len(qc)} object(s); {n_fallback} fell back to a lower-ranked"
            f" affine, {n_rejected} refinement(s) rejected"
        ]
        if n_fallback or n_rejected:
            tail.append("  ^ check the per-object QC figures for these")
        return "\n".join(lines + tail)

    def combine_object_results(self, exclude_objects=None):
        to_include = np.ones(len(self.shift_masks), dtype=bool)
        if exclude_objects is not None:
            for ii in exclude_objects:
                to_include[ii] = False
        if not to_include.any():
            raise ValueError(
                f"`exclude_objects={sorted(exclude_objects)}` excludes all"
                f" {len(self.shift_masks)} aligned object(s); at least one must"
                f" remain"
            )
        masks = self.shift_masks[to_include]
        mxs = self.block_mxs[to_include]
        passed = np.argmax(masks.reshape(len(masks), -1), axis=0)
        mxs_final = np.zeros_like(mxs[0])
        for idx, bb in enumerate(mxs):
            mm = passed == idx
            mxs_final[mm] = bb[mm]
        mxs_final[~masks.reshape(len(masks), -1).max(axis=0)] = (
            self.baseline_affine_matrix
        )
        self.block_affine_matrices_da = align.block_affine_matrices_da(
            mxs_final, self.aligner.grid_shape
        )

    def displacement_transformed_moving_img(
        self,
        moving_img,
        sigma_blocks=0.0,
        field_order=1,
        is_mask=False,
        cval=0.0,
        exclude_objects=None,
        interpolation="skimage",
    ):
        """Seam-free, mask-constrained multi-object warp.

        Each object is warped by its own affine plus a continuous displacement
        field (its per-block shifts interpolated and smoothed *within the
        object's mask* via normalized convolution, so the smoothing never
        bleeds across the object boundary). The labeled `segmentation_mask`
        assigns every output pixel to exactly one object at full resolution, so
        intra-object block cracks disappear while genuine inter-object
        discontinuities are preserved. Background (and excluded objects) fall
        back to the baseline affine.

        `sigma_blocks` controls how gradually each object's displacement blends
        between its blocks (in block units); see
        `align.block_displacement_transformed_moving_img`.

        `exclude_objects` defaults to whatever `run` was given, so the warp and
        `block_affine_matrices_da` cannot disagree about which objects are in.
        Pass a value only to override that for one call.
        """
        import scipy.ndimage as ndi

        if exclude_objects is None:
            exclude_objects = self.exclude_objects

        c21l = self.aligner
        ref_img = c21l.ref_img
        out_shape = tuple(int(s) for s in ref_img.shape[-2:])
        cy, cx = ref_img.chunksize[-2:]
        grid_shape = c21l.grid_shape
        # output (level1) pixels per labeled-mask (ref-thumbnail) pixel
        mask_scale = float(c21l.ref_thumbnail_down_factor)

        exclude = set(exclude_objects or [])
        base_inv_affine = np.linalg.inv(
            np.asarray(self.baseline_affine_matrix, dtype="float64")
        )

        label_to_obj = {}
        for i, label in enumerate(self._object_labels):
            if i in exclude:
                continue
            a_inv = np.linalg.inv(np.asarray(self.object_affines[i], dtype="float64"))
            # displacement = -shift (see block_affine_matrices convention)
            d = -np.asarray(self.object_shifts[i], dtype="float32").reshape(
                *grid_shape, 2
            )
            # Outside-object blocks never held a measurement: `_pc` returns inf
            # there, and `constrain_block_shifts` either extrapolates them from
            # the object's trend or -- on one of its degenerate early returns --
            # leaves the inf for `MultiResAligner.constrain_shifts` to normalize
            # to 0. That 0 is indistinguishable from a measured "no displacement
            # here", and `_sample_displacement` samples one cell *outside* the
            # object when it bilinearly interpolates the rim, so it would pull
            # the rim toward zero.
            #
            # Fill those cells from the nearest in-object block instead. The rim
            # then interpolates against a continuation of the object's own
            # field, and no value that merely means "not measured" survives to
            # be read as data. In-object cells map to themselves, so this is an
            # identity there.
            #
            # Note this cannot be folded into the normalized convolution below:
            # at the default `sigma_blocks=0` a gaussian filter is an identity,
            # so that branch would leave `d` exactly as it found it.
            in_object = self.object_block_mask(i, grid_shape)
            if in_object.any():
                _, (fill_r, fill_c) = ndi.distance_transform_edt(
                    ~in_object, return_indices=True
                )
                d = d[fill_r, fill_c]
            if sigma_blocks and sigma_blocks > 0:
                # normalized convolution: smooth only over the object's own
                # blocks so the field isn't pulled toward neighbours/background
                weight = in_object.astype("float32")
                num = ndi.gaussian_filter(
                    d * weight[..., None],
                    sigma=(sigma_blocks, sigma_blocks, 0),
                    mode="constant",
                )
                den = ndi.gaussian_filter(
                    weight, sigma=(sigma_blocks, sigma_blocks), mode="constant"
                )
                valid = den > 1e-3
                d = np.where(
                    valid[..., None], num / np.maximum(den[..., None], 1e-6), d
                )
            label_to_obj[int(label)] = (a_inv, np.ascontiguousarray(d))

        field_interp = cv2.INTER_LINEAR if field_order == 1 else cv2.INTER_CUBIC
        order = 0 if is_mask else 1
        label_mask = np.ascontiguousarray(self.segmentation_mask)

        return_slice = slice(None)
        mimg = moving_img
        if img_util.is_single_channel(mimg) and mimg.ndim == 2:
            mimg = mimg[np.newaxis]
            return_slice = 0

        template = da.zeros(out_shape, dtype="uint8", chunks=(cy, cx))
        warped = [
            template.map_blocks(
                align._multiobj_displacement_block,
                src_array=c,
                label_to_obj=label_to_obj,
                base_inv_affine=base_inv_affine,
                label_mask=label_mask,
                mask_scale=mask_scale,
                out_shape=out_shape,
                cval=float(cval),
                module=interpolation,
                order=order,
                field_interp=field_interp,
                dtype=mimg.dtype,
            )
            for c in mimg
        ]
        return da.array(warped)[return_slice]
