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
    shift_domains,
)

# Largest neighbour-to-neighbour shift difference still counted as one domain.
# Sits in the gap the reference set shows: within a domain neighbours differ by
# a few px of local deformation, between domains by 50-150px. Kept above the
# level-quantisation floor -- a block resolved at rung 3 carries its shift in
# 8px level-0 steps, so two neighbours resolved at different rungs can differ
# by that much without disagreeing about anything.
DEFAULT_DOMAIN_TOL = shift_domains.DEFAULT_TOL


def transform_bbox(bbox, affine_mx, shape=None):
    """Map one reference-frame bbox through `affine_mx` into the moving frame.

    `shape` clips the result to the moving image's bounds. Without it a bad
    affine can hand back a box that lies entirely outside the image, whose slice
    is empty -- which silently turns the masked thumbnail into a constant, and
    the object's coarse fit into a registration against a blank image.
    """
    rs, re, cs, ce = bbox
    tform = skimage.transform.AffineTransform(affine_mx)
    hi = (None, None) if shape is None else (shape[0], shape[1])
    xx, yy = tform.inverse(list(itertools.product([cs, ce], [rs, re]))).T
    rs2, cs2 = np.floor([yy.min(), xx.min()]).astype(int)
    re2, ce2 = np.ceil([yy.max(), xx.max()]).astype(int)
    rs2, re2 = np.clip([rs2, re2], 0, hi[0])
    cs2, ce2 = np.clip([cs2, ce2], 0, hi[1])
    return [rs2, re2, cs2, ce2]


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
        # QC record, set by `align_tissue`; initialized here so calling
        # `qc_summary` before it is not an AttributeError
        self.qc = None
        # set by `run`; None keeps every QC figure open for the caller to do as
        # it likes with (notebooks, `.dev/golden/capture.py`), which is what
        # calling `align_tissue` on its own should do
        self.qc_dir = None
        self._qc_count = 0

    def run(
        self,
        downscale_factor=8,
        merge_gap=500.0,
        segment=True,
        refine=True,
        min_num_blocks=25,
        windowed_coarse=True,
        coarse_kwargs=None,
        domain_tol=DEFAULT_DOMAIN_TOL,
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
        self.align_tissue(
            plot_shifts=plot,
            refine=refine,
            min_num_blocks=min_num_blocks,
            windowed_coarse=windowed_coarse,
            coarse_kwargs=coarse_kwargs,
            domain_tol=domain_tol,
        )
        logger.info("Alignment QC summary\n" + self.qc_summary())

    def seed_baseline_coarse(self, coarse_affine_matrix, match_config=None):
        """Seed the baseline (whole-image) coarse affine from outside, instead
        of letting `self.aligner` register it lazily.

        `coarse_affine_matrix` is in the thumbnail frame (2x3 or 3x3, as
        produced by `register_coarse.coarse_register`). The baseline is used for
        the tissue bbox transform, the background fill in the block-affine
        grid, and the fallback affine in the displacement warp.

        Pass `match_config` (the seeding fit's `Aligner.coarse_match_config`)
        along with it -- a matrix alone carries no record of the configuration it
        was matched under, and `match_config` would have to search for one again.
        """
        self.aligner.coarse_affine_matrix = coarse_affine_matrix
        self.aligner.coarse_match_config = match_config
        # `tissue_bbox_moving` was derived from the previous baseline, and
        # `match_config` from the previous config
        self.__dict__.pop("_tissue_bbox_moving", None)
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
        `_pick_affine` drops it back to the baseline affine (visible in the QC
        panel and scores). Upgrade when seen in
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
    def tissue_bbox(self):
        """`(rs, re, cs, ce)` of all tissue, in reference-thumbnail pixels."""
        if not hasattr(self, "_tissue_bbox"):
            # Deliberately not a lazy `segment_objects()`: it would run with the
            # *default* `merge_gap`/`downscale_factor`, quietly ignoring what the
            # caller meant to segment with, and plot as a side effect.
            raise AttributeError(
                "no tissue segmented yet; call `segment_objects` (or `run`,"
                " which calls it) first"
            )
        return self._tissue_bbox

    @property
    def tissue_bbox_moving(self):
        """`tissue_bbox` mapped through the baseline coarse affine.

        Cached, and invalidated by `segment_objects` (new box) and
        `seed_baseline_coarse` (new affine).
        """
        if not hasattr(self, "_tissue_bbox_moving"):
            self._tissue_bbox_moving = transform_bbox(
                self.tissue_bbox,
                self.baseline_coarse_affine_matrix,
                shape=self.moving_thumbnail.shape,
            )
        return self._tissue_bbox_moving

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
        # One object. Components that clear `min_area` are merged rather than
        # ranked and aligned separately: a per-piece rigid offset is the shift
        # field's job now that it is partitioned into domains, and on the same
        # physical section the pieces cannot move relative to each other anyway
        # (`docs/05` P1, `docs/13`). Specks below `min_area` stay out, exactly
        # as they did when they were labels no object claimed -- the warp gave
        # them the baseline affine then and gives them the baseline affine now.
        merged = np.isin(labeled, np.array(regionprops["label"])[keep])
        rr, cc = np.where(merged)
        self._tissue_bbox = (
            downscale_factor
            * np.array([rr.min(), rr.max() + 1, cc.min(), cc.max() + 1])
            if rr.size
            else np.zeros(4, dtype=int)
        )
        # derived from the box above; re-segmenting invalidates it
        self.__dict__.pop("_tissue_bbox_moving", None)
        self.segmentation_mask = img_util.repeat_2d(
            merged.astype("int32"), (downscale_factor, downscale_factor)
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

    def tissue_block_mask(self, grid_shape=None, threshold=1.0 / 16):
        """Boolean mask over a block grid for the tissue, from the segmentation
        mask rather than its bounding box, so background inside the box is out.

        `grid_shape` defaults to the finest (level1) grid; pass a coarser level's
        `grid_shape` for the multi-res path. Always returns at least one True
        block (falls back to the centroid block when the tissue fills less than
        `threshold` of every block), so masked shift computation never produces
        an all-infinite -- and thus crashing -- level.
        """
        if grid_shape is None:
            grid_shape = self.aligner.grid_shape
        nbi, nbj = grid_shape
        obj = self.segmentation_mask > 0
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
        import matplotlib.pyplot as plt

        colors = matplotlib.cm.Set3.colors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("tissue segmentation")

        def _proc_img(img):
            if img_util.is_brightfield_img(img):
                return img
            return np.log1p(img)

        ax1.imshow(_proc_img(self.ref_thumbnail), cmap="gray")
        ax2.imshow(_proc_img(self.moving_thumbnail), cmap="gray")
        ax1.set_title("reference (tissue outline + bbox)", fontsize=8)
        ax2.set_title(
            "moving (bbox through the baseline coarse affine;\n"
            "dashed = the axis-aligned crop taken from it)",
            fontsize=8,
        )
        tform = skimage.transform.AffineTransform(self.baseline_coarse_affine_matrix)
        rs, re, cs, ce = self.tissue_bbox
        rs2, re2, cs2, ce2 = self.tissue_bbox_moving
        color = colors[0]
        # a contour, not `find_boundaries`: a raster outline at thumbnail
        # resolution aliases into dashes once the figure is rendered small
        ax1.contour(
            self.segmentation_mask > 0, levels=[0.5], colors=[color], linewidths=0.8
        )
        mpatch = matplotlib.patches.Rectangle(
            (cs, rs), ce - cs, re - rs, fill=False, edgecolor=color
        )
        ax1.add_patch(mpatch)
        ax2.add_patch(
            matplotlib.patches.Polygon(
                tform.inverse(mpatch.get_corners()), fill=False, edgecolor=color
            )
        )
        # what is actually cropped for the coarse fit: the mapped box is a
        # parallelogram, the crop is its axis-aligned hull clipped to the image,
        # and a bad affine shows up as the two disagreeing
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

    def _pick_affine(self, candidates, ref_crop, moving, to_crop=None):
        """Choose among coarse affine candidates by overlap score.

        This used to accept the masked fit unconditionally. That fit is a fresh
        feature match on masked thumbnails, so it can land well off while the
        whole-image baseline -- which it is a perturbation of -- was fine. A bad
        affine is a hard visible misregistration, so the fit has to beat the
        fallback rather than merely exist.

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
                f"'{preferred}' coarse affine scores {scores[preferred]:.3f}"
                f" against {scores[chosen]:.3f} for '{chosen}'; falling back"
                f" to '{chosen}'"
            )
        logger.info(
            f"Coarse affine '{chosen}' ("
            + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
            + ")"
        )
        return chosen, scores

    def make_multi_res_aligner(self, min_num_blocks=25):
        """The resolution ladder every object's block shifts are measured on.

        Nothing about it is per-object -- the object enters only through the
        masked thumbnails, the coarse affine and the block mask, all of which
        `align_tissue` reassigns on the aligner it is handed.

        That matters because `MultiResAligner.__init__` persists the ladder,
        which walks the whole moving image once (on the melanoma pair, ~40s to
        decode a 10.8 GB level-0 SVS). Per object, that is the single most
        expensive thing in a multi-piece run.

        Its base rung is the very `get_aligner` call `make_aligner()` makes,
        argument for argument; an unreachable `min_num_blocks` stops the ladder
        there for a single-rung run. Routing through here also gets
        `MultiResAligner.constrain_shifts`, which normalizes
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
            # the class default of 4 would add coarser rungs
            min_num_blocks=min_num_blocks,
        )
        # `block_mask` and the block-affine grid are on `self.aligner`'s grid;
        # `MultiResAligner` keeps level1
        # unconditionally, so its finest rung is the same grid. Pin it -- a
        # mismatch would surface far away, as an IndexError when `block_mask`
        # indexes `shifts` in `align_tissue`.
        assert mr.levels[0] == self.level1, (
            f"multi-res finest level {mr.levels[0]} != level1 {self.level1}"
        )
        return mr

    def align_tissue(
        self,
        plot_shifts=True,
        refine=True,
        min_num_blocks=25,
        windowed_coarse=True,
        coarse_kwargs=None,
        mr=None,
        domain_tol=DEFAULT_DOMAIN_TOL,
    ):
        rs, re, cs, ce = np.array(self.tissue_bbox).astype(int)
        rsm, rem, csm, cem = self.tissue_bbox_moving

        masked_t_ref = np.ones_like(self.ref_thumbnail) * self.fill_value_ref_thumbnail
        masked_t_ref[rs:re, cs:ce] = self.ref_thumbnail[rs:re, cs:ce]

        masked_t_moving = (
            np.ones_like(self.moving_thumbnail) * self.fill_value_moving_thumbnail
        )
        masked_t_moving[rsm:rem, csm:cem] = self.moving_thumbnail[rsm:rem, csm:cem]

        # Built up front so its finest rung can *be* `c21l`: it is at `level1`
        # with the same thumbnails, so a separate `make_aligner()` only
        # duplicated it -- and that duplicate cost a full thumbnail build.
        if mr is None:
            mr = self.make_multi_res_aligner(min_num_blocks=min_num_blocks)
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
                "The baseline affine maps the tissue bbox outside the moving"
                " thumbnail; keeping the baseline coarse affine"
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
                # `_pick_affine` drops it to the baseline, so it is never
                # better than the whole-slide affine. Observed on LSP74545
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
            self._finish_new_figs(figs_before, "Coarse alignment")
            candidates.append(("object", c21l.coarse_affine_matrix))

        # block region from the segmentation mask rather than the bbox, so
        # background inside the box does not steer the fit
        block_mask = self.tissue_block_mask()

        refine_stats = None
        if refine:
            figs_before = self._fignums()
            refined, refine_stats = align_refine.refine_affine_by_block_translation(
                c21l, block_mask=block_mask, plot=plot_shifts
            )
            self._finish_new_figs(figs_before, "Coarse affine refinement")
            if refined is not None:
                c21l.coarse_affine_matrix = refined
                candidates.append(("object+refine", c21l.coarse_affine_matrix))

        chosen, scores = self._pick_affine(
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
        mr.align(mask_fn=self.tissue_block_mask)
        mr.constrain_shifts(domain_tol=domain_tol)
        # the finest rung carries this object's affine; the shifts are the
        # cross-rung pick made by `constrain_shifts`
        affine_matrix, shifts = mr.aligners[0].affine_matrix, mr.shifts
        # QC: show the per-rung selection, not just the combined field
        shift_plotter = mr
        plot_failed = False
        in_object = np.asarray(block_mask).ravel()
        domain_labels = shift_domains.label_domains(
            np.asarray(shifts),
            mr.aligners[0].grid_shape,
            valid=(np.asarray(mr.result_levels) >= 0)
            & in_object.reshape(mr.aligners[0].grid_shape),
            tol=domain_tol,
        )
        if plot_shifts:
            figs_before = self._fignums()
            try:
                shift_plotter.plot_shifts(domain_labels=domain_labels)
            except Exception as e:
                plot_failed = True
                logger.warning(f"Failed plotting shifts: {e}")
            finally:
                # in `finally` so a call that raised partway still hands over
                # what it drew: with nothing sweeping the open figures at the
                # end of the run any more, a figure left behind here is one
                # nothing will ever write or close
                self._finish_new_figs(figs_before, "Block shifts")

        magnitudes = np.linalg.norm(np.asarray(shifts)[in_object], axis=1)
        # Phase-correlation confidence of the blocks this object actually uses.
        # Reported, not yet acted on -- `constrain_block_shifts`' triangle
        # thresholds still decide validity. Logged so a threshold can be
        # calibrated on real slides before anything depends on it.
        errors = np.asarray(mr.shift_errors)[in_object]
        finite = errors[np.isfinite(errors)]
        err_stats = {
            "median": float(np.median(finite)) if finite.size else None,
            "p90": float(np.percentile(finite, 90)) if finite.size else None,
            "frac_inf": float((~np.isfinite(errors)).mean()) if errors.size else None,
        }
        if finite.size:
            logger.info(
                f"Block PC error: median {err_stats['median']:.3f},"
                f" p90 {err_stats['p90']:.3f},"
                f" non-finite {100 * err_stats['frac_inf']:.0f}%"
            )

        # Partition the object's shift field into domains of agreeing shifts.
        # Reported only -- the warp still treats the object as one piece. A
        # block is trusted where some rung actually resolved it; `result_levels
        # == -1` marks the ones carrying rung 0's extrapolation.
        domain_stats = shift_domains.summarize(
            np.asarray(shifts),
            domain_labels,
            within=in_object.reshape(mr.aligners[0].grid_shape),
        )
        separation = shift_domains.max_separation(np.asarray(shifts), domain_labels)
        rows = domain_stats["domains"]
        if len(rows) > 1:
            logger.info(
                f"{len(rows)} shift domains, max separation"
                f" {separation:.1f}px, covering"
                f" {100 * domain_stats['coverage']:.0f}% of"
                f" {domain_stats['n_blocks']} blocks (tol={domain_tol}px)"
            )
            # A domain under 1% of the object is an island in noise, not a
            # piece; 23390 produced 19 of them on a field that was 59%
            # unresolved. Show the substantial ones and count the rest.
            floor = max(shift_domains.MIN_DOMAIN_BLOCKS,
                        0.01 * domain_stats["n_blocks"])
            shown = [r for r in rows if r["blocks"] >= floor][:6]
            for r in shown:
                logger.info(
                    f"    domain {r['domain']}: {r['blocks']:5} blocks,"
                    f" offset ({r['offset'][0]:+7.1f},{r['offset'][1]:+7.1f}),"
                    f" spread {r['spread']:.1f}px"
                )
            # one tail for everything not shown, whether it fell under the
            # floor or off the end of the list -- counting the two separately
            # left domains in neither total
            if len(rows) > len(shown):
                logger.info(f"    + {len(rows) - len(shown)} smaller domain(s)")
        self.qc = (
            {
                "n_blocks": int(in_object.sum()),
                "affine": chosen,
                # what would have been used absent a score collapse; recorded
                # rather than re-derived, since the candidate list is not fixed
                # (a bbox landing off the moving image never gets an "object"
                # candidate at all)
                "preferred": candidates[-1][0],
                "scores": scores,
                "refine": refine_stats,
                "shift_median": float(np.median(magnitudes))
                if magnitudes.size
                else None,
                "shift_max": float(magnitudes.max()) if magnitudes.size else None,
                "levels": mr.level_histogram(in_object),
                "pc_error": err_stats,
                "domains": {**domain_stats, "separation": separation},
                "plot_failed": plot_failed,
            }
        )
        self.tissue_affine = np.asarray(affine_matrix)
        self.tissue_shifts = np.asarray(shifts)
        self.tissue_shift_mask = shift_mask
        # block matrices: the tissue's own affine+shift where it owns the block,
        # the baseline affine everywhere else. This used to be an argmax over
        # per-object masks (`combine_object_results`); with one object it is a
        # plain two-way choice.
        mxs = align.block_affine_matrices(affine_matrix, shifts)
        in_grid = np.asarray(block_mask).ravel()
        mxs[~in_grid] = self.baseline_affine_matrix
        self.block_affine_matrices_da = align.block_affine_matrices_da(
            mxs, self.aligner.grid_shape
        )

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

    @staticmethod
    def _format_levels(hist):
        """Percent of the object's blocks resolved at each level, coarse last.

        A trailing `+N!` is the share no level resolved -- those blocks carry
        level 0's extrapolation, not a measurement.
        """
        if not hist:
            return "-"
        total = sum(hist.values())
        out = "/".join(
            f"{100 * hist[k] / total:.0f}" for k in sorted(k for k in hist if k >= 0)
        )
        if hist.get(-1):
            out += f" +{100 * hist[-1] / total:.0f}!"
        return out or "-"

    def qc_summary(self):
        """One line saying what the alignment actually did.

        Every decision here is already logged, but as single lines among
        thousands: a rejected refinement, a coarse fit that lost to the
        baseline, a shift plot that raised. Those are exactly the outcomes worth
        seeing, and a run that ends without a verdict is not hands-off, only
        quiet.
        """
        r = getattr(self, "qc", None)
        if not r:
            return "  (nothing aligned)"
        head = (
            f"  {'blocks':>6}  {'affine':<14} {'score':>5}  {'refinement':<30}"
            f" {'shift med/max':>13} {'lvl%':>14}  notes"
        )
        # what was actually preferred; fall back to the candidate order for
        # rows built by hand (the self-checks)
        preferred = r.get("preferred") or next(
            name
            for name in ("object+refine", "object", "baseline")
            if name in r["scores"]
        )
        notes = []
        if r["affine"] != preferred:
            notes.append(f"FELL BACK to {r['affine']}")
        dom = r.get("domains") or {}
        # Surfaced only when the field disagrees with itself by more than the
        # displacement warp can smooth over: that is the whole reason to look at
        # the QC figures.
        if len(dom.get("domains", ())) > 1 and dom.get("separation", 0) > 0:
            notes.append(
                f"{len(dom['domains'])} domains, max sep"
                f" {dom['separation']:.0f}px, {100 * dom['coverage']:.0f}% covered"
            )
        # Flagged, not fatal: low coverage has legitimate causes -- tissue
        # genuinely lost between the two scans reads exactly like this -- so the
        # run proceeds and the slide goes on the review list.
        low_coverage = 0 < dom.get("coverage", 1.0) < shift_domains.LOW_COVERAGE
        if low_coverage:
            notes.append("REVIEW: shift field mostly unresolved")
        if r["plot_failed"]:
            notes.append("shift plot failed")
        shift = "-"
        if r["shift_median"] is not None:
            shift = f"{r['shift_median']:.1f} / {r['shift_max']:.1f}"
        score = r["scores"].get(r["affine"], float("nan"))
        rejected = r["refine"] is not None and not r["refine"]["accepted"]
        lines = [
            head,
            "  " + "-" * (len(head) - 2),
            f"  {r['n_blocks']:>6}  {r['affine']:<14} {score:>5.3f} "
            f" {self._format_refine(r['refine']):<30} {shift:>13}"
            f" {self._format_levels(r.get('levels')):>14}  {', '.join(notes)}",
        ]
        if r["affine"] != preferred or rejected or low_coverage:
            lines.append("  ^ check the QC figures")
        lines.append(
            "  lvl% = share of blocks resolved per pyramid level, finest first;"
            " +N! = no level resolved"
        )
        return "\n".join(lines)

    def displacement_transformed_moving_img(
        self,
        moving_img,
        sigma_blocks=0.0,
        field_order=1,
        is_mask=False,
        cval=0.0,
        interpolation="skimage",
    ):
        """Seam-free, mask-constrained warp.

        The tissue is warped by its affine plus one continuous displacement
        field -- its per-block shifts interpolated and smoothed *within the
        tissue mask* via normalized convolution, so the smoothing never bleeds
        across the boundary. `segmentation_mask` says which output pixels are
        tissue at full resolution; background falls back to the baseline affine.

        A per-piece rigid offset is carried inside this one field as a ramp over
        roughly one block, rather than by compositing separate per-piece fields
        (`docs/07`, 2026-08-28 note; strain measured in `docs/12`).

        `sigma_blocks` controls how gradually the displacement blends between
        blocks, in block units: 0 is pure interpolation, ~0.3-1.5 dissolves
        block seams further at the cost of erasing genuine local deformation.
        """
        import scipy.ndimage as ndi

        c21l = self.aligner
        ref_img = c21l.ref_img
        out_shape = tuple(int(s) for s in ref_img.shape[-2:])
        cy, cx = ref_img.chunksize[-2:]
        grid_shape = c21l.grid_shape
        # output (level1) pixels per labeled-mask (ref-thumbnail) pixel
        mask_scale = float(c21l.ref_thumbnail_down_factor)

        base_inv_affine = np.linalg.inv(
            np.asarray(self.baseline_affine_matrix, dtype="float64")
        )
        a_inv = np.linalg.inv(np.asarray(self.tissue_affine, dtype="float64"))
        # displacement = -shift (see block_affine_matrices convention)
        d = -np.asarray(self.tissue_shifts, dtype="float32").reshape(*grid_shape, 2)

        # Outside-tissue blocks never held a measurement: `_pc` returns inf
        # there, and `constrain_block_shifts` either extrapolates them from the
        # tissue's trend or -- on one of its degenerate early returns -- leaves
        # the inf for `MultiResAligner.constrain_shifts` to normalize to 0. That
        # 0 is indistinguishable from a measured "no displacement here", and
        # `_sample_displacement` samples one cell *outside* the tissue when it
        # bilinearly interpolates the rim, so it would pull the rim toward zero.
        #
        # Fill those cells from the nearest in-tissue block instead. The rim
        # then interpolates against a continuation of the field, and no value
        # that merely means "not measured" survives to be read as data.
        # In-tissue cells map to themselves, so this is an identity there.
        #
        # Note this cannot be folded into the normalized convolution below: at
        # the default `sigma_blocks=0` a gaussian filter is an identity, so that
        # branch would leave `d` exactly as it found it.
        in_tissue = self.tissue_block_mask(grid_shape)
        if in_tissue.any():
            _, (fill_r, fill_c) = ndi.distance_transform_edt(
                ~in_tissue, return_indices=True
            )
            d = d[fill_r, fill_c]
        if sigma_blocks and sigma_blocks > 0:
            # normalized convolution: smooth only over the tissue's own blocks
            # so the field isn't pulled toward the background
            weight = in_tissue.astype("float32")
            num = ndi.gaussian_filter(
                d * weight[..., None],
                sigma=(sigma_blocks, sigma_blocks, 0),
                mode="constant",
            )
            den = ndi.gaussian_filter(
                weight, sigma=(sigma_blocks, sigma_blocks), mode="constant"
            )
            valid = den > 1e-3
            d = np.where(valid[..., None], num / np.maximum(den[..., None], 1e-6), d)
        d = np.ascontiguousarray(d)

        field_interp = cv2.INTER_LINEAR if field_order == 1 else cv2.INTER_CUBIC
        order = 0 if is_mask else 1
        tissue_mask = np.ascontiguousarray(self.segmentation_mask > 0)

        return_slice = slice(None)
        mimg = moving_img
        if img_util.is_single_channel(mimg) and mimg.ndim == 2:
            mimg = mimg[np.newaxis]
            return_slice = 0

        template = da.zeros(out_shape, dtype="uint8", chunks=(cy, cx))
        warped = [
            template.map_blocks(
                align._displacement_block,
                src_array=c,
                inv_affine=a_inv,
                grid=d,
                base_inv_affine=base_inv_affine,
                tissue_mask=tissue_mask,
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
