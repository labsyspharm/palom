import pathlib
import pprint
import sys

import matplotlib
import matplotlib.pyplot as plt
import skimage.exposure
from loguru import logger

import palom


class CoarseAlignmentFailed(RuntimeError):
    """The coarse affine landed nowhere; later stages cannot mean anything."""


def align_he(
    p1: str | pathlib.Path,
    p2: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    out_name: str | None = None,
    thumbnail_channel1: int = 1,
    thumbnail_channel2: int = 1,
    channel1: int = 0,
    channel2: int = 2,
    px_size1: float | None = None,
    px_size2: float | None = None,
    n_keypoints: int = 10_000,
    coarse_n_workers: int = 1,
    thumbnail_pixel_size: float | None = None,
    shift_block_size: int | None = None,
    multi_obj: bool = True,
    merge_gap: float = 500.0,
    exclude_objects: list | None = None,
    min_num_blocks: int = 25,
    windowed_coarse: bool = True,
    min_coarse_ncc: float = 0.02,
    only_coarse: bool = False,
    only_qc: bool = False,
    viz_coarse_napari: bool = False,
    displacement_warp: bool = True,
    smooth_shifts_sigma: float = 0.0,
    warp_interpolation: str = "skimage",
    intensity_in_range: tuple[int, int] | None = None,
    jpeg_compress: bool = False,
):
    _args = locals()
    out_dir, p1, p2 = pathlib.Path(out_dir), pathlib.Path(p1), pathlib.Path(p2)
    if out_name is None:
        out_name = f"{p2.stem}-registered.ome.tif"
    log_path = out_dir / "log" / f"{out_name}.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger.remove()
    logger.add(sys.stderr)
    logger.add(log_path, rotation="5 MB")
    logger.info(f"Start processing {p2.name}")
    logger.info(
        f"\nFunction args\n{pprint.pformat(_args, indent=4, sort_dicts=False, width=600)}\n"
    )
    out_path = out_dir / out_name
    assert "".join(out_path.suffixes[-2:]) in (".ome.tif", ".ome.tiff")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if intensity_in_range is not None:
        assert sorted(intensity_in_range) == list(intensity_in_range)
        assert len(intensity_in_range) == 2

    set_matplotlib_font(font_size=8)

    r1 = get_reader(p1)(p1, pixel_size=px_size1)
    r2 = get_reader(p2)(p2, pixel_size=px_size2)

    if shift_block_size is not None:
        # Resolution of the deformation field: one shift is measured per block,
        # and the warp interpolates between block centers, so smaller blocks
        # resolve finer local deformation. Set by rechunking the reference
        # pyramid's spatial axes; the moving image and the per-level aligners
        # inherit it. Smaller blocks carry less texture per phase correlation,
        # but the multi-res pass already guards that -- a block whose shift is
        # unreliable is rejected and the coarser level's value wins.
        r1.pyramid = [
            level.rechunk({1: shift_block_size, 2: shift_block_size})
            for level in r1.pyramid
        ]
    logger.info(
        f"Shift-field block size (`shift_block_size`={shift_block_size}):"
        f" {r1.pyramid[0].chunksize[1:]} px at level 0"
    )

    LEVEL1 = 0
    aligner = palom.align.get_aligner(
        r1,
        r2,
        level1=LEVEL1,
        channel1=channel1,
        channel2=channel2,
        # when thumbnail_pixel_size is set it wins and builds both thumbnails at
        # that fixed physical pixel size; when it's None, thumbnail_level1=None
        # falls back to the original matched-level behavior
        thumbnail_level1=None,
        thumbnail_channel1=thumbnail_channel1,
        thumbnail_channel2=thumbnail_channel2,
        thumbnails_pixel_size=thumbnail_pixel_size,
    )
    if not (r1.has_pixel_size and r2.has_pixel_size):
        # `get_aligner` leaves the thumbnail pixel sizes unset, so the coarse
        # route is picked from match confidence alone -- the honest degradation,
        # since a placeholder 1 µm on one side alone skews the physical-footprint
        # ratio by the true pixel size and can flip the route
        logger.warning(
            "Missing pixel size metadata; choosing the coarse route from match"
            " confidence alone. Pass `px_size1`/`px_size2` to enable the"
            " physical-footprint test"
        )
    aligner.coarse_register_affine(
        n_keypoints=n_keypoints, n_workers=coarse_n_workers
    )
    fig = plt.gcf()
    fig.suptitle(f"{p2.name} (coarse alignment)", fontsize=8)
    ax = fig.axes[0]  # feature-match axes (first for both the whole-image and
    # windowed routes). Keep whatever title the route set (the windowed route puts
    # its tile/ncc info there) and add the ref name only -- `p2.name` is already in
    # the suptitle, and both names together overflow the narrower windowed panel.
    ax.set_title(
        "\n".join(filter(None, [ax.get_title(), f"ref: {p1.name}"])), fontsize=6
    )
    # no sizing here: `plot_util.size_axes_to_image` runs where the figure is
    # drawn, so this one and the per-object coarse plots match
    save_all_figs(out_dir=out_dir / "qc", format="jpg", dpi=144)

    # Fail fast: a coarse affine that landed nowhere makes every later stage
    # meaningless, and `block_affine` eventually raises on the absurd source
    # crop it implies (LSP74569c, ~4 min in). This is a floor for the hopeless,
    # NOT a quality gate -- over the 21 reference slides the committed ncc runs
    # 0.009-0.60, the only crasher scored 0.0093, and the next lowest (0.0590)
    # produced one of the best results in the set. Keep the floor well under
    # that; `register_coarse.min_ncc` (0.10, the route-choice threshold) would
    # throw good slides away.
    coarse_ncc = palom.register_coarse.committed_ncc(
        aligner.ref_thumbnail, aligner.moving_thumbnail, aligner.coarse_affine_matrix
    )
    if min_coarse_ncc is not None and coarse_ncc < min_coarse_ncc:
        raise CoarseAlignmentFailed(
            f"{p2.name}: coarse alignment scored ncc={coarse_ncc:.4f}, below"
            f" min_coarse_ncc={min_coarse_ncc}. Check the coarse QC figure in"
            f" {out_dir / 'qc'}; try --thumbnail_channel1/2, --n_keypoints, or"
            " --px_size1/2. Pass --min_coarse_ncc=0 to run anyway."
        )

    if viz_coarse_napari:
        _ = viz_coarse(
            r1, r2, LEVEL1, aligner.level2, channel1, channel2,
            aligner.affine_matrix
        )

    if not only_coarse:
        # Single alignment path: the multi-object orchestrator, always
        # multi-res. `multi_obj` toggles segmentation -- True (default) splits
        # the scan into tissue pieces and aligns each independently; False
        # treats the whole scan as one global object (the classic
        # single-object, multi-res alignment == N=1).
        mo_aligner = palom.align_multi_obj.MultiObjAligner(
            r1,
            r2,
            level1=LEVEL1,
            channel1=channel1,
            channel2=channel2,
            thumbnail_channel1=thumbnail_channel1,
            thumbnail_channel2=thumbnail_channel2,
            thumbnails_pixel_size=thumbnail_pixel_size,
        )
        # reuse the coarse affine computed above as the baseline, instead of
        # poking private attributes -- with the config it was matched under, so
        # the per-object fits inherit it instead of searching for their own
        mo_aligner.seed_baseline_coarse(
            aligner.coarse_affine_matrix, aligner.coarse_match_config
        )
        mo_aligner.run(
            segment=multi_obj,
            merge_gap=merge_gap,
            exclude_objects=exclude_objects,
            min_num_blocks=min_num_blocks,
            windowed_coarse=windowed_coarse,
            # the same coarse budget/workers the baseline fit above used, so
            # every coarse registration in the run is the same registration
            coarse_kwargs=dict(n_keypoints=n_keypoints, n_workers=coarse_n_workers),
            # each QC figure is written as it is drawn rather than swept up
            # here, so a run that dies on object 5 still leaves the figures for
            # objects 0-4 -- and object 5's earlier stages -- to look at
            qc_dir=out_dir / "qc" / p2.stem,
        )
        block_mx = mo_aligner.block_affine_matrices_da

    if not only_qc:
        # the moving level must be the one whose frame `mx` was fit in -- the
        # two aligners derive it identically, but read it off whichever one
        # produced the matrix rather than assuming they agree
        mx, level2 = aligner.affine_matrix, aligner.level2
        if not only_coarse:
            mx, level2 = block_mx, mo_aligner.level2

        if displacement_warp and not only_coarse:
            # seam-free, mask-constrained warp: each object gets its own
            # continuous displacement field (a single object in the
            # non-segmented case), composited per pixel by the labeled
            # segmentation mask.
            mosaic = mo_aligner.displacement_transformed_moving_img(
                r2.pyramid[level2],
                sigma_blocks=smooth_shifts_sigma,
                exclude_objects=exclude_objects,
                interpolation=warp_interpolation,
            )
        else:
            # `only_coarse` has no block shifts, so there is no displacement
            # field to build -- the coarse affine warp is the only option
            mosaic = palom.align.block_affine_transformed_moving_img(
                ref_img=aligner.ref_img, moving_img=r2.pyramid[level2], mxs=mx
            )

        if (mosaic.shape[0] == 3) & (intensity_in_range is not None):
            out_dtype = mosaic.dtype.name
            mosaic = mosaic.map_blocks(
                lambda x: skimage.exposure.rescale_intensity(
                    x, in_range=intensity_in_range, out_range=out_dtype
                )
                .round()
                .astype(out_dtype),
                dtype=out_dtype,
            )
        tifffile_kwarg = {}
        if palom.pyramid.count_num_channels([mosaic]) == 3:
            tifffile_kwarg.update(dict(photometric="rgb", planarconfig="separate"))
        palom.pyramid.write_pyramid(
            mosaics=[mosaic],
            output_path=out_path,
            pixel_size=r1.pixel_size * r1.level_downsamples[LEVEL1],
            channel_names=[list("RBG")],
            compression="zlib",
            downscale_factor=2,
            save_RAM=True,
            tile_size=1024,
            kwargs_tifffile=tifffile_kwarg,
        )
        if jpeg_compress:
            from palom.cli import compress_rgb_jpeg

            compress_rgb_jpeg.compress_rgb(
                out_path,
                output=out_path.parent
                / out_path.name.replace(".ome.tif", "-jpeg.ome.tif"),
            )
    return 0


def viz_coarse(r1, r2, level1, level2, channel1, channel2, mx):
    try:
        import napari
    except ImportError:
        return
    import dask.array as da

    v = napari.Viewer()
    is_bf1 = palom.img_util.is_brightfield_img(r1.pyramid[-1][channel1])
    is_bf2 = palom.img_util.is_brightfield_img(r2.pyramid[-1][channel2])
    inv = {True: da.invert, False: da.array}
    v.add_image(
        [inv[is_bf1](p[channel1]) for p in r1.pyramid[level1:]],
        colormap="bop orange",
        blending="additive",
    )
    v.add_image(
        [inv[is_bf2](p[channel2]) for p in r2.pyramid[level2:]],
        affine=palom.img_util.to_napari_affine(mx),
        colormap="bop blue",
        blending="additive",
    )
    return v


def get_reader(path):
    path = pathlib.Path(path)
    if path.suffix in [".svs", ".ndpi"]:
        return palom.reader.SvsReader
    elif path.suffix == ".vsi":
        return palom.reader.VsiReader
    elif path.suffix == ".qptiff":
        return palom.reader.QptiffPyramidReader
    else:
        return palom.reader.OmePyramidReader


def set_matplotlib_font(font_size=12):
    font_families = matplotlib.rcParams["font.sans-serif"]
    if font_families[0] != "Arial":
        font_families.insert(0, "Arial")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams.update({"font.size": font_size})


def save_all_figs(dpi=300, format="pdf", out_dir=None):
    """Write and close every open figure, naming each from its suptitle.

    The only caller left is the coarse QC flush, where exactly one figure is
    open and the sweep is unambiguous. Everything the multi-object run draws is
    written by `MultiObjAligner` as it goes (`run(qc_dir=...)`), so this no
    longer has to guess which of the open figures belong to the stage that just
    finished -- nor rename them after the fact, which is what the old `prefix`
    did.
    """
    figs = [plt.figure(i) for i in plt.get_fignums()]
    names = [f._suptitle.get_text() if f._suptitle else "" for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches="tight")
        plt.close(f)


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    import csv
    import inspect
    import pprint
    import types

    from fire.parser import DefaultParseValue

    func = align_he

    if print_args:
        _args = [str(vv) for vv in inspect.signature(func).parameters.values()]
        print(f"\nFunction args\n{pprint.pformat(_args, indent=4)}\n")
    _arg_types = inspect.get_annotations(func)
    arg_types = {}
    for k, v in _arg_types.items():
        if isinstance(v, types.UnionType):
            v = v.__args__[0]
        arg_types[k] = v

    with open(csv_path) as f:
        csv_kwargs = [
            {
                kk: arg_types[kk](DefaultParseValue(vv))
                for kk, vv in rr.items()
                if (kk in arg_types) & (vv is not None)
            }
            for rr in csv.DictReader(f)
        ]

    if dryrun:
        for kk in csv_kwargs:
            pprint.pprint({**kwargs, **kk}, sort_dicts=False)
            print()
        return

    # One bad row must not cost the rest of the batch -- a pair whose coarse
    # alignment lands nowhere can raise deep in the warp, and until 2026-08-27
    # that aborted every remaining row.
    failures = []
    for i, kk in enumerate(csv_kwargs):
        merged = {**kwargs, **kk}
        try:
            func(**merged)
        except Exception as e:
            name = pathlib.Path(str(merged.get("p2", f"row {i}"))).name
            failures.append((name, f"{type(e).__name__}: {e}"))
            logger.opt(exception=True).error(f"FAILED {name}; continuing batch")
    logger.info(
        f"Batch finished: {len(csv_kwargs) - len(failures)}/{len(csv_kwargs)} succeeded"
    )
    for name, err in failures:
        logger.error(f"  FAILED {name} -- {err}")
    return 1 if failures else 0


def main():
    import fire

    fire.Fire({"run-pair": align_he, "run-batch": run_batch})

    if ("--viz_coarse_napari" in sys.argv) or ("-v" in sys.argv):
        try:
            import napari
        except ImportError:
            print("napari is not installed")
        else:
            napari.run()


if __name__ == "__main__":
    import sys

    sys.exit(main())

    r"""
    Example 1: inspect coarse alignment using napari
    python align_he.py run-pair\
        Z:\RareCyte-S3\P54_CRCstudy_Bridge\P54_S33_Full_Or6_A31_C90c_HMS@20221025_001610_632297.ome.tiff \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\22199$P54_33_HE$US$SCAN$OR$001 _104050.svs" \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\test" \
        --px_size1 0.325 --only_qc --only_coarse --viz_coarse_napari

    Example 2: process pair and output registered image
    python align_he.py run-pair\
        Z:\RareCyte-S3\P54_CRCstudy_Bridge\P54_S33_Full_Or6_A31_C90c_HMS@20221025_001610_632297.ome.tiff \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\22199$P54_33_HE$US$SCAN$OR$001 _104050.svs" \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\test" \
        --px_size1 0.325
    """
