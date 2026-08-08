import pathlib
import pprint
import sys

import matplotlib
import matplotlib.pyplot as plt
import skimage.exposure
from loguru import logger

import palom
from palom.plot_util import set_subplot_size


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
    if len(fig.axes) == 1:
        # whole-image single-panel plot: size the subplot to the match image
        im_h, im_w = ax.images[0].get_array().shape
        set_subplot_size(im_w / 288, im_h / 288, ax=ax)
        ax.set_anchor("N")
        # use 0.5 inch on the top for figure title -- grow the figure to make
        # room instead of eating into the axes, which fails outright
        # (`bottom cannot be >= top`) when the match image is under ~0.5 inch
        figw, figh = fig.get_size_inches()
        fig.set_size_inches(figw, figh + 0.5)
        fig.subplots_adjust(top=1 - 0.5 / (figh + 0.5))
    save_all_figs(out_dir=out_dir / "qc", format="jpg", dpi=144)

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
        # poking private attributes
        mo_aligner.seed_baseline_coarse(aligner.coarse_affine_matrix)
        mo_aligner.run(
            segment=multi_obj,
            merge_gap=merge_gap,
            exclude_objects=exclude_objects,
            min_num_blocks=min_num_blocks,
            windowed_coarse=windowed_coarse,
            # the same coarse budget/workers the baseline fit above used, so
            # every coarse registration in the run is the same registration
            coarse_kwargs=dict(n_keypoints=n_keypoints, n_workers=coarse_n_workers),
        )
        save_all_figs(
            out_dir=out_dir / "qc" / p2.stem, format="png", dpi=144, prefix=p2.name
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


def save_all_figs(dpi=300, format="pdf", out_dir=None, prefix=None):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if prefix is not None:
        for f in figs:
            if f._suptitle:
                f.suptitle(f"{prefix} {f._suptitle.get_text()}")
            else:
                f.suptitle(prefix)
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

    for kk in csv_kwargs:
        func(**{**kwargs, **kk})


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
