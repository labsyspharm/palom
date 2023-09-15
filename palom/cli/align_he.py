import pathlib

import matplotlib
import matplotlib.pyplot as plt

import palom


def align_he(
    p1: str | pathlib.Path,
    p2: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    out_name: str = None,
    thumbnail_channel1: int = 1,
    thumbnail_channel2: int = 2,
    channel1: int = 0,
    channel2: int = 2,
    px_size1: float = None,
    px_size2: float = None,
    n_keypoints: int = 10_000,
    auto_mask=True,
    only_coarse: bool = False,
    only_qc: bool = False,
    viz_coarse_napari: bool = False,
    multi_res: bool = False,
    multi_obj: bool = False,
    multi_obj_kwarg: dict = None
):
    assert not (multi_res and multi_obj), (
    'setting both `multi_res` and `multi_obj` to `True` is not supported,'
    ' choose at most one'
    )
    out_dir, p1, p2 = pathlib.Path(out_dir), pathlib.Path(p1), pathlib.Path(p2)
    if out_name is None:
        out_name = f"{p2.stem}-registered.ome.tif"
    out_path = out_dir / out_name
    assert ''.join(out_path.suffixes[-2:]) in ('.ome.tif', '.ome.tiff')
    out_path.parent.mkdir(exist_ok=True, parents=True)
    set_matplotlib_font(font_size=8)

    r1 = get_reader(p1)(p1, pixel_size=px_size1)
    r2 = get_reader(p2)(p2, pixel_size=px_size2)

    LEVEL = 0
    aligner = palom.align.get_aligner(
        r1, r2,
        level1=LEVEL, level2=LEVEL,
        channel1=channel1, channel2=channel2,
        # make thumbnail level pair based on pixel_size
        thumbnail_level1=None,
        thumbnail_channel1=thumbnail_channel1, thumbnail_level2=thumbnail_channel2
    )

    aligner.coarse_register_affine(
        n_keypoints=n_keypoints,
        test_flip=True,
        test_intensity_invert=True,
        auto_mask=auto_mask
    )
    fig, ax = plt.gcf(), plt.gca()
    fig.suptitle(f"{p2.name} (coarse alignment)", fontsize=8)
    ax.set_title(f"{p1.name} - {p2.name}", fontsize=6)
    im_h, im_w = ax.images[0].get_array().shape
    set_subplot_size(im_w/288, im_h/288, ax=ax)
    ax.set_anchor('N')
    # use 0.5 inch on the top for figure title
    fig.subplots_adjust(top=1 - .5 / fig.get_size_inches()[1])
    save_all_figs(out_dir=out_dir / 'qc', format='jpg', dpi=144)
   
    if viz_coarse_napari:
        _ = viz_coarse(r1, r2, LEVEL, LEVEL, channel1, channel2, aligner.affine_matrix)

    if not only_coarse:
        # the default
        if not (multi_res or multi_obj):  
            aligner.compute_shifts()
        
            fig = aligner.plot_shifts()
            fig.suptitle(f"{p2.name} (block shift distance)", fontsize=8)
            fig.axes[0].set_title(p1.name, fontsize=6)
            save_all_figs(out_dir=out_dir / 'qc', format='png')

            aligner.constrain_shifts()
            block_mx = aligner.block_affine_matrices_da
        elif multi_res:
            mr_aligner = palom.align_multi_res.MultiResAligner(
                r1, r2,
                level1=LEVEL,
                channel1=channel1, channel2=channel2,
                thumbnail_channel1=thumbnail_channel1,
                thumbnail_channel2=thumbnail_channel2,
                min_num_blocks=25
            )
            mr_aligner._coarse_affine_matrix = aligner.coarse_affine_matrix
            mr_aligner.align()
            mr_aligner.constrain_shifts()
            
            fig = mr_aligner.plot_shifts()
            fig.suptitle(f"{p2.name} (multi-res aligment)", fontsize=8)
            fig.axes[0].set_title(p1.name, fontsize=6)
            save_all_figs(out_dir=out_dir / 'qc', format='png', dpi=144)

            pickle_dir = out_dir / 'pickle'
            if not pickle_dir.exists():
                pickle_dir.mkdir(parents=True)
            import pickle
            with open(pickle_dir / f"{p2.name}-palom.pkl", 'wb') as f:
                pickle.dump(mr_aligner, f)

            aligner = mr_aligner.base_aligner
            block_mx = mr_aligner.base_aligner.block_affine_matrices_da
        elif multi_obj:
            mo_aligner = palom.align_multi_obj.MultiObjAligner(
                r1, r2,
                level1=LEVEL,
                channel1=channel1, channel2=channel2,
                thumbnail_channel1=thumbnail_channel1,
                thumbnail_channel2=thumbnail_channel2
            )
            mo_aligner._affine_matrix = aligner.affine_matrix
            mo_aligner._coarse_affine_matrix = aligner.coarse_affine_matrix
            if multi_obj_kwarg is None: multi_obj_kwarg = {}
            mo_aligner.run(**multi_obj_kwarg)
            save_all_figs(out_dir=out_dir / 'qc' / p2.stem, format='png', dpi=144, prefix=p2.name)
            block_mx = mo_aligner.block_affine_matrices_da

   
    if not only_qc:
        mx = aligner.affine_matrix
        if not only_coarse:
            mx = block_mx
        
        mosaic = palom.align.block_affine_transformed_moving_img(
            ref_img=aligner.ref_img,
            moving_img=r2.pyramid[LEVEL],
            mxs=mx
        )
        palom.pyramid.write_pyramid(
            mosaics=[mosaic],
            output_path=out_path,
            pixel_size=r1.pixel_size*r1.level_downsamples[LEVEL],
            channel_names=[list('RBG')],
            compression='zlib',
            downscale_factor=2,
            save_RAM=True,
            tile_size=1024,
            kwargs_tifffile=dict(photometric='rgb', planarconfig='separate')
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
        colormap='bop orange', blending='additive'
    )
    v.add_image(
        [inv[is_bf2](p[channel2]) for p in r2.pyramid[level2:]],
        affine=palom.img_util.to_napari_affine(mx),
        colormap='bop blue', blending='additive'
    )
    return v


def get_reader(path):
    path = pathlib.Path(path)
    if path.suffix == '.svs':
        return palom.reader.SvsReader
    else:
        return palom.reader.OmePyramidReader


def set_matplotlib_font(font_size=12):
    font_families = matplotlib.rcParams['font.sans-serif']
    if font_families[0] != 'Arial':
        font_families.insert(0, 'Arial')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams.update({'font.size': font_size})


def save_all_figs(dpi=300, format='pdf', out_dir=None, prefix=None):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if prefix is not None:
        for f in figs:
            if f._suptitle:
                f.suptitle(f"{prefix} {f._suptitle.get_text()}")
            else:
                f.suptitle(prefix)
    names = [f._suptitle.get_text() if f._suptitle else '' for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches='tight')
        plt.close(f)


def set_subplot_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


if __name__ == '__main__':
    import fire
    import sys

    fire.Fire(align_he)

    if ('--viz_coarse_napari' in sys.argv) or ('-v' in sys.argv):
        try: import napari
        except ImportError: print("napari is not installed")
        else: napari.run()
   
    '''
    Example 1: inspect coarse alignment using napari
    python align_he.py \
        Z:\RareCyte-S3\P54_CRCstudy_Bridge\P54_S33_Full_Or6_A31_C90c_HMS@20221025_001610_632297.ome.tiff \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\22199$P54_33_HE$US$SCAN$OR$001 _104050.svs" \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\test" \
        --px_size1 0.325 --only_qc --only_coarse --viz_coarse_napari

    Example 2: process pair and output registered image
    python align_he.py \
        Z:\RareCyte-S3\P54_CRCstudy_Bridge\P54_S33_Full_Or6_A31_C90c_HMS@20221025_001610_632297.ome.tiff \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\22199$P54_33_HE$US$SCAN$OR$001 _104050.svs" \
        "X:\crc-scans\histowiz scans\20230105-orion_2_cycles\test" \
        --px_size1 0.325
   
    '''
    