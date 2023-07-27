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
    only_coarse: bool = False,
    only_qc: bool = False,
    viz_coarse_napari: bool = False,
    multires: bool = False
):
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
        thumbnail_level1=None,
        channel1=thumbnail_channel1, channel2=thumbnail_channel2,
        level1=LEVEL, level2=LEVEL
    )

    aligner.coarse_register_affine(n_keypoints=n_keypoints, detect_flip_rotate=True)
    plt.gcf().suptitle(f"{p2.name} (coarse alignment)", fontsize=8)
    plt.gca().set_title(f"{p1.name} - {p2.name}", fontsize=6)
    save_all_figs(out_dir=out_dir / 'qc', format='png')
   
    if viz_coarse_napari:
        _ = viz_coarse(r1, r2, LEVEL, LEVEL, channel1, channel2, aligner.affine_matrix)

    if not only_coarse:
        if not multires:
            aligner.ref_img = r1.read_level_channels(LEVEL, channel1)
            aligner.moving_img = r2.read_level_channels(LEVEL, channel2)
        
            aligner.compute_shifts()
        
            fig = aligner.plot_shifts()
            fig.suptitle(f"{p2.name} (block shift distance)", fontsize=8)
            fig.axes[0].set_title(p1.name, fontsize=6)
            save_all_figs(out_dir=out_dir / 'qc', format='png')

            aligner.constrain_shifts()
        else:
            mr_aligner = palom.align_multires.MultiresAligner(
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
            fig.suptitle(f"{p2.name} (multires aligment)", fontsize=8)
            fig.axes[0].set_title(p1.name, fontsize=6)
            save_all_figs(out_dir=out_dir / 'qc', format='png', dpi=144)

            pickle_dir = out_dir / 'pickle'
            if not pickle_dir.exists():
                pickle_dir.mkdir(parents=True)
            import pickle
            with open(pickle_dir / f"{p2.name}-palom.pkl", 'wb') as f:
                pickle.dump(mr_aligner, f)

            aligner = mr_aligner.base_aligner
   
    if not only_qc:
        mx = aligner.affine_matrix
        if not only_coarse:
            mx = aligner.block_affine_matrices_da
        
        mosaic = palom.align.block_affine_transformed_moving_img(
            ref_img=aligner.ref_img,
            moving_img=r2.pyramid[LEVEL],
            mxs=mx
        )
        palom.pyramid.write_pyramid(
            mosaics=[mosaic],
            output_path=out_path,
            pixel_size=px_size1*r1.level_downsamples[LEVEL],
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


def save_all_figs(dpi=300, format='pdf', out_dir=None):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    names = [f._suptitle.get_text() if f._suptitle else '' for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f"{n}-{nm}.{format}", dpi=dpi, bbox_inches='tight')
        plt.close(f)


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
    