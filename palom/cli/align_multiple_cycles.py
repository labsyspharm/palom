import pathlib
import argparse
import matplotlib
import matplotlib.pyplot as plt
import palom
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Align multiple cycles of images using PALOM'
        )   
    )
    parser.add_argument(
        '--img_list',
        dest='img_list',
        nargs='+',
        help='list of image paths to be aligned'
    )
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        help='output directory'
    )
    parser.add_argument(
        '--out_name',
        dest='out_name',
        help='output file name'
    )
    parser.add_argument(
        '--thumbnail_level',
        metavar='thumbnail_level',
        type=int,
        default=None,
        help='Level to use for coarse alignment'
    )
    parser.add_argument(
        '--channel',
        dest='channel',
        type=int,
        default=0,
        help='channel to use for alignment'
    )
    parser.add_argument(
        '--px_size',
        dest='px_size',
        type=float,
        help='pixel size'
    )
    parser.add_argument(
        '--n_keypoints',
        dest='n_keypoints',
        type=int,
        default=10000,
        help='number of keypoints'
    )
    parser.add_argument(
        '--auto_mask',
        dest='auto_mask',
        type=bool,
        default=True,
        help='auto mask'
    )
    parser.add_argument(
        '--only_coarse',
        dest='only_coarse',
        action='store_true',
        help='only coarse alignment'
    )
    parser.add_argument(
        '--only_qc',
        dest='only_qc',
        type=bool,
        default=False,
        help='only quality control'
    )
    arg = parser.parse_args()

    return arg

def align_multiple_cycles(args):
    #thumbnail_channel: int = 1,

    only_coarse = args.only_coarse
    only_qc = args.only_qc
    auto_mask = args.auto_mask
    n_keypoints = args.n_keypoints
    px_size = args.px_size

    out_dir = pathlib.Path(args.out_dir)
    img_list = [pathlib.Path(img_path) for img_path in args.img_list]
    assert len(img_list) >= 2, "At least two images are required for registration"

    if args.out_name is None:
        out_name = f"{img_list[1].stem}-registered.ome.tif"
    else:
        out_name = args.out_name
    channel = args.channel

    out_path = out_dir / out_name
    assert ''.join(out_path.suffixes[-2:]) in ('.ome.tif', '.ome.tiff')
    out_path.parent.mkdir(exist_ok=True, parents=True)
    set_matplotlib_font(font_size=8)

    ref_reader = palom.reader.OmePyramidReader(img_list[0], pixel_size=px_size)
    moving_readers = [palom.reader.OmePyramidReader(file) for file in img_list[1:]]

    LEVEL = 0
    if args.thumbnail_level is None:
        THUMBNAIL_LEVEL = ref_reader.get_thumbnail_level_of_size(2000)
    else:
        THUMBNAIL_LEVEL = args.thumbnail_level

    mosaic_list = [ref_reader.pyramid[LEVEL]]
    for idx, moving_reader in enumerate(moving_readers):
        aligner = palom.align.Aligner(
                # use the first channel (Hoechst staining) in the reference image as the
                # registration reference
                ref_img=ref_reader.read_level_channels(LEVEL, channel),
                # use the second channel (G channel) in the moving image, it usually has
                # better contrast
                moving_img=moving_reader.read_level_channels(LEVEL, channel),
                # select the same channels for the thumbnail images
                ref_thumbnail=ref_reader.read_level_channels(THUMBNAIL_LEVEL, channel).compute(),
                moving_thumbnail=moving_reader.read_level_channels(THUMBNAIL_LEVEL, channel).compute(),
                # specify the downsizing factors so that the affine matrix can be scaled to
                # match the registration reference
                ref_thumbnail_down_factor=ref_reader.level_downsamples[THUMBNAIL_LEVEL] /ref_reader.level_downsamples[LEVEL],
                moving_thumbnail_down_factor=moving_reader.level_downsamples[THUMBNAIL_LEVEL] / moving_reader.level_downsamples[LEVEL]
            )

        aligner.coarse_register_affine(
            n_keypoints=n_keypoints,
            test_flip=True,
            test_intensity_invert=True,
            auto_mask=auto_mask
        )

        fig, ax = plt.gcf(), plt.gca()
        fig.suptitle(f"{img_list[idx+1].name} (coarse alignment)", fontsize=8)
        ax.set_title(f"{img_list[0].name} - {img_list[idx+1].name}", fontsize=6)
        im_h, im_w = ax.images[0].get_array().shape
        set_subplot_size(im_w/288, im_h/288, ax=ax)
        ax.set_anchor('N')
        # use 0.5 inch on the top for figure title
        fig.subplots_adjust(top=1 - .5 / fig.get_size_inches()[1])
        save_all_figs(out_dir=out_dir / 'qc', format='jpg', dpi=144)

        if not only_coarse:
            
            aligner.compute_shifts()
        
            fig = aligner.plot_shifts()
            fig.suptitle(f"{img_list[idx+1].name} (block shift distance)", fontsize=8)
            fig.axes[0].set_title(img_list[0].name, fontsize=6)
            save_all_figs(out_dir=out_dir / 'qc', format='png')

            aligner.constrain_shifts()
            block_mx = aligner.block_affine_matrices_da

   
        if not only_qc:
            mx = aligner.affine_matrix
            if not only_coarse:
                mx = block_mx
            
            mosaic = palom.align.block_affine_transformed_moving_img(
                ref_img=ref_reader.read_level_channels(LEVEL, channel),
                moving_img=moving_reader.pyramid[LEVEL],
                mxs=mx
            )
            mosaic_list.append(mosaic)

    if not only_coarse and not only_qc:

        palom.pyramid.write_pyramid(
            mosaics=mosaic_list,
            output_path=out_path,
            pixel_size=ref_reader.pixel_size*ref_reader.level_downsamples[LEVEL],
            compression='zlib',
            downscale_factor=2,
            save_RAM=True,
            tile_size=1024
        )
    return 0


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

def main():
    align_multiple_cycles(parse_args())

if __name__ == '__main__':
    sys.exit(main())

    