import pathlib

import tifffile
from loguru import logger

import palom


PYRAMID_DEFAULTS = dict(
    downscale_factor=2,
    compression='zlib',
    tile_size=1024,
    save_RAM=True
)


def validate_out_path(out_path, default, overwrite):
    if out_path is None:
        out_path = default
    out_path = pathlib.Path(out_path).absolute()
    if ''.join(out_path.suffixes[-2:]).lower() not in ('.ome.tif', '.ome.tiff'):
        logger.error(
            'out_path must ends with .ome.tif or .ome.tiff'
        )
    if (not overwrite) & out_path.exists():
        logger.error(
            f"Aborted. Destination file exists {out_path.name}."
            " Set `overwrite=True` to overwrite it."
        )
        raise OSError('File exists')
    return out_path


def merge_channels(img_paths, out_path=None, overwrite=False, pyramid_config=None):
    readers = [palom.reader.OmePyramidReader(pp) for pp in img_paths]
    img_path = pathlib.Path(img_paths[0])
    stem = img_path.name.replace(''.join(img_path.suffixes), '')
    out_path = validate_out_path(
        out_path,
        img_path.parent / f"merged-{stem}-zlib.ome.tif",
        overwrite=overwrite
    )
    mosaics = [rr.pyramid[0] for rr in readers]
    pixel_size = readers[0].pixel_size
    try:
        tif_tags = src_tif_tags(img_path)
    except Exception:
        tif_tags = {}
    if pyramid_config is None: pyramid_config = {}
    text = '''
    Processing:
        {}
    '''.format('\n\t'.join([pathlib.Path(p).name for p in img_paths]))
    logger.info(text)
    palom.pyramid.write_pyramid(
        mosaics,
        out_path,
        **{
            **dict(
                pixel_size=pixel_size,
                kwargs_tifffile=tif_tags,
            ),
            **PYRAMID_DEFAULTS,
            **pyramid_config
        }
    )
    return out_path


def extract_channels(
    img_path, channels, out_path=None, overwrite=False, pyramid_config=None
):
    img_path = pathlib.Path(img_path)
    reader = palom.reader.OmePyramidReader(img_path)
    stem = img_path.name.replace(''.join(img_path.suffixes), '')
    out_path = validate_out_path(
        out_path,
        img_path.parent / f"extracted-{stem}-channel-{'_'.join([str(c) for c in channels])}.ome.tif",
        overwrite=overwrite
    )
    mosaics = [reader.pyramid[0][channels]]
    try:
        tif_tags = src_tif_tags(img_path)
    except Exception:
        tif_tags = {}
    if pyramid_config is None: pyramid_config = {}
    text = f'''
    Processing: {img_path}
        extract channels: {channels}
    '''
    logger.info(text)
    palom.pyramid.write_pyramid(
        mosaics,
        out_path,
        **{
            **dict(
                pixel_size=reader.pixel_size,
                kwargs_tifffile=tif_tags,
            ),
            **PYRAMID_DEFAULTS,
            **pyramid_config
        }
    )
    return out_path


def src_tif_tags(img_path):
    kwargs_tifffile = {}
    with tifffile.TiffFile(img_path) as tif:
        kwargs_tifffile.update(dict(
            photometric=tif.pages[0].photometric.value,
            resolution=tif.pages[0].resolution,
            resolutionunit=tif.pages[0].resolutionunit.value,
            software=tif.pages[0].software
        ))
    return kwargs_tifffile


def compress_pyramid(img_path, out_path=None, overwrite=False):
    img_path = pathlib.Path(img_path)
    out_path = merge_channels([img_path], out_path, overwrite)
    ome_xml = tifffile.tiffcomment(img_path)
    tifffile.tiffcomment(out_path, ome_xml.encode())
    return out_path


def fix_stitcher_ome_xml(
    img_path,
    replace_from,
    replace_to
):
    ori = tifffile.tiffcomment(img_path)
    n_to_replace = ori.count(replace_from)
    if n_to_replace == 0:
        print(f"Substring to be replaced not found in the file ({img_path})")
        return 0
    fixed = ori.replace(replace_from, replace_to)
    tifffile.tiffcomment(img_path, fixed.encode())
    print(f"{n_to_replace} instance(s) of {replace_from} replaced with {replace_to}")
    return 0


def compress_rarecyte_ome_tiff(img_path, out_path=None, overwrite=False):
    ome_path = compress_pyramid(img_path, out_path, overwrite)
    fix_stitcher_ome_xml(
        ome_path,
        '</Channel><Plane',
        '</Channel><MetadataOnly></MetadataOnly><Plane'
    )
    return


def adjust_rgb_contrast(
    in_path: str,
    in_range: tuple[float, float],
):
    import skimage.exposure

    reader = palom.reader.OmePyramidReader(in_path)
    assert reader.pyramid[0].shape[0] == 3
    assert reader.pixel_dtype == "uint8"
    assert ".ome.tif" in reader.path.name
    out_name = ".".join(reader.path.name.split(".")[:-2]) + "-adjusted.ome.tif"
    ".".join(reader.path.name.split(".")[:-2])
    out_path = reader.path.parent / out_name
    mosaic = reader.pyramid[0].map_blocks(
        lambda x: skimage.exposure.rescale_intensity(
            x, in_range=in_range, out_range="uint8"
        ).astype("uint8"),
        dtype="uint8",
    )
    palom.pyramid.write_pyramid(
        [mosaic],
        out_path,
        pixel_size=reader.pixel_size,
        downscale_factor=2,
        compression="zlib",
        save_RAM=True,
        kwargs_tifffile=dict(photometric="rgb", planarconfig="separate")
    )
    return 0


def main():
    import fire
    fire.Fire({
        'merge': merge_channels,
        'extract': extract_channels,
        'compress': compress_pyramid,
        'compress-rarecyte': compress_rarecyte_ome_tiff,
        'adjust-rgb-contrast': adjust_rgb_contrast,
    })


if __name__ == '__main__':
    import sys
    sys.exit(main())