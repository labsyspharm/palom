import math
import tifffile
import tqdm
import numpy as np
import skimage.transform
from loguru import logger

from . import __version__ as _version


class PyramidSetting:

    def __init__(
        self,
        downscale_factor=2,
        tile_size=1024,
        max_pyramid_img_size=1024
    ):
        self.downscale_factor = downscale_factor
        self.tile_size = tile_size
        self.max_pyramid_img_size = max_pyramid_img_size

    def tile_shapes(self, base_shape):
        shapes = np.array(self.pyramid_shapes(base_shape))
        n_rows_n_cols = np.ceil(shapes / self.tile_size)
        tile_shapes = np.ceil(shapes / n_rows_n_cols / 16) * 16
        return [tuple(map(int, s)) for s in tile_shapes]

    def pyramid_shapes(self, base_shape):
        num_levels = self.num_levels(base_shape)
        factors = self.downscale_factor ** np.arange(num_levels)
        shapes = np.ceil(np.array(base_shape) / factors[:,None])
        return [tuple(map(int, s)) for s in shapes]

    def num_levels(self, base_shape):
        factor = max(base_shape) / self.max_pyramid_img_size
        return math.ceil(math.log(factor, self.downscale_factor)) + 1


def format_channel_names(num_channels_each_mosaic, channel_names):
    '''
    format_channel_names(
        [1, 2, 3, 4, 5], ['x', 'x', ['x'], ['x', 'x']]
    )
    >>> [
        'x_1',
        'x_2',
        'x_3',
        'x_4',
        'x_5',
        'x_6',
        'Mosaic 4_1',
        'Mosaic 4_2',
        'Mosaic 4_3',
        'Mosaic 4_4'
    ]
    '''
    matched_channel_names = []
    for idx, (n, c) in enumerate(
        zip(channel_names, num_channels_each_mosaic)
    ):
        nl = n
        if type(n) == str:
            nl = [n] * c
        if len(nl) == 1:
            nl = nl * c
        if len(nl) != c:
            nl = [f"Mosaic {idx+1}"] * c
        matched_channel_names.append(nl)
    return make_unique_str(
        [n for l in matched_channel_names for n in l]
    )


def make_unique_str(str_list):
    if len(set(str_list)) == len(str_list):
        return str_list
    else:
        max_length = max([len(s) for s in str_list])
        str_np = np.array(str_list, dtype=np.dtype(('U', max_length+10)))
        unique, counts = np.unique(str_np, return_counts=True)
        has_duplicate = unique[counts > 1]
        for n in has_duplicate:
            suffixes = [
                f"_{i}"
                for i in range(1, (str_np == n).sum()+1)
            ]
            str_np[str_np == n] = np.char.add(n, suffixes)
    return make_unique_str(list(str_np))


def normalize_mosaics(mosaics):
    dtypes = set(m.dtype for m in mosaics)
    if any([np.issubdtype(d, np.floating) for d in dtypes]):
        max_dtype = np.dtype(np.float32)
    else:
        max_dtype = max(dtypes)
    normalized = []
    for m in mosaics:
        assert m.ndim == 2 or m.ndim == 3
        if m.ndim == 2:
            m = m[np.newaxis, :]
        normalized.append(m.astype(max_dtype, copy=False))
    return normalized


def write_pyramid(
    mosaics,
    output_path,
    pixel_size=1,
    channel_names=None,
    verbose=True,
    downscale_factor=4
):
    ref_m = mosaics[0]
    path = output_path
    num_channels = count_num_channels(mosaics)
    base_shape = ref_m.shape[1:3]
    assert int(downscale_factor) == downscale_factor
    assert downscale_factor < min(base_shape)
    pyramid_setting = PyramidSetting(
        downscale_factor=int(downscale_factor),
        tile_size=max(ref_m.chunksize)
    )
    num_levels = pyramid_setting.num_levels(base_shape)
    tile_shapes = pyramid_setting.tile_shapes(base_shape)
    shapes = pyramid_setting.pyramid_shapes(base_shape)

    dtype = ref_m.dtype

    software = f'palom {_version}'
    pixel_size = pixel_size
    metadata = {
        'Creator': software,
        'Pixels': {
            'PhysicalSizeX': pixel_size, 'PhysicalSizeXUnit': '\u00b5m',
            'PhysicalSizeY': pixel_size, 'PhysicalSizeYUnit': '\u00b5m'
        },
    }

    if channel_names is not None:
        num_channels_each_mosaic = [
            count_num_channels([m])
            for m in mosaics
        ]
        names = format_channel_names(num_channels_each_mosaic, channel_names)
        if len(names) == num_channels:
            metadata.update({
                'Channel': {'Name': names},
            })

    logger.info(f"Writing to {path}")
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        tif.write(
            data=tile_from_combined_mosaics(
                mosaics, tile_shape=tile_shapes[0]
            ),
            metadata=metadata,
            software=software,
            shape=(num_channels, *shapes[0]),
            subifds=int(num_levels - 1),
            dtype=dtype,
            tile=tile_shapes[0]
        )
        logger.info('Generating pyramid')
        for level, (shape, tile_shape) in enumerate(
            zip(shapes[1:], tile_shapes[1:])
        ):
            if verbose:
                logger.info(f"    Level {level+1} ({shape[0]} x {shape[1]})")
            tif.write(
                data=tile_from_pyramid(
                    path,
                    num_channels,
                    tile_shape=tile_shape,
                    downscale_factor=downscale_factor,
                    level=level,
                ),
                shape=(num_channels, *shape),
                subfiletype=1,
                dtype=dtype,
                tile=tile_shape
            )


def count_num_channels(imgs):
    for img in imgs:
        assert img.ndim == 2 or img.ndim == 3
    return sum([
        1 if img.ndim == 2 else img.shape[0]
        for img in imgs
    ])
    

def tile_from_combined_mosaics(mosaics, tile_shape):
    num_rows, num_cols = mosaics[0].shape[1:3]
    h, w = tile_shape
    n = len(mosaics)
    for idx, m in enumerate(mosaics):
        for cidx, c in enumerate(m):
            # the performance is heavily degraded without pre-computing the
            # mosaic channel
            with tqdm.dask.TqdmCallback(
                ascii=True,
                desc=(
                    f"Assembling mosaic {idx+1:2}/{n:2} (channel"
                    f" {cidx+1:2}/{m.shape[0]:2})"
                ),
            ):
                c = c.compute()
            for y in range(0, num_rows, h):
                for x in range(0, num_cols, w):
                    yield np.array(c[y:y+h, x:x+w])
                # yield m[y:y+h, x:x+w].copy().compute()


def tile_from_pyramid(
    path,
    num_channels,
    tile_shape,
    downscale_factor=2,
    level=0
):
    h, w = tile_shape
    for c in tqdm.trange(
            num_channels,
            ascii=True, desc=f'Processing channel'
        ):
        img = tifffile.imread(
            path, is_ome=False, series=0, key=c, level=level
        )
        img = skimage.transform.downscale_local_mean(
            img, (downscale_factor, downscale_factor)
        ).astype(img.dtype)
        num_rows, num_columns = img.shape
        for y in range(0, num_rows, h):
            for x in range(0, num_columns, w):
                yield np.array(img[y:y+h, x:x+w])
