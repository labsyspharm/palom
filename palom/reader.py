from __future__ import annotations

import pathlib

import dask.array as da
import numpy as np
import ome_types
import pint
import tifffile
import zarr
from loguru import logger

from . import pyramid as pyramid_util


class DaPyramidChannelReader:

    def __init__(
        self,
        pyramid: list[da.Array],
        channel_axis: int
    ) -> None:
        self.pyramid = pyramid
        self.channel_axis = channel_axis
        if self.validate_pyramid(self.pyramid, self.channel_axis):
            self.pyramid = self.normalize_axis_order()
            self.pyramid = self.auto_format_pyramid(self.pyramid)

    @staticmethod
    def validate_pyramid(pyramid: list[da.Array], channel_axis:int) -> bool:
        for i, level in enumerate(pyramid):
            assert level.ndim == 3, ''
            if np.argmin(level.shape) != channel_axis:
                logger.warning(
                    f"level {i} has shape of {level.shape} while given"
                    f" `channel_axis` is {channel_axis}"
                )
        return True
   
    def normalize_axis_order(self):
        if self.channel_axis == 0:
            return self.pyramid
        return [
            da.moveaxis(level, self.channel_axis, 0)
            for level in self.pyramid
        ]

    def read_level_channels(
        self,
        level: int,
        channels: int | list[int]
    ) -> da.Array:
        target_level = self.pyramid[level]
        return target_level[channels]

    @staticmethod
    def auto_format_pyramid(
        pyramid: list[da.Array],
    ) -> list[da.Array]:
        first = pyramid[0]
        if len(pyramid) > 1: return pyramid
        # Assumption: if the image is pyramidal, it must also be tiled
        if max(first.shape) < 1024: return pyramid
        logger.warning(
                f'Unable to detect pyramid levels, it may take a while'
                f' to compute thumbnails during coarse alignment'
            )
        if first.numblocks[1:3] == (1, 1):
            first = first.rechunk((1, 1024, 1024))
        pyramid_setting = pyramid_util.PyramidSetting(downscale_factor=2)
        num_levels = pyramid_setting.num_levels(first.shape[1:3])
        return [
            da.coarsen(
                np.mean,
                first,
                {0:1, 1:2**i, 2:2**i},
                trim_excess=True
            ).astype(first.dtype)
            for i in range(num_levels)
        ]

    @property
    def level_downsamples(self) -> dict[int, int]:
        return {
            i: round(self.pyramid[0].shape[1] / level.shape[1])
            for i, level in enumerate(self.pyramid)
        }
   
    @property
    def pixel_dtype(self) -> np.dtype:
        return self.pyramid[0].dtype

    def get_thumbnail_level_of_size(self, size: float) -> int:
        shapes = [
            np.abs(np.mean(level.shape[1:3]) - size)
            for level in self.pyramid
        ]
        return np.argmin(shapes)


class OmePyramidReader(DaPyramidChannelReader):

    def __init__(
        self,
        path: str | pathlib.Path,
        pixel_size: float | None = None
    ) -> None:
        self.path = pathlib.Path(path)
        pyramid = self.pyramid_from_ometiff(self.path)
        channel_axis = 0
        self._pixel_size = pixel_size
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['pyramid']
        state['path'] = state['path'].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state['path'], pixel_size=state['_pixel_size'])

    @staticmethod
    def pyramid_from_ometiff(path: str | pathlib.Path) -> list[da.Array]:
        with tifffile.TiffFile(path) as tif:
            num_series = len(tif.series)
            if num_series == 1:
                pyramid = tif.series[0].levels
            elif num_series > 1:
                pyramid = tif.series
            zarr_pyramid = [
                zarr.open(level.aszarr(), 'r')
                for level in pyramid
            ]
            da_pyramid = []
            for z in zarr_pyramid:
                if issubclass(type(z), zarr.hierarchy.Group):
                    da_level = da.from_zarr(z[0])
                else:
                    da_level = da.from_zarr(z)
                if da_level.ndim == 2:
                    da_level = da_level.reshape(1, *da_level.shape)
                if da_level.ndim == 3:
                    if da_level.shape[2] in (3, 4):
                        da_level = da.moveaxis(da_level, 2, 0)
                da_pyramid.append(da_level)
            return da_pyramid

    @property
    def pixel_size(self) -> float:
        if self._pixel_size is not None:
            return self._pixel_size
        try:
            # ome-types v0.4 does not have `parser` kwarg in `from_tiff`
            import inspect
            kwargs = dict(path=self.path, validate=False)
            keys = inspect.signature(ome_types.from_tiff).parameters
            if 'parser' in keys:
                kwargs.update(dict(parser='lxml'))
            ome = ome_types.from_tiff(**kwargs)
            px_size = ome.images[0].pixels.physical_size_x
            # convert length unit to µm
            unit = ome.images[0].pixels.physical_size_x_unit.value
            ureg = pint.UnitRegistry()
            px_size_micron = px_size * ureg(unit).to(ureg.micron).magnitude
            logger.info(
                f"Detected pixel size: {px_size_micron:.4f} µm"
            )
            self._pixel_size = px_size_micron
            return self._pixel_size
        except Exception:
            logger.warning(
                f'Unable to parse pixel size from {self.path.name};'
                f' assuming 1 µm. Use `_pixel_size` to set it manually'
            )
            self._pixel_size = 1
            return self._pixel_size


class SvsReader(DaPyramidChannelReader):

    def __init__(
        self,
        path: str | pathlib.Path,
        pixel_size: float | None = None
    ) -> None:
        # FIXME maybe move napari_lazy_openslide to optional dependency?
        # https://python-poetry.org/docs/pyproject/#extras
        # https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/readers/bioformats_reader.py#L33-L40
        from napari_lazy_openslide import OpenSlideStore
       
        self.path = pathlib.Path(path)
        self.store = OpenSlideStore(str(self.path))
        self.zarr = zarr.open(self.store, mode='r')
        self._pixel_size = pixel_size
        pyramid = self.pyramid_from_svs()
        channel_axis = 2
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['pyramid'], state['store'], state['zarr']
        state['path'] = state['path'].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state['path'], pixel_size=state['_pixel_size'])

    def pyramid_from_svs(self) -> list[da.Array]:
        return [
            da.from_zarr(self.store, component=d['path'])[..., :3]
            for d in self.zarr.attrs['multiscales'][0]['datasets']
        ]
   
    @property
    def pixel_size(self):
        if self._pixel_size is not None:
            return self._pixel_size
        try:
            return float(self.store._slide.properties['openslide.mpp-x'])
        except Exception:
            logger.warning(
                f'Unable to parse pixel size from {self.path.name};'
                f' assuming 1 µm. Use `_pixel_size` to set it manually'
            )
            self._pixel_size = 1
            return self._pixel_size
