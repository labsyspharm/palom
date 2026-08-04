from __future__ import annotations

import itertools
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
    def __init__(self, pyramid: list[da.Array], channel_axis: int) -> None:
        self.pyramid = pyramid
        self.channel_axis = channel_axis
        self._pixel_size_assumed = False
        if self.validate_pyramid(self.pyramid, self.channel_axis):
            self.pyramid = self.normalize_axis_order()
            self.pyramid = self.auto_format_pyramid(self.pyramid)

    def _assume_pixel_size(self) -> float:
        """Fall back to a placeholder pixel size, recording that it is a guess."""
        logger.warning(
            f"Unable to parse pixel size from {self.path.name};"
            f" assuming 1 µm. Pass `pixel_size=` to set it manually"
        )
        self._pixel_size_assumed = True
        self._pixel_size = 1
        return self._pixel_size

    @property
    def has_pixel_size(self) -> bool:
        """Whether `pixel_size` is real (parsed from metadata, or supplied by the
        caller) rather than the 1 µm placeholder.

        Anything converting a physical length (µm) into pixels must check this
        first. A placeholder pixel size is not a harmless approximation: it is
        wrong by the true pixel size -- 3x on a 0.325 µm scan -- so every derived
        physical quantity is silently off by that factor. Prefer a
        resolution-independent fallback, or skip the physical reasoning
        altogether, over trusting the placeholder.
        """
        self.pixel_size  # resolve lazily; the fallback path sets the flag
        return not self._pixel_size_assumed

    @staticmethod
    def validate_pyramid(pyramid: list[da.Array], channel_axis: int) -> bool:
        for i, level in enumerate(pyramid):
            assert level.ndim == 3
            if np.argmin(level.shape) != channel_axis:
                logger.warning(
                    f"level {i} has shape of {level.shape} while given"
                    f" `channel_axis` is {channel_axis}"
                )
        return True

    def normalize_axis_order(self):
        if self.channel_axis == 0:
            return self.pyramid
        return [da.moveaxis(level, self.channel_axis, 0) for level in self.pyramid]

    def read_level_channels(self, level: int, channels: int | list[int]) -> da.Array:
        target_level = self.pyramid[level]
        return target_level[channels]

    @staticmethod
    def auto_format_pyramid(
        pyramid: list[da.Array],
    ) -> list[da.Array]:
        first = pyramid[0]
        if len(pyramid) > 1:
            return pyramid
        # Assumption: if the image is pyramidal, it must also be tiled
        if max(first.shape) < 1024:
            return pyramid
        logger.warning(
            "Unable to detect pyramid levels, it may take a while"
            " to compute thumbnails during coarse alignment"
        )
        if first.numblocks[1:3] == (1, 1):
            first = first.rechunk((1, 1024, 1024))
        pyramid_setting = pyramid_util.PyramidSetting(downscale_factor=2)
        num_levels = pyramid_setting.num_levels(first.shape[1:3])
        return [
            da.coarsen(
                np.mean, first, {0: 1, 1: 2**i, 2: 2**i}, trim_excess=True
            ).astype(first.dtype)
            for i in range(num_levels)
        ]

    @property
    def level_downsamples(self) -> dict[int, float]:
        shapes = [np.array(ss.shape[1:3], dtype=float) for ss in self.pyramid]
        # Per-step downscale factor between consecutive levels. The raw shape
        # ratio only approximates the true factor because each level's
        # dimensions are rounded -- up *or* down -- when shrunk (e.g. an odd
        # 3671 px level halves to 1836 px, a ratio of 1.9995). If rounding the
        # ratio to an integer reproduces the actual level shape to within a
        # pixel on every axis, treat it as an exact integer downscale: this
        # covers both floor- and ceil-built pyramids and stops the accumulated
        # factor from drifting (128 instead of 127.86 by level 7). Otherwise
        # keep the measured ratio, which preserves genuinely non-integer and
        # non-constant (e.g. magnification-matched) factors.
        #
        # Caveat: integer rounding and a true non-integer factor cannot be told
        # apart from shape alone at coarse levels, where a single pixel is a
        # large fraction of the dimension; the test is reliable at fine levels
        # where the shapes are large, which is what matters for registration.
        # FIXME image-based registration would refine factors that aren't exact
        # integers.
        factors = [1.0]
        for level, (prev, cur) in enumerate(itertools.pairwise(shapes), start=1):
            ratio = float((prev / cur).mean())
            factor = round(ratio)
            if factor >= 1 and np.all(np.abs(prev / factor - cur) < 1):
                factors.append(float(factor))
            else:
                logger.warning(
                    f"level {level} has a non-integer downsample factor of"
                    f" {ratio:.4f} relative to level {level - 1}; coarse"
                    f" alignment may be slightly less accurate"
                )
                factors.append(ratio)
        return dict(enumerate(itertools.accumulate(factors, func=np.multiply)))

    @property
    def pixel_dtype(self) -> np.dtype:
        return self.pyramid[0].dtype

    def get_thumbnail_level_of_size(self, size: float) -> int:
        shapes = [np.abs(np.mean(level.shape[1:3]) - size) for level in self.pyramid]
        return np.argmin(shapes)


class OmePyramidReader(DaPyramidChannelReader):
    def __init__(
        self, path: str | pathlib.Path, pixel_size: float | None = None
    ) -> None:
        self.path = pathlib.Path(path)
        pyramid = self.pyramid_from_ometiff(self.path)
        channel_axis = 0
        self._pixel_size = pixel_size
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pyramid"]
        state["path"] = state["path"].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state["path"], pixel_size=state["_pixel_size"])

    @staticmethod
    def pyramid_from_ometiff(path: str | pathlib.Path) -> list[da.Array]:
        with tifffile.TiffFile(path) as tif:
            num_series = len(tif.series)
            if num_series == 1:
                pyramid = tif.series[0].levels
            elif num_series > 1:
                pyramid = tif.series
            zarr_pyramid = [zarr.open(level.aszarr(), "r") for level in pyramid]
            da_pyramid = []
            for z in zarr_pyramid:
                if issubclass(type(z), zarr.hierarchy.Group):
                    da_level = da.from_zarr(z[0], name=False)
                else:
                    da_level = da.from_zarr(z, name=False)
                da_level = da_level.squeeze()
                if da_level.ndim == 2:
                    da_level = da_level.reshape(1, *da_level.shape)
                elif da_level.ndim == 3:
                    if da_level.shape[2] in (3, 4):
                        da_level = da.moveaxis(da_level, 2, 0)
                else:
                    raise ValueError(
                        f"Image with {da_level.ndim} dimension {da_level.shape} is not supported"
                    )
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
            if "parser" in keys:
                kwargs.update(dict(parser="lxml"))
            ome = ome_types.from_tiff(**kwargs)
            px_size = ome.images[0].pixels.physical_size_x
            # convert length unit to µm
            unit = ome.images[0].pixels.physical_size_x_unit.value
            ureg = pint.UnitRegistry()
            px_size_micron = px_size * ureg(unit).to(ureg.micron).magnitude
            logger.info(f"Detected pixel size: {px_size_micron:.4f} µm")
            self._pixel_size = px_size_micron
            return self._pixel_size
        except Exception:
            return self._assume_pixel_size()


class SvsReader(DaPyramidChannelReader):
    def __init__(
        self, path: str | pathlib.Path, pixel_size: float | None = None
    ) -> None:
        # FIXME maybe move napari_lazy_openslide to optional dependency?
        # https://python-poetry.org/docs/pyproject/#extras
        # https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/readers/bioformats_reader.py#L33-L40
        from . import openslide_store

        self.path = pathlib.Path(path)
        self.store = openslide_store.OpenSlideStore(str(self.path))
        self.zarr = zarr.open(self.store, mode="r")
        self._pixel_size = pixel_size
        pyramid = self.pyramid_from_svs()
        channel_axis = 2
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pyramid"], state["store"], state["zarr"]
        state["path"] = state["path"].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state["path"], pixel_size=state["_pixel_size"])

    def pyramid_from_svs(self) -> list[da.Array]:
        return [
            da.from_zarr(self.store, component=d["path"], name=False)[..., :3]
            for d in self.zarr.attrs["multiscales"][0]["datasets"]
        ]

    @property
    def pixel_size(self):
        if self._pixel_size is not None:
            return self._pixel_size
        try:
            return float(self.store._slide.properties["openslide.mpp-x"])
        except Exception:
            return self._assume_pixel_size()


class QptiffPyramidReader(DaPyramidChannelReader):
    def __init__(
        self, path: str | pathlib.Path, pixel_size: float | None = None
    ) -> None:
        self.path = pathlib.Path(path)
        self._pixel_size = pixel_size
        pyramid, channel_names, detected_pixel_size = self._parse_qptiff(self.path)
        self._channel_names = channel_names
        if self._pixel_size is None and detected_pixel_size is not None:
            logger.info(f"Detected pixel size: {detected_pixel_size:.4f} µm")
            self._pixel_size = detected_pixel_size
        super().__init__(pyramid, channel_axis=0)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pyramid"]
        state["path"] = state["path"].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(path=state["path"], pixel_size=state["_pixel_size"])

    @staticmethod
    def _parse_qptiff(
        path: str | pathlib.Path,
    ) -> tuple[list[da.Array], list[str], float | None]:
        import xml.etree.ElementTree as ET

        with tifffile.TiffFile(path) as tif:
            # Series 0 is always the full-resolution (Baseline/FullResolution) series
            series = tif.series[0]
            pyramid_levels = series.levels

            zarr_pyramid = [zarr.open(level.aszarr(), "r") for level in pyramid_levels]
            da_pyramid = []
            for z in zarr_pyramid:
                if issubclass(type(z), zarr.hierarchy.Group):
                    da_level = da.from_zarr(z[0], name=False)
                else:
                    da_level = da.from_zarr(z, name=False)
                da_level = da_level.squeeze()
                if da_level.ndim == 2:
                    da_level = da_level.reshape(1, *da_level.shape)
                da_pyramid.append(da_level)

            # Each page in level 0 is one channel; parse Biomarker from its XML
            channel_names = []
            for page in pyramid_levels[0].pages:
                desc_tag = page.tags.get("ImageDescription")
                if desc_tag:
                    root = ET.fromstring(desc_tag.value)
                    name = root.findtext("Biomarker") or root.findtext("Name") or ""
                    channel_names.append(name)
                else:
                    channel_names.append("")

            # Pixel size from standard TIFF XResolution / ResolutionUnit tags
            pixel_size: float | None = None
            try:
                page0 = pyramid_levels[0].pages[0]
                xres_tag = page0.tags.get("XResolution")
                resunit_tag = page0.tags.get("ResolutionUnit")
                if xres_tag is not None and resunit_tag is not None:
                    num, denom = xres_tag.value
                    resunit = str(resunit_tag.value).upper()
                    if num != 0:
                        if "CENTIMETER" in resunit:
                            pixel_size = denom / num * 1e4  # cm → µm
                        elif "INCH" in resunit:
                            pixel_size = denom / num * 25400  # inch → µm
            except Exception:
                pass

        return da_pyramid, channel_names, pixel_size

    @property
    def pixel_size(self) -> float:
        if self._pixel_size is not None:
            return self._pixel_size
        return self._assume_pixel_size()

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names


class VsiReader(DaPyramidChannelReader):
    def __init__(
        self, path: str | pathlib.Path, scene: int = 0, pixel_size: float | None = None
    ) -> None:
        from . import slideio_store

        self.path = pathlib.Path(path)
        self.scene = scene
        self.store = slideio_store.SlideIoVsiStore(str(self.path), scene=self.scene)
        self.zarr = zarr.open(self.store, mode="r")
        self._pixel_size = pixel_size
        pyramid = self.pyramid_from_vsi()
        channel_axis = 2
        super().__init__(pyramid, channel_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pyramid"], state["store"], state["zarr"]
        state["path"] = state["path"].resolve()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__(
            path=state["path"], scene=state["scene"], pixel_size=state["_pixel_size"]
        )

    def pyramid_from_vsi(self) -> list[da.Array]:
        return [
            da.from_zarr(self.store, component=d["path"], name=False)
            for d in self.zarr.attrs["multiscales"][0]["datasets"]
        ]

    @property
    def pixel_size(self):
        from . import slideio_store

        if self._pixel_size is not None:
            return self._pixel_size
        px_size = slideio_store._parse_pixel_size(self.store._slide)
        if px_size == 1.0:
            # `_parse_pixel_size` returns 1.0 both when it fails and (in
            # principle) for a real 1 µm scan; treat it as a guess
            return self._assume_pixel_size()
        self._pixel_size = px_size
        return self._pixel_size
