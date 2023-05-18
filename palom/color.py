from __future__ import annotations
import numpy as np
import dask.array as da
import skimage.color
import skimage.exposure

from . import extract


class HaxProcessor:

    def __init__(
        self,
        contrast_ref_img: da.Array | np.array,
        channel_axis: int = 0,
    ) -> None:
        self.contrast_ref_img = np.moveaxis(contrast_ref_img, channel_axis, 2)
        assert self.contrast_ref_img.ndim == 3, (
            f"`contrast_ref_img` must be 3D, preferably in (C, Y, X) order"
        )

    def find_processed_color_contrast_range(self, mode: str):
        supported_modes = ['grayscale', 'hematoxylin', 'aec', 'dab']
        assert mode in supported_modes
       
        mode_range = f'_{mode}_range'
        if hasattr(self, mode_range):
            return getattr(self, mode_range)
       
        contrast_ref_img = self.contrast_img
       
        process_func = self.__getattribute__(f"rgb2{mode}")
        img = process_func(contrast_ref_img)

        setattr(self, mode_range, (
            img.min(), img.max()
        ))
        return getattr(self, mode_range)

    @property
    def contrast_img(self):
        if hasattr(self, '_contrast_img'):
            return self._contrast_img
        self._contrast_img = self.contrast_ref_img
        if isinstance(self._contrast_img, da.Array):
            self._contrast_img = self._contrast_img.compute()
        return self._contrast_img

    def rgb2grayscale(self, rgb_img):
        import cv2
        # FIXME should be cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        # return skimage.color.rgb2gray(rgb_img).astype(np.float32)

    def rgb2aec(self, rgb_img):
        return extract.rgb2aec(rgb_img).astype(np.float32)

    def rgb2dab(self, rgb_img):
        hdx = skimage.color.separate_stains(rgb_img, skimage.color.hdx_from_rgb)
        return hdx[..., 1].astype(np.float32)

    def rgb2hematoxylin(self, rgb_img):
        hax = skimage.color.separate_stains(rgb_img, skimage.color.hax_from_rgb)
        return hax[..., 0].astype(np.float32)

    def get_processed_color(self, rgb_img, mode='grayscale'):
        assert mode in ['grayscale', 'hematoxylin', 'aec', 'dab']

        process_func = self.__getattribute__(f"rgb2{mode}")
        intensity_range = self.find_processed_color_contrast_range(mode)
       
        processed = da.map_blocks(
            process_func,
            rgb_img,
            dtype=np.float32,
            drop_axis=2
        )

        return da.map_blocks(
            lambda x: skimage.exposure.rescale_intensity(
                x, in_range=intensity_range, out_range=(0, 1)
            ).astype(np.float32),
            processed,
            dtype=np.float32
        )


class PyramidHaxProcessor(HaxProcessor):

    def __init__(
        self,
        pyramid: list[da.Array],
        thumbnail_level: int = None
    ) -> None:
        if thumbnail_level is None:
            thumbnail_level = len(pyramid) - 1
        super().__init__(pyramid[thumbnail_level], channel_axis=0)
        self.pyramid = pyramid
   
    def get_processed_color(self, level, mode='grayscale', out_dtype=None):
        if mode == 'color':
            return self.pyramid[level]
        rgb_img = np.moveaxis(self.pyramid[level], 0, 2)
        processed = super().get_processed_color(rgb_img, mode=mode)
        if out_dtype is None:
            out_dtype = rgb_img.dtype
        if isinstance(out_dtype, np.dtype):
            out_dtype = out_dtype.type
        return da.map_blocks(
            lambda x: skimage.exposure.rescale_intensity(
                x, in_range=(0, 1), out_range=out_dtype
            ).astype(out_dtype),
            processed,
            dtype=out_dtype
        )
