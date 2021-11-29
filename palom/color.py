from __future__ import annotations
import numpy as np
import dask.array as da
import skimage.color
import skimage.exposure

from . import extract


class HaxProcessor:

    def __init__(
        self,
        rgb_img: da.Array,
        channel_axis: int = 0
    ) -> None:
        self.rgb_img = np.moveaxis(rgb_img, channel_axis, 2)
        assert self.rgb_img.ndim == 3, (
            f"`rgb_img` must be 3D, preferably in (C, Y, X) order"
        )
        assert self.rgb_img.shape[2] == 3
        self.make_thumbnail()

    def make_thumbnail(self):
        self.thumbnail = self.rgb_img

    def find_processed_color_contrast_range(self, mode: str):
        assert mode in ['grayscale', 'hematoxylin', 'aec']
        
        mode_range = f'_{mode}_range'
        if hasattr(self, mode_range):
            return getattr(self, mode_range)
        
        thumbnail = self.thumbnail
        if mode == 'grayscale':
            img = self.rgb2gray(thumbnail)
        elif mode == 'hematoxylin':
            img = self.rgb2hematoxylin(thumbnail)
        elif mode == 'aec':
            img = self.rgb2aec(thumbnail)

        setattr(self, mode_range, (
            img.min(), img.max()
        ))
        return getattr(self, mode_range)

    def rgb2gray(self, rgb_img):
        import cv2
        return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        # return skimage.color.rgb2gray(rgb_img).astype(np.float32)

    def rgb2aec(self, rgb_img):
        return extract.rgb2aec(rgb_img).astype(np.float32)

    def rgb2hematoxylin(self, rgb_img):
        hax = skimage.color.separate_stains(rgb_img, skimage.color.hax_from_rgb)
        return hax[..., 0].astype(np.float32)

    def get_processed_color(self, rgb_img, mode='grayscale'):
        assert mode in ['grayscale', 'hematoxylin', 'aec']

        if mode == 'grayscale':
            process_func = self.rgb2gray
        elif mode == 'hematoxylin':
            process_func = self.rgb2hematoxylin
        elif mode == 'aec':
            process_func = self.rgb2aec

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
        self.thumbnail = self.thumbnail.compute()
    
    def get_processed_color(self, level, mode='grayscale', out_dtype=None):
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
