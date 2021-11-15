import pathlib
from napari_lazy_openslide import OpenSlideStore
import zarr
import dask.array as da
import numpy as np
import skimage.color
import skimage.exposure

from loguru import logger

from . import extract


class SvsReader:
    
    def __init__(self, path) -> None:
        self.path = pathlib.Path(path)
        self.store = OpenSlideStore(str(self.path))
        self.zarr = zarr.open(self.store, mode='r')

    @property
    def pixel_size(self):
        try:
            return float(self.store._slide.properties['openslide.mpp-x'])
        except:
            logger.warning(
                f'Unable to parse pixel size from {self.path.name};'
                f' assuming 1 µm'
            )
            return 1

    @property
    def pixel_dtype(self):
        return self.pyramid_color[0].dtype

    @property
    def level_downsamples(self):
        return {
            i: int(np.round(d))
            for i, d in enumerate(self.store._slide.level_downsamples)
        }

    @property
    def pyramid_color(self):
        return [da.from_zarr(self.store, component=d['path'])[..., :3]
            for d in
            self.zarr.attrs['multiscales'][0]['datasets']
        ]

    def find_processed_color_contrast_range(self, mode):
        assert mode in ['grayscale', 'hematoxylin', 'aec']
        
        mode_range = f'_{mode}_range'
        if hasattr(self, mode_range):
            return getattr(self, mode_range)
        
        thumbnail = self.pyramid_color[-1].compute()
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

    def get_processed_color(self, level, mode='grayscale'):
        assert mode in ['grayscale', 'hematoxylin', 'aec']
        target = self.pyramid_color[level]

        if mode == 'grayscale':
            process_func = self.rgb2gray
        elif mode == 'hematoxylin':
            process_func = self.rgb2hematoxylin
        elif mode == 'aec':
            process_func = self.rgb2aec

        intensity_range = self.find_processed_color_contrast_range(mode)
        
        processed = da.map_blocks(
            process_func,
            target,
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

    def read_image(self, channels, mode, level): 
        if channels is not None: 
            mode = 'multi-channel' 
            return self.pyramid_color[level][..., channels] 
        else: 
            return self.get_processed_color(level, mode) 
 
    def read_block(self, block_idx, channels, mode, level): 
        full_img = self.read_image(channels, mode, level) 
        assert block_idx < full_img.npartitions 
        idx = np.unravel_index(block_idx, full_img.numblocks) 
        return full_img[idx] 