# based on https://github.com/manzt/napari-lazy-openslide/blob/7b6656f6338260072a2fc06512cfe3ab54731e18/napari_lazy_openslide/store.py

import re
from ctypes import ArgumentError
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np
import slideio
from zarr.storage import BaseStore, _path_to_prefix, attrs_key, init_array, init_group
from zarr.util import json_dumps, json_loads, normalize_storage_path


def init_attrs(store: MutableMapping, attrs: Mapping[str, Any], path: str = None):
    path = normalize_storage_path(path)
    path = _path_to_prefix(path)
    store[path + attrs_key] = json_dumps(attrs)


def create_meta_store(scene: slideio.Scene) -> Dict[str, bytes]:
    """Creates a dict containing the zarr metadata for the multiscale openslide image."""
    level_info = _parse_level_info(scene=scene)
    store = dict()
    root_attrs = {
        "multiscales": [
            {
                "name": Path(scene.file_path).name,
                "datasets": [{"path": str(i)} for i in range(len(level_info))],
                "version": "0.1",
                # "metadata": json_loads(slide.raw_metadata)
            }
        ]
    }
    init_group(store)
    init_attrs(store, root_attrs)
    for i, info in enumerate(level_info):
        init_array(
            store,
            path=str(i),
            shape=(*info["shape"], scene.num_channels),
            chunks=(*info["tile_size"], scene.num_channels),
            dtype=scene.get_channel_data_type(0),
            compressor=None,
        )
    return store


def _parse_chunk_path(path: str):
    """Returns x,y chunk coords and pyramid level from string key"""
    level, ckey = path.split("/")
    y, x, _ = map(int, ckey.split("."))
    return x, y, int(level)


def _parse_level_info(scene: slideio.Scene):
    import itertools

    levels = range(scene.num_zoom_levels)
    level_info = []
    for ll in levels:
        ii = {}
        info = scene.get_zoom_level_info(ll)
        ii["shape"] = (info.size.height, info.size.width)
        ii["tile_size"] = (info.tile_size.height, info.tile_size.width)
        ii["downsample"] = 1
        level_info.append(ii)
    for aa, bb in itertools.pairwise(level_info):
        bb["downsample"] = aa["downsample"] * round(aa["shape"][0] / bb["shape"][0])
    return level_info


def slideio_vsi_patch(path: str):
    path = Path(path)
    assert path.name.endswith(".vsi")
    data_dir = path.parent / f'_{path.name.replace(".vsi", "")}_'
    assert data_dir.exists()
    ets_files = sorted(data_dir.rglob("*.ets"))
    assert len(ets_files)
    temp_files = [ff.replace(ff.parent / f"{ff.name}_") for ff in ets_files]
    for tt, ee in zip(temp_files, ets_files):
        tt.replace(ee)
        try:
            slide = slideio.open_slide(str(path), "VSI")
        except RuntimeError:
            ee.replace(tt)
    for tt, ee in zip(temp_files, ets_files):
        if tt.exists():
            tt.replace(ee)
    return slide


def _parse_pixel_size(slide: slideio.Slide):
    metadata = slide.raw_metadata
    pattern = r'"Physical pixel size","value":"\(([\d.]+)'
    found = re.findall(pattern=pattern, string=metadata)
    if not found:
        return 1.0
    return float(found[0])


class SlideIoVsiStore(BaseStore):
    """Wraps an OpenSlide object as a multiscale Zarr Store.

    Parameters
    ----------
    path: str
        The file to open with OpenSlide.
    tilesize: int
        Desired "chunk" size for zarr store.
    """

    def __init__(self, path: str, scene: int = 0):
        # self._slide = slideio.Slide(path, driver="VSI")
        self._slide = slideio_vsi_patch(path)
        self._scene = self._slide.get_scene(scene)
        self._level_info = _parse_level_info(self._scene)
        self._store = create_meta_store(self._scene)

    def __getitem__(self, key: str):
        if key in self._store:
            # key is for metadata
            return self._store[key]

        # key should now be a path to an array chunk
        # e.g '3/4.5.0' -> '<level>/<chunk_key>'
        try:
            x, y, level = _parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = self._level_info[level]["tile_size"]
            tile = self._scene.read_block(location, size)
        except ArgumentError as err:
            # Can occur if trying to read a closed slide
            raise err
        except Exception:
            # TODO: probably need better error handling.
            # If anything goes wrong, we just signal the chunk
            # is missing from the store.
            raise KeyError(key)

        return np.array(tile).tobytes()

    def __contains__(self, key: str):
        return key in self._store

    def __eq__(self, other):
        return (
            isinstance(other, SlideIoVsiStore)
            and self._slide.file_path == other._slide.file_path
        )

    def __setitem__(self, key, val):
        raise RuntimeError("__setitem__ not implemented")

    def __delitem__(self, key):
        raise RuntimeError("__setitem__ not implemented")

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(1 for _ in self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _ref_pos(self, x: int, y: int, level: int):
        dsample = self._level_info[level]["downsample"]
        ty, tx = self._level_info[level]["tile_size"]
        xref = int(x * dsample * tx)
        yref = int(y * dsample * ty)
        return xref, yref, tx * dsample, ty * dsample

    def keys(self):
        return self._store.keys()

    def close(self):
        self._scene = None
        self._slide = None


if __name__ == "__main__":
    import sys

    store = SlideIoVsiStore(sys.argv[1])
