# based on https://github.com/manzt/napari-lazy-openslide/blob/7b6656f6338260072a2fc06512cfe3ab54731e18/napari_lazy_openslide/store.py

import asyncio
import json
import re
from ctypes import ArgumentError
from pathlib import Path

import numpy as np
import slideio
import zarr

# See palom/openslide_store.py for why the store base class is selected by the
# installed zarr version. Metadata is emitted in zarr v2 format for both.
_ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3


def create_meta_store(scene: slideio.Scene, tilesize: int) -> dict:
    """Creates a dict of zarr v2 metadata (as bytes) for the multiscale image."""
    level_info = _parse_level_info(scene=scene)
    store = {}
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
    store[".zgroup"] = json.dumps({"zarr_format": 2}).encode()
    store[".zattrs"] = json.dumps(root_attrs).encode()
    dtype = np.dtype(scene.get_channel_data_type(0)).str
    tilesize = int(tilesize)
    num_channels = int(scene.num_channels)
    for i, info in enumerate(level_info):
        store[f"{i}/.zarray"] = json.dumps(
            {
                "zarr_format": 2,
                "shape": [int(info["shape"][0]), int(info["shape"][1]), num_channels],
                "chunks": [tilesize, tilesize, num_channels],
                "dtype": dtype,
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            }
        ).encode()
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


def _parse_pixel_size(slide: slideio.Slide):
    metadata = slide.raw_metadata
    name = "Physical pixel size"
    found = find_objects_with_name(metadata, name)
    if not found:
        return 1.0
    return eval(found[0].get("value", "(1.0, 1.0)"))[0]


def find_objects_with_name(json_text, name):
    """
    Finds and extracts JSON objects with a specified "name" property.

    Args:
        json_text (str): The JSON content as a string.
        name (str): The value of the "name" property to search for.

    Returns:
        list: A list of matching JSON objects as Python dictionaries.
    """
    import json

    # Regex pattern to match objects with the specified "name"
    pattern = rf'{{[^{{]*?"name":\s*"{re.escape(name)}".*?}}'

    # Find all matches
    matches = re.findall(pattern, json_text)

    # Convert matches to Python dictionaries
    objects = [json.loads(match) for match in matches]

    return objects


class _SlideIoVsiStoreBase:
    """Scene-backed metadata + tile reading shared by the v2 and v3 stores.

    Parameters
    ----------
    path: str
        The file to open with slideio.
    scene: int
        Selected scene where full resolution pyramid data is stored.
    tilesize: int
        Desired "chunk" size for zarr store.
    """

    def _init_source(self, path: str, scene: int, tilesize: int) -> None:
        self._slide = slideio.Slide(path, driver="VSI")
        self._scene = self._slide.get_scene(scene)
        self._level_info = _parse_level_info(self._scene)
        self._tilesize = self._optimize_tile_size(tilesize)
        self._store = create_meta_store(self._scene, self._tilesize)

    def _ref_pos(self, x: int, y: int, level: int):
        dsample = self._level_info[level]["downsample"]
        ty, tx = (self._tilesize, self._tilesize)
        xref = int(x * dsample * tx)
        yref = int(y * dsample * ty)
        return xref, yref, tx * dsample, ty * dsample

    def _optimize_tile_size(self, tilesize):
        vsi_tile_size = self._level_info[0]["tile_size"]
        optimized_tile_size = np.ceil(
            np.divide(tilesize, vsi_tile_size).max()
        ) * np.max(vsi_tile_size)
        optimized_tile_size = optimized_tile_size.astype("int")
        if tilesize != optimized_tile_size:
            import logging

            logging.warning(
                f"Adjust tile size to {optimized_tile_size} (was {tilesize})"
            )
        return optimized_tile_size

    def read_tile_bytes(self, key: str) -> bytes:
        """Return raw bytes for a metadata or chunk key, else raise KeyError."""
        if key in self._store:
            # key is for metadata
            return self._store[key]

        # key should now be a path to an array chunk
        # e.g '3/4.5.0' -> '<level>/<chunk_key>'
        try:
            x, y, level = _parse_chunk_path(key)
            location = self._ref_pos(x, y, level)
            size = (self._tilesize, self._tilesize)
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

    def __eq__(self, other):
        return (
            isinstance(other, SlideIoVsiStore)
            and self._slide.file_path == other._slide.file_path
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._scene = None
        self._slide = None


if _ZARR_V3:
    from zarr.abc.store import (
        ByteRequest,
        OffsetByteRequest,
        RangeByteRequest,
        Store,
        SuffixByteRequest,
    )

    def _slice_value(value: bytes, byte_range: "ByteRequest | None") -> bytes:
        if byte_range is None:
            return value
        if isinstance(byte_range, RangeByteRequest):
            return value[byte_range.start : byte_range.end]
        if isinstance(byte_range, OffsetByteRequest):
            return value[byte_range.offset :]
        if isinstance(byte_range, SuffixByteRequest):
            return value[-byte_range.suffix :]
        return value

    class SlideIoVsiStore(_SlideIoVsiStoreBase, Store):
        """Wraps a slideio VSI scene as a read-only multiscale zarr v3 Store."""

        def __init__(self, path: str, scene: int = 0, tilesize: int = 1024):
            Store.__init__(self, read_only=True)
            self._init_source(path, scene, tilesize)

        @property
        def supports_writes(self) -> bool:
            return False

        @property
        def supports_deletes(self) -> bool:
            return False

        @property
        def supports_partial_writes(self) -> bool:
            return False

        @property
        def supports_listing(self) -> bool:
            return True

        def _get_bytes(self, key):
            try:
                return self.read_tile_bytes(key)
            except KeyError:
                return None

        async def get(self, key, prototype, byte_range=None):
            # zarr v3 drives all store I/O through a single event loop, so the
            # blocking slideio decode is offloaded to a thread to let dask's
            # worker threads read tiles concurrently (matching the v2 behavior).
            loop = asyncio.get_running_loop()
            value = await loop.run_in_executor(None, self._get_bytes, key)
            if value is None:
                return None
            return prototype.buffer.from_bytes(_slice_value(value, byte_range))

        async def get_partial_values(self, prototype, key_ranges):
            return [
                await self.get(key, prototype, byte_range)
                for key, byte_range in key_ranges
            ]

        async def exists(self, key) -> bool:
            if key in self._store:
                return True
            try:
                self.read_tile_bytes(key)
                return True
            except KeyError:
                return False

        async def set(self, key, value) -> None:
            raise NotImplementedError("SlideIoVsiStore is read-only")

        async def delete(self, key) -> None:
            raise NotImplementedError("SlideIoVsiStore is read-only")

        async def list(self):
            for key in self._store:
                yield key

        async def list_prefix(self, prefix):
            for key in self._store:
                if key.startswith(prefix):
                    yield key

        async def list_dir(self, prefix):
            prefix = prefix.rstrip("/")
            seen = set()
            for key in self._store:
                if prefix:
                    if not key.startswith(prefix + "/"):
                        continue
                    rest = key[len(prefix) + 1 :]
                else:
                    rest = key
                first = rest.split("/", 1)[0]
                if first not in seen:
                    seen.add(first)
                    yield first

        def close(self):
            _SlideIoVsiStoreBase.close(self)
            Store.close(self)

else:
    from zarr.storage import Store

    class SlideIoVsiStore(_SlideIoVsiStoreBase, Store):
        """Wraps a slideio VSI scene as a read-only multiscale zarr v2 Store."""

        def __init__(self, path: str, scene: int = 0, tilesize: int = 1024):
            self._init_source(path, scene, tilesize)

        def __getitem__(self, key: str):
            return self.read_tile_bytes(key)

        def getitems(self, keys, *, contexts):
            return {k: self[k] for k in keys}

        def __contains__(self, key: str):
            return key in self._store

        def __setitem__(self, key, val):
            raise RuntimeError("__setitem__ not implemented")

        def __delitem__(self, key):
            raise RuntimeError("__delitem__ not implemented")

        def __iter__(self):
            return iter(self.keys())

        def __len__(self):
            return sum(1 for _ in self)

        def keys(self):
            return self._store.keys()


if __name__ == "__main__":
    import sys

    store = SlideIoVsiStore(sys.argv[1])
