import asyncio
import json
from ctypes import ArgumentError
from pathlib import Path

import numpy as np
import zarr
from openslide import OpenSlide

# zarr v3 replaced the synchronous ``zarr.storage.Store`` mapping interface with
# an async ``zarr.abc.store.Store`` and removed the v2 metadata helpers
# (``init_array``/``init_group``/``json_dumps``/...). We keep supporting both by
# hand-building v2-format metadata (which zarr v3 still reads) and selecting the
# store base class at import time based on the installed zarr version.
_ZARR_V3 = int(zarr.__version__.split(".")[0]) >= 3


def create_meta_store(slide: OpenSlide, tilesize: int) -> dict:
    """Creates a dict of zarr v2 metadata (as bytes) for the multiscale image."""
    store = {}
    root_attrs = {
        "multiscales": [
            {
                "name": Path(slide._filename).name,
                "datasets": [{"path": str(i)} for i in range(slide.level_count)],
                "version": "0.1",
            }
        ]
    }
    store[".zgroup"] = json.dumps({"zarr_format": 2}).encode()
    store[".zattrs"] = json.dumps(root_attrs).encode()
    for i, (x, y) in enumerate(slide.level_dimensions):
        store[f"{i}/.zarray"] = json.dumps(
            {
                "zarr_format": 2,
                "shape": [y, x, 4],
                "chunks": [tilesize, tilesize, 4],
                "dtype": "|u1",
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


class _OpenSlideStoreBase:
    """Slide-backed metadata + tile reading shared by the v2 and v3 stores.

    Parameters
    ----------
    path: str
        The file to open with OpenSlide.
    tilesize: int
        Desired "chunk" size for zarr store.
    """

    def _init_source(self, path: str, tilesize: int) -> None:
        self._slide = OpenSlide(path)
        self._tilesize = tilesize
        self._store = create_meta_store(self._slide, tilesize)

    def _ref_pos(self, x: int, y: int, level: int):
        dsample = self._slide.level_downsamples[level]
        xref = int(x * dsample * self._tilesize)
        yref = int(y * dsample * self._tilesize)
        return xref, yref

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
            tile = self._slide.read_region(location, level, size)
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
            isinstance(other, OpenSlideStore)
            and self._slide._filename == other._slide._filename
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._slide.close()


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

    class OpenSlideStore(_OpenSlideStoreBase, Store):
        """Wraps an OpenSlide object as a read-only multiscale zarr v3 Store."""

        def __init__(self, path: str, tilesize: int = 1024):
            Store.__init__(self, read_only=True)
            self._init_source(path, tilesize)

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
            # blocking OpenSlide decode is offloaded to a thread to let dask's
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
            raise NotImplementedError("OpenSlideStore is read-only")

        async def delete(self, key) -> None:
            raise NotImplementedError("OpenSlideStore is read-only")

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
            _OpenSlideStoreBase.close(self)
            Store.close(self)

else:
    from zarr.storage import Store

    class OpenSlideStore(_OpenSlideStoreBase, Store):
        """Wraps an OpenSlide object as a read-only multiscale zarr v2 Store."""

        def __init__(self, path: str, tilesize: int = 1024):
            self._init_source(path, tilesize)

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

    store = OpenSlideStore(sys.argv[1])
