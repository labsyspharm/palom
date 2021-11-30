try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)


from . import (
    reader,
    align,
    pyramid,
    color,
    
    # debugging
    block_affine,
    extract,
    img_util,
    register
)