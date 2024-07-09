from .render import render
from . import utils

try:
    # check if ipython is available
    import IPython as _
    from .viewer import viewer, ViewerSettings, VolumeViewer
except ImportError:
    pass

from . import vape4d as vape_py

__doc__ = vape_py.__doc__
if hasattr(vape_py, "__all__"):
    __all__ = vape_py.__all__
