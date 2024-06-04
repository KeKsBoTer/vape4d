from typing import Optional
from matplotlib.colors import Colormap
import numpy as np
from . import vape


def render(
    volume: np.ndarray,
    cmap: Colormap,
    time: float,
    width: int = 1024,
    height: int = 1024,
    background: tuple[float, float, float, float] = (0, 0, 0, 1),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    distance_scale: float = 1.0,
):
    colormap_data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    img = vape.render_img(
        volume,
        colormap_data,
        width,
        height,
        time,
        background,
        distance_scale,
        vmin,
        vmax,
    )
    return img
