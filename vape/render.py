from typing import Optional, Union
from matplotlib.colors import Colormap
import numpy as np
from . import vape


def render(
    volume: np.ndarray,
    cmap: Colormap,
    time: Union[float, list[float]],
    width: int = 1024,
    height: int = 1024,
    background: tuple[float, float, float, float] = (0, 0, 0, 1),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    distance_scale: float = 1.0,
):
    colormap_data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    if isinstance(time, list):
        img = vape.render_video(
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
    else:
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
