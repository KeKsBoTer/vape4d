from matplotlib.colors import Colormap
import numpy as np
from . import v4dv


def render(
    volume: np.ndarray,
    cmap: Colormap,
    time: float,
    width: int = 1024,
    height: int = 1024,
    background: tuple[float, float, float, float] = (0, 0, 0, 1),
):
    colormap_data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    img = v4dv.render_img(volume, colormap_data, width, height, time, background)
    return img
