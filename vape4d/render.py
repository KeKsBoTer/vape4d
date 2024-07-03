from typing import Optional, Union
from matplotlib.colors import Colormap
import numpy as np
from . import vape4d


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
    spatial_interpolation: str = "linear",
    temporal_interpolation: str = "linear",
) -> np.ndarray:
    """renders a single or multiple images of a volume

    Args:
        volume (np.ndarray): volume data of shape [N, H, W, D]
        cmap (Colormap): colormap to use for rendering
        time (Union[float, list[float]]): if a single value is given, a single image is rendered at that time. If a list of values is given, a video is rendered with the given times.
        width (int, optional): image width. Defaults to 1024.
        height (int, optional): image height. Defaults to 1024.
        background (tuple[float, float, float, float], optional): background color. Defaults to (0, 0, 0, 1).
        vmin (Optional[float], optional): minimum value for colormap. defaults to minimum value in volume.
        vmax (Optional[float], optional): maximum value for colormap. defaults to maximum value in volume.
        distance_scale (float, optional): distance scale for rendering. A larger value makes everything more opaque. Defaults to 1.0.
        spatial_interpolation (str, optional): interpolation in space. Linear or Nearest. Defaults to "linear".
        temporal_interpolation (str, optional): interpolation in time. Linear or Nearest. Defaults to "linear".

    Returns:
        np.ndarray: [T, H, W, 4] if time is a list, [H, W, 4] if time is a single value
    """
    colormap_data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    if isinstance(time, list):
        img = vape4d.render_video(
            volume,
            colormap_data,
            width,
            height,
            time,
            background,
            distance_scale,
            vmin,
            vmax,
            spatial_interpolation,
            temporal_interpolation,
        )
        return img
    else:
        img = vape4d.render_img(
            volume,
            colormap_data,
            width,
            height,
            time,
            background,
            distance_scale,
            vmin,
            vmax,
            spatial_interpolation,
            temporal_interpolation,
        )
        return img
