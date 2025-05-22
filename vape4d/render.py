from typing import Optional, Union,Tuple,List
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import numpy as np
from . import vape4d


def render(
    volume: np.ndarray,
    cmap: Optional[Colormap] = None,
    time: Optional[Union[float, List[float]]] = 0.0,
    width: int = 1024,
    height: int = 1024,
    background: Tuple[float, float, float, float] = (0, 0, 0, 1),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    distance_scale: float = 1.0,
    spatial_interpolation: str = "linear",
    temporal_interpolation: str = "linear",
    camera_angle: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """renders a single or multiple images of a volume

    Args:
        volume (np.ndarray): volume data of shape [N, D, H, W]
        cmap (Colormap): colormap to use for rendering. Defaults to matplotlib's default colormap.
        time (Union[float, list[float]]): if a single value is given, a single image is rendered at that time. If a list of values is given, a video is rendered with the given times.
        width (int, optional): image width. Defaults to 1024.
        height (int, optional): image height. Defaults to 1024.
        background (tuple[float, float, float, float], optional): background color. Defaults to (0, 0, 0, 1).
        vmin (Optional[float], optional): minimum value for colormap. defaults to minimum value in volume.
        vmax (Optional[float], optional): maximum value for colormap. defaults to maximum value in volume.
        distance_scale (float, optional): distance scale for rendering. A larger value makes everything more opaque. Defaults to 1.0.
        spatial_interpolation (str, optional): interpolation in space. Linear or Nearest. Defaults to "linear".
        temporal_interpolation (str, optional): interpolation in time. Linear or Nearest. Defaults to "linear".
        camera_angle (Optional[Tuple[float, float]], optional): camera angle for rendering in spherical coordinates (polar,azimuthal angle ). Defaults to None.

    Returns:
        np.ndarray: [T, H, W, 4] if time is a list, [H, W, 4] if time is a single value
    """

    if cmap is None:
        cmap = plt.get_cmap()

    if volume.ndim == 5:
        # check if we have a single channel
        if volume.shape[1] != 1:
            raise ValueError("only one channel supported")
    elif volume.ndim == 4:
        pass
    elif volume.ndim == 3:
        # add one channel
        volume = volume[:, None]
    else:
        raise ValueError(
            "volume must have shape [T,1, D, H, W], [T, D, H, W] or [D,H,W] "
        )

    colormap_data = cmap(np.linspace(0, 1, 256)).astype(np.float32)

    if isinstance(time, np.ndarray):
        time = time.tolist()

    if not isinstance(time, list):
        time = [time]

    frames = vape4d.render_video(
        np.ascontiguousarray(volume).astype(np.float16),
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
        camera_angle,
    )
    if len(time) == 1:
        return frames[0]
    return frames
