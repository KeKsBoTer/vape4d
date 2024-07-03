from copy import deepcopy
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
import numpy as np


def diverging_alpha(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
        cmap.colors = deepcopy(cmap.colors)
        for i, a in enumerate(cmap.colors):
            a.append(2 * abs(i / cmap.N - 0.5))
    elif isinstance(cmap, LinearSegmentedColormap):
        cmap._segmentdata["alpha"] = np.array(
            [[0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )
    return cmap


def linear_increasing_alpha(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be linear (0->0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
        cmap.colors = cmap.colors.copy()
        for i, a in enumerate(cmap.colors):
            a.append(i / (cmap.N - 1))
    elif isinstance(cmap, LinearSegmentedColormap):
        cmap._segmentdata["alpha"] = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )
    return cmap
