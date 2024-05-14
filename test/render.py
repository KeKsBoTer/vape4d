from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
import v4dv
import numpy as np


def felix_cmap_hack(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
        cmap.colors = cmap.colors.copy()
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


colormap = felix_cmap_hack(plt.get_cmap("magma"))
img = (
    v4dv.render(
        np.load("volumes/diffusion_trj.npz")["trj"][0, :1],
        colormap,
        0.5,
        width=1024,
        height=1024,
    ).astype(np.float32)
    / 255
)
# img[:, :, :3] = img[:, :, :3] * img[:, :, 3:4]
img[:, :, 3] = 1
Image.fromarray((img * 255).astype(np.uint8)).show()
