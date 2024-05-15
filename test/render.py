from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

from v4dv import felix_cmap_hack, render


colormap = felix_cmap_hack(plt.get_cmap("magma"))
img = (
    render(
        np.load("volumes/diffusion_trj.npz")["trj"][0, :1],
        colormap,
        0.5,
        width=1024,
        height=1024,
        vmin=-5,
        vmax=5,
    ).astype(np.float32)
    / 255
)
# img[:, :, :3] = img[:, :, :3] * img[:, :, 3:4]
img[:, :, 3] = 1
Image.fromarray((img * 255).astype(np.uint8)).show()
