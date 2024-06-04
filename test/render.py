from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

from vape import diverging_alpha, render


colormap = diverging_alpha(plt.get_cmap("magma"))
img = (
    render(
        np.load("volumes/diffusion_trj.npz")["trj"][0, :1],
        colormap,
        0.5,
        width=1024,
        height=1024,
        vmin=-10,
        vmax=10,
        background=(0, 255, 0, 255),
        distance_scale=10,
    ).astype(np.float32)
    / 255
)
# img[:, :, :3] = img[:, :, :3] * img[:, :, 3:4]
img[:, :, 3] = 1
Image.fromarray((img * 255).astype(np.uint8)).show()
