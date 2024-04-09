# %%
from matplotlib import pyplot as plt
from os import makedirs
import numpy as np
makedirs("colormaps",exist_ok=True)
for name in plt.colormaps():
    cmap = plt.get_cmap(name)
    data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
    np.save("colormaps/{}.npy".format(name), data)

# %%



