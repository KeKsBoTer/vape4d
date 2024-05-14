# %%
import json
from matplotlib import pyplot as plt
from os import makedirs
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np

makedirs("colormaps", exist_ok=True)
for name in plt.colormaps():
    cmap = plt.get_cmap(name)
    if isinstance(cmap, ListedColormap):
        data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
        np.save("colormaps/{}.npy".format(name), data)
    elif isinstance(cmap, LinearSegmentedColormap):
        segments = cmap._segmentdata

        if any(
            not isinstance(segments[c], np.ndarray)
            and not isinstance(segments[c], list)
            and not isinstance(segments[c], tuple)
            for c in ["red", "green", "blue"]
        ):
            data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
            np.save("colormaps/{}.npy".format(name), data)
        else:
            channels = {
                c: (
                    segments[c].tolist()
                    if isinstance(segments[c], np.ndarray)
                    else segments[c]
                )
                for c in ["red", "green", "blue"]
            }
            if "alpha" in channels:
                channels["alpha"] = segments["alpha"].tolist()
            with open("colormaps/{}.json".format(name), "w") as f:
                json.dump(
                    channels,
                    f,
                    indent=2,
                )
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )

# %%
