# %%
import json
from matplotlib import pyplot as plt
from os import makedirs
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Colormap
import numpy as np


def export_colormaps(cm_module, folder: str):
    makedirs(folder, exist_ok=True)
    for name, cmap in vars(cm_module).items():
        if not isinstance(cmap, Colormap):
            continue
        if name.endswith("_r"):
            continue
        if name.startswith("cmr."):
            name = name[4:]
        if isinstance(cmap, ListedColormap):
            data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
            np.save("{}/{}.npy".format(folder, name), data)
        elif isinstance(cmap, LinearSegmentedColormap):
            segments = cmap._segmentdata

            if any(
                not isinstance(segments[c], np.ndarray)
                and not isinstance(segments[c], list)
                and not isinstance(segments[c], tuple)
                for c in ["red", "green", "blue"]
            ):
                data = cmap(np.linspace(0, 1, 256)).astype(np.float32)
                np.save("{}/{}.npy".format(folder, name), data)
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
                with open("{}/{}.json".format(folder, name), "w") as f:
                    json.dump(
                        channels,
                        f,
                        indent=2,
                    )
        else:
            raise TypeError(
                "cmap must be either a ListedColormap or a LinearSegmentedColormap"
            )


export_colormaps(plt.cm, "colormaps/matplotlib")

import seaborn as sns

export_colormaps(sns.cm, "colormaps/seaborn")

import cmasher

export_colormaps(cmasher.cm, "colormaps/cmasher")
