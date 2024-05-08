import dataclasses
import io
import json
from typing import Optional
import numpy as np
import random
from IPython.display import DisplayObject
import base64
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


TEMPLATE_IFRAME = """
    <div>
        <iframe id="{canvas_id}" src="http://localhost:5500/public/index.html?inline" width="{canvas_width}" height="{canvas_height}" frameBorder="0" sandbox="allow-same-origin allow-scripts"></iframe>
    </div>
    <script>

        window.addEventListener(
            "message",
            (event) => {{
                if (event.data !== "ready") {{
                    return;
                }}
                let data_decoded = Uint8Array.from(atob("{data_code}"), c => c.charCodeAt(0));
                let cmap_decoded = Uint8Array.from(atob("{cmap_code}"), c => c.charCodeAt(0));
                const iframe = document.getElementById("{canvas_id}");
                if (iframe === null) return;
                iframe.contentWindow.postMessage({{
                    volume: data_decoded,
                    cmap: cmap_decoded,
                    settings: {settings_json}
                }},
                "*");
            }},
            false,
        );
    </script>
"""


@dataclass(unsafe_hash=True)
class ViewerSettings:
    width: int
    height: int
    background_color: tuple
    show_colormap_editor: bool
    show_volume_info: bool
    vmin: Optional[float]
    vmax: Optional[float]


def show(
    data: np.ndarray,
    colormap,
    width: int = 800,
    height: int = 600,
    background_color=(0.0, 0.0, 0.0, 1.0),
    show_colormap_editor=False,
    show_volume_info=False,
    vmin=None,
    vmax=None,
):
    return VolumeRenderer(
        data,
        colormap,
        ViewerSettings(
            width,
            height,
            background_color,
            show_colormap_editor,
            show_volume_info,
            vmin,
            vmax,
        ),
    )


class VolumeRenderer(DisplayObject):
    def __init__(self, data: np.ndarray, colormap, settings: ViewerSettings):
        super(VolumeRenderer, self).__init__(
            data={"volume": data, "cmap": colormap, "settings": settings}
        )

    def _repr_html_(self):

        data = self.data["volume"]
        colormap = self.data["cmap"]
        settings = self.data["settings"]
        buffer = io.BytesIO()
        np.save(buffer, data.astype(np.float32))
        data_code = base64.b64encode(buffer.getvalue())

        buffer2 = io.BytesIO()
        colormap_data = colormap(np.linspace(0, 1, 256)).astype(np.float32)
        np.save(buffer2, colormap_data)
        cmap_code = base64.b64encode(buffer2.getvalue())

        canvas_id = f"v4dv_canvas_{str(random.randint(0,2**32))}"
        html_code = TEMPLATE_IFRAME.format(
            canvas_id=canvas_id,
            data_code=data_code.decode("utf-8"),
            cmap_code=cmap_code.decode("utf-8"),
            canvas_width=settings.width,
            canvas_height=settings.height,
            settings_json=json.dumps(dataclasses.asdict(settings)),
        )
        return html_code

    def __html__(self):
        """
        This method exists to inform other HTML-using modules (e.g. Markupsafe,
        htmltag, etc) that this object is HTML and does not need things like
        special characters (<>&) escaped.
        """
        return self._repr_html_()


def felix_cmap_hack(cmap: Colormap) -> Colormap:
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:
        Colormap: new colormap
    """
    cmap = cmap.copy()
    if isinstance(cmap, ListedColormap):
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
