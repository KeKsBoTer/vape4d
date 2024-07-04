import base64
import dataclasses
import io
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from IPython.display import DisplayObject

VAPE_URL = os.environ.get("VAPE_URL", "https://keksboter.github.io/vape4d")


_TEMPLATE_IFRAME = """
    <div>
        <iframe id="{canvas_id}" src="{viewer_url}/index.html?inline" width="{canvas_width}" height="{canvas_height}" frameBorder="0" sandbox="allow-same-origin allow-scripts"></iframe>
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
    distance_scale: float
    duration: Optional[float] = None


def viewer(
    data: np.ndarray,
    colormap,
    width: int = 800,
    height: int = 600,
    background_color=(0.0, 0.0, 0.0, 1.0),
    show_colormap_editor=False,
    show_volume_info=False,
    vmin=None,
    vmax=None,
    distance_scale=1.0,
    duration=None,
):
    return VolumeViewer(
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
            distance_scale,
            duration,
        ),
    )


class VolumeViewer(DisplayObject):
    def __init__(self, data: np.ndarray, colormap, settings: ViewerSettings):
        super(VolumeViewer, self).__init__(
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

        canvas_id = f"vape4d_canvas_{str(random.randint(0,2**32))}"
        html_code = _TEMPLATE_IFRAME.format(
            canvas_id=canvas_id,
            data_code=data_code.decode("utf-8"),
            cmap_code=cmap_code.decode("utf-8"),
            canvas_width=settings.width,
            canvas_height=settings.height,
            settings_json=json.dumps(dataclasses.asdict(settings)),
            viewer_url=VAPE_URL,
        )
        return html_code

    def __html__(self):
        return self._repr_html_()
