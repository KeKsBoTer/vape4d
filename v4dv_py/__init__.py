import io
import numpy as np
import random
from IPython.display import HTML, DisplayObject
from matplotlib.colors import Colormap
from matplotlib.cm import get_cmap
import base64

TEMPLATE = """
    <div>
        <canvas id="{canvas_id}" width="800" height="600"></canvas>
    </div>
    <script>
        (function() {{
            {js_code}
            const {{ initSync,run_wasm }} = wasm_bindgen;
            initSync(Uint8Array.from(atob("{wasm_code}"), c => c.charCodeAt(0)));
            run_wasm(
                Uint8Array.from(atob("{data_code}"), c => c.charCodeAt(0)),
                Uint8Array.from(atob("{cmap_code}"), c => c.charCodeAt(0)),
                "{canvas_id}"
            );
        }})();
    </script>
"""


def show(data: np.ndarray, colormap: Colormap):

    return VolumeRenderer(data, colormap)


class VolumeRenderer(DisplayObject):
    def __init__(self, data: np.ndarray, colormap: Colormap):
        super(VolumeRenderer, self).__init__(data={"volume": data, "cmap": colormap})

    def _repr_html_(self):
        with open("public/v4dv.js", "r") as f:
            js_code = f.read()
        with open("public/v4dv_bg.wasm", "rb") as f:
            wasm_code = f.read()

        data = self.data["volume"]
        colormap = self.data["cmap"]
        buffer = io.BytesIO()
        np.savez(buffer, trj=data.astype(np.float32))
        data_code = base64.b64encode(buffer.getvalue())

        buffer2 = io.BytesIO()
        colormap_data = colormap(np.linspace(0, 1, 256)).astype(np.float32)
        np.save(buffer2, colormap_data)
        cmap_code = base64.b64encode(buffer2.getvalue())

        wasm_code_b64 = base64.b64encode(wasm_code)

        canvas_id = f"v4dv_canvas_{str(random.randint(0,2**32))}"
        html_code = TEMPLATE.format(
            js_code=js_code,
            canvas_id=canvas_id,
            data_code=data_code.decode("utf-8"),
            cmap_code=cmap_code.decode("utf-8"),
            wasm_code=wasm_code_b64.decode("utf-8"),
        )
        return html_code

    def __html__(self):
        """
        This method exists to inform other HTML-using modules (e.g. Markupsafe,
        htmltag, etc) that this object is HTML and does not need things like
        special characters (<>&) escaped.
        """
        return self._repr_html_()
