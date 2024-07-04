# vAPE4D
Real-time rendering for time-variying volumes (4D = 3D Space + Time) written in WebGPU and Rust.

**The renderer runs in your brower or embeded in a Jupyter Notebook (e.g. inside vscode)**

## How it works

**TLDR**: We map the values in the volume to RGBA color with a user defined transfer function (colormap) and use standard volume rendering.

![Volume Rendering](img/volume_rendering_light.svg#only-light)
![Volume Rendering](img/volume_rendering_dark.svg#only-dark)

## Rendering Technique
The rendering techniques used for volume rendering rely heavily on methods described in the book [Real-Time Volume Graphics](http://www.real-time-volume-graphics.org/).

A user defined transfer function $f$ is used that maps a value $\hat{v_i}\in[0,1]$ to RGBA color:   

$c_i := f_\textrm{RGB}(\hat{v_i})\quad a_i := f_\textrm{A}(\hat{v_i})$


For each pixel we march along a ray with step size $\delta$. At each step the volume is sampled to retrive a value $v_i$. The value $v_i$ is normalized using the user provided $v_\textrm{min}$ and $v_\textrm{max}$ (defaults to minimum and maximum of volume):

$\hat{v_i} = \min(\max(\frac{v_i -v_\textrm{min}}{v_\textrm{max}-v_\textrm{min}}$,0),1)

To calculate the final pixel color the $N$ samples along a ray are accumulated using alpha blending:

$C_p = \sum_{i=0}^{N} \hat{a_i} C_i \prod_{j=0}^{i-1}(1-\hat{a_j})$

The step size $\delta$ and a **distance scaling** factor $s$ are user controlled and the opacity (alpha) must be corrected accordingly:

$\hat{a_i} = 1-a_i^{\delta s}$

**Step size $\delta$**: A smaller step size gives more detailed / accurate results but results in worse performance.

**Distance Scaling**: Since the "original" sampling rate is not known the user has to specify it. A larger value makes the whole volume appear more opaque/dense.

## Getting Started

## Installation

```bash
pip install vape4d
```

## Usage

**Viewer** (jupyter notebook)
```python
from vape4d import viewer
import numpy as np
from matplotlib import pyplot as plt

viewer(
    #[T,C,D,W,H]
    np.random.rand(1,1,32,32,32),
    plt.get_cmap("viridis"),   
)
```

**Render Image**
```python
import numpy as np
from vape4d import diverging_alpha, render
import matplotlib.pyplot as plt

colormap = diverging_alpha(plt.get_cmap("magma"))
img = render(
        # [T,D,H,W]
        np.random.rand(2,32,32,32).astype(np.float32),
        colormap,
        0.5, # timestep
        width=1024,
        height=1024,
    )

plt.imshow(img)
plt.axis("off")

plt.savefig("test.png", bbox_inches="tight", pad_inches=0)
plt.show()
```

