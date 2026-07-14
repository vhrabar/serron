# Serron

![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.12-ee4c2c?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/cuda-13.X-76b900?style=for-the-badge&logo=nvidia)
![PyPI Version](https://img.shields.io/pypi/v/serron?style=for-the-badge&logo=pypi&logoColor=orange)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=opensourceinitiative)


Mathematical Morphology module for PyTorch (CUDA), providing differentiable operators and learnable network layers.

## Layout

This is an [uv](https://docs.astral.sh/uv/) workspace:

- `packages/serron` - the published kernel package ([README](packages/morphottention/README.md)).

## Install

Prebuilt wheels (CPython 3.12–3.14; Linux x86_64/aarch64, Windows x86_64) require a
CUDA-enabled `torch >= 2.12` already installed:

```bash
pip install serron
```

## Usage

> **Alpha:** the CUDA kernels are not wired up yet, so the operators below currently
> raise `NotImplementedError`.

Serron works on standard PyTorch tensors laid out as `(N, C, H, W)` and living on a CUDA
device.

### Functional operators

Stateless operators live in `serron.functional` and are re-exported at the top level.
Each takes an input tensor, a structuring element, and an optional `border` mode:

```python
import torch
import serron
from serron import BorderMode
from serron import structuring_element as se

x = torch.rand(1, 1, 256, 256, device="cuda")
kernel = se.disk(3, device="cuda")

eroded  = serron.erosion(x, kernel)
dilated = serron.dilatation(x, kernel)
opened  = serron.opening(x, kernel)
closed  = serron.closing(x, kernel)

# Derived operators
grad    = serron.gradient(x, kernel)   # dilate(x) - erode(x)
white   = serron.top_hat(x, kernel)    # x - open(x)
black   = serron.black_hat(x, kernel)  # close(x) - x

# Control border handling
eroded_reflect = serron.erosion(x, kernel, border=BorderMode.REFLECT)
```

Available operators: `erosion`, `dilatation`, `opening`, `closing`, `gradient`,
`top_hat`, `black_hat`.

### Structuring elements

`serron.structuring_element` builds common SE shapes on the requested device:

```python
from serron import structuring_element as se

se.square(5, device="cuda")      # (5, 5) full square
se.cross(5, device="cuda")       # (5, 5) plus shape
se.disk(3, device="cuda")        # (7, 7) disk, radius 3
se.diamond(3, device="cuda")     # (7, 7) diamond, radius 3
se.from_tensor(my_weights)       # wrap an arbitrary 2-D tensor as a grayscale SE
```

### Border modes

`BorderMode` controls how out-of-bounds neighbors are handled:

| Mode | Behavior |
| --- | --- |
| `BorderMode.REPLICATE` | Repeat the edge value (default) |
| `BorderMode.REFLECT` | Mirror across the edge |
| `BorderMode.CONSTANT` | Pad with a constant |

### Learnable layers

`serron` also exposes `torch.nn.Module` layers with a learnable structuring element,
so morphology can be trained end-to-end inside a network. Each layer takes the number
of `channels` and a `kernel_size`:

```python
import torch
from serron import Erosion2d, Dilation2d, Opening2d, Closing2d
from serron import BorderMode

layer = Erosion2d(channels=3, kernel_size=5, border=BorderMode.REPLICATE).cuda()

x = torch.rand(8, 3, 64, 64, device="cuda")
y = layer(x)          # forward pass; layer.weight is a trainable (C, k, k) SE
y.sum().backward()    # gradients flow into layer.weight
```

Available layers: `Erosion2d`, `Dilation2d`, `Opening2d`, `Closing2d`.


## Develop

Set up the workspace:
```bash
uv sync
```

Building the CUDA extension from source needs the CUDA 13.X toolkit (`nvcc`):

```bash
uv sync --package serron --no-dev --group build
uv build --package serron --wheel --no-build-isolation
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

Copyright © 2026 Vedran Hrabar
