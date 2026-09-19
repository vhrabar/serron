# Serron

Mathematical Morphology module for PyTorch (CUDA), providing differentiable operators and learnable network layers.

## Install

Prebuilt wheels are published for CPython 3.12–3.14 on Linux (x86_64, aarch64) and
Windows (x86_64).

```bash
pip install serron
```

## Usage

Serron works on standard PyTorch tensors laid out as `(N, C, H, W)` and living on a CUDA
or CPU device.

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

eroded = serron.erosion(x, kernel)
dilated = serron.dilation(x, kernel)
opened = serron.opening(x, kernel)
closed = serron.closing(x, kernel)

# Derived operators
grad = serron.gradient(x, kernel)  # dilate(x) - erode(x)
white = serron.top_hat(x, kernel)  # x - open(x)
black = serron.black_hat(x, kernel)  # close(x) - x

# Control border handling
eroded_reflect = serron.erosion(x, kernel, border=BorderMode.REFLECT)
```

Available operators: `erosion`, `dilation`, `opening`, `closing`, `gradient`,
`top_hat`, `black_hat`.

### Structuring elements

`serron.structuring_element` builds common SE shapes on the requested device:

```python
from serron import structuring_element as se

se.square(5, device="cuda")  # (5, 5) full square
se.cross(5, device="cuda")  # (5, 5) plus shape
se.disk(3, device="cuda")  # (7, 7) disk, radius 3
se.diamond(3, device="cuda")  # (7, 7) diamond, radius 3
se.from_tensor(my_weights)  # wrap an arbitrary 2-D tensor as a grayscale SE
```

### Border modes

`BorderMode` controls how out-of-bounds neighbors are handled:

| Mode                   | Behavior                        |
|------------------------|---------------------------------|
| `BorderMode.REPLICATE` | Repeat the edge value (default) |
| `BorderMode.REFLECT`   | Mirror across the edge          |
| `BorderMode.CONSTANT`  | Pad with a constant             |

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
y = layer(x)  # forward pass; layer.weight is a trainable (C, k, k) SE
y.sum().backward()  # gradients flow into layer.weight
```

Available layers: `Erosion2d`, `Dilation2d`, `Opening2d`, `Closing2d`.

## Implementation

Every operator picks one of a few kernels at call time. The structuring element and the
device's shared memory decide which one.

### Which path runs



| Path                          | Runs when                                                     | Cost per output           |
|-------------------------------|---------------------------------------------------------------|---------------------------|
| Separable van Herk–Gil–Werman | Flat SE and `max(kH, kW) >= SERRON_SEPARABLE_MIN_K`           | O(1) in the window length |
| Tiled 2-D                     | Otherwise, when the halo tile and the SE fit in shared memory | O(kH × kW)                |
| Element-wise                  | Otherwise; reads through global memory (CUDA only)            | O(kH × kW)                |

`SERRON_SEPARABLE_MIN_K` is the window length where the separable path takes over. It
accepts any positive integer. The default differs per backend because it marks where the
separable path starts beating the 2-D one, and that point is not the same on both: `11` on
CUDA, `5` on CPU, where there is no tiled 2-D kernel to soften the `kH × kW` window.

### The separable path

Splitting a flat `kH × kW` window into a `1 × kW` row pass and a `kH × 1` column pass
already drops the per-pixel work from `kH × kW` to `kH + kW`, and the Van Herk–Gil–Werman algorithm 
drops it further to a constant three operations per ouput, no matter the sequence length.

On CUDA both scans sit in shared memory. A chunk at least a warp wide gets a warp to
itself and the lanes cooperate through shuffles; anything shorter gets a single thread.
On CPU the lines go through `at::parallel_for`.

### Backward

Backward needs to know *where* the winning sample was, not just what it was, so its scans
carry a `(value, offset)` pair rather than a bare value. Combining two of those keeps the
lower offset on a tie, which is the rule the direct kernel gets from its strict-improvement
loop — reproducing it exactly is what makes the two paths agree on which tap receives the
gradient. The window is one pair-combine as before, so the search is O(1) in the window
length on both the row and the column pass.

It splits on the same condition: a separable row and column argreduce for a flat SE at or
above the threshold, and a direct recompute of the winning tap otherwise. The direct
recompute stages its window in shared memory the same way the forward's tiled path does.


## Benchmarks

Per-operator throughput and the cross-library comparison live in
[`benchmarks/README.md`](https://github.com/vhrabar/serron/blob/main/benchmarks/README.md).

Erosion with a flat SE, `8x3x512x512` float32 on an H100 PCIe (with Intel Xeon Platinum 8480+), in milliseconds:

|   k |   serron | PyTorch | Kornia | CuPy | SciPy (CPU) |
|----:|---------:|--------:|-------:|-----:|------------:|
|   7 |     0.30 |    0.42 |   2.80 | 0.20 |         159 |
|  31 | **0.23** |    5.46 |  41.87 | 0.58 |         151 |
|  63 | **0.31** |   21.48 |    OOM | 1.05 |         144 |
| 127 | **0.25** |   81.03 |    OOM | 2.00 |         142 |



## Building from source

### Clone repo
```bash
git clone git@github.com:vhrabar/serron.git
cd serron
```

### Choosing a torch build

`torch` is pulled from a specific wheel index via mutually-exclusive extras. Pick the
one matching your machine, plain `uv sync` (no extra) falls back to the default
CUDA-enabled wheel from PyPI:

```bash
uv sync --extra cu132   # CUDA 13.2 build
uv sync --extra cpu     # CPU-only build
```

### Building the wheel

Building the CUDA extension from source needs the CUDA 13.X toolkit (`nvcc`):

```bash
uv sync --package serron --no-dev --group build --extra cu132
uv build --package serron --wheel --no-build-isolation
```

On a GPU-less machine, sync the CPU torch build instead; the extension then builds
C++ only (no `nvcc` required):

```bash
uv sync --package serron --no-dev --group build --extra cpu
uv build --package serron --wheel --no-build-isolation
```

## License

MIT

Copyright © 2026 Vedran Hrabar.
