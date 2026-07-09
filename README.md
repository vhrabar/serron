# Serron

![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.12-ee4c2c?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/cuda-13.X-76b900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge&logo=opensourceinitiative)


Mathematical Morphology-based self-attention module for PyTorch (CUDA) using Flash-style kernel fusion.

## Layout

This is a [uv](https://docs.astral.sh/uv/) workspace:

- `packages/serron` - the published kernel package ([README](packages/morphottention/README.md)).

## Install

Prebuilt wheels (CPython 3.12–3.14; Linux x86_64/aarch64, Windows x86_64) require a
CUDA-enabled `torch >= 2.12` already installed:


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
