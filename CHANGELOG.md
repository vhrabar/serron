# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-10-01

### Added

- Geodesic reconstruction: `reconstruction_by_dilation` and `reconstruction_by_erosion` iterate
    `min(dilation(y), mask)` (resp. `max(erosion(y), mask)`) from a marker to its fixed point. The iteration is a
    Python loop over the compiled `erode`/`dilate` ops, so it runs on both backends, and it defaults to
    `BorderMode.CONSTANT`, which reads out-of-image taps as the neutral element. Convergence is tested once every
    `check_every` steps instead of every step: the iteration is monotone and bounded by the mask, so a block that
    leaves the image unchanged has reached the fixed point, and the device-to-host reads stay rare. The marker is
    clipped into the mask rather than checked against it, which costs no sync and lands on the same fixed point.
    There is no backward pass yet, so an input that wants a gradient is rejected instead of silently detached (#35).
- Separable van Herk–Gil–Werman path for `erode`/`dilate` on CUDA and CPU, forward and backward, so a flat,
    axis-separable structuring element costs O(1) per output in the window length instead of O(kH × kW). A flat SE
    splits into a `1 × kW` row pass and a `kH × 1` column pass; each line is then scanned forward and backward in
    chunks of `k`, so every output window straddles exactly one chunk boundary and resolves in a single combine. (#34).
- Backward support on the separable path: its scans carry a `(value, offset)` pair rather than a bare value, since the
    gradient needs the winning sample's position and not only its value. Ties keep the lowest offset, the same rule the
    direct kernel gets from its strict-improvement loop, so both paths route the gradient to the same tap (#34).
- Tiled backward kernel for the non-separable case, staging its halo tile and the structuring element in shared memory
    so an input sample is read once per block rather than once per window it falls in. The element-wise kernel remains
    the fallback when the tile does not fit (#34).

### Changed

- Structuring-element flatness is now cached instead of measured on every call, which removes a per-call device-to-host
    sync from the path selection and makes `erosion` and `dilation` capturable in a CUDA graph.  (#34).
- CUDA kernels now opt into the device's full dynamic shared-memory limit rather than using the default one,
    so the tiled and separable paths reach further before handing off. (#34).

### Fixed

- `import serron` no longer fails on a CPU-only torch with `libc10_cuda.so: cannot open shared object file`. The
    extension is now split in two: `_C` holds the op schemas and the CPU kernels and links only `torch_cpu`/`c10`,
    while the CUDA kernels live in `_C_cuda`, which is loaded only when the installed torch was built with CUDA (#41).
- The torch requirement is pinned to the minor release the wheels are built against (`torch>=2.14,<2.15`), since the
    torch C++ ABI is not stable across minor releases; a newer torch is now a resolver error instead of an
    `undefined symbol` at import. The release workflow reads this pin from `pyproject.toml` when installing the torch
    it compiles against (#41).
- A torch/extension mismatch at import now raises an `ImportError` naming the torch version the extension was built
    against and the installed one, instead of the raw loader error (#41).

## [0.3.1] - 2026-08-31

### Fixed

- Release-wheel builds no longer fail on a `scikit-build-core` version mismatch: the CI
    workflow installs the exact pin from `packages/serron/pyproject.toml`
    (`[build-system].requires`) instead of a stale hardcoded one. As a result, 0.3.0 shipped
    no prebuilt wheels.
- The `ghcr.io/vhrabar/serron` container image builds again: the `ubuntu:26.04` CUDA base
    ships a UID-1000 user, which aborted `useradd --uid 1000 app` during the 0.3.0 image build.

## [0.3.0] - 2026-08-31

### Added

- CPU backward kernels for `erode`/`dilate`, registered on the `CPU` dispatch key, so `.backward()` now works for CPU
    tensors: gradients flow into both the input and the structuring element through `erosion`, `dilation`, the composite
    ops (`opening`, `closing`, `gradient`, `top_hat`, `black_hat`) and the learnable layers, matching the CUDA path (#28).
- Autocast support for `erode`/`dilate` through dedicated `Autocast` and `AutocastCPU` dispatch implementations, so
    `erosion`, `dilation`, the composite ops and the learnable layers run correctly inside `torch.autocast` regions: the
    image and the structuring element are cast to the autocast execution dtype before the kernel runs.
- Published container image `ghcr.io/vhrabar/serron`, built and pushed to GHCR on every release tag, bundling both the testing and benchmarking
    suites for easier reproducibility.

### Changed

- Renamed the dilation operator from `dilatation` to `dilation`, matching the spelling used across the ecosystem. This
    affects `serron.dilation`, `serron.functional.dilation` and the internal call sites of `Dilation2d`, `opening`,
    `closing` and `gradient`. **Breaking:** the old `dilatation` / `serron.functional.dilatation` name is no longer
    exported.
- `erosion` and `dilation` (and the `_ErodeFunction`/`_DilateFunction` autograd bindings) now promote a mismatched
    image/structuring-element dtype to their common type via `torch.promote_types` instead of raising; the raw
    `torch.ops.serron.*` operators still require both tensors to share a dtype.

## [0.2.0] - 2026-08-26

### Added

- CUDA backward kernels for `erode`/`dilate`, wired through new `_ErodeFunction`/`_DilateFunction` autograd bindings, 
    so gradients now flow through `erosion`, `dilation`, and the composite ops (`opening`, `closing`, `gradient`, `top_hat`,
    `black_hat`) into both the input and the structuring element, including the learnable layers (`Erosion2d`, `Dilation2d`,
    `Opening2d`, `Closing2d`), which can now be optimized E2E (#13, #15). CUDA only for now; CPU tensors still raise on `.backward()`.

## [0.1.1] - 2026-07-25

### Added

- CPU backend for `erode` and `dilate`, registered on the `CPU` dispatch key so CPU tensors no longer require a CUDA build (#17).
- CPU-only and CUDA 13.x build paths in the installation instructions (#17).

## [0.1.0] - 2026-07-18

### Added

- CUDA grayscale erosion and dilation kernels, including the shared-memory tiled fast path (#4, #8).
- Boundary handling shared by both kernels: `REFLECT`, `REPLICATE` and `CONSTANT` (#4).
- Structuring-element builders (#6).
- Python API: `erosion`, `dilation`, `opening`, `closing`, `gradient`, `top_hat` and `black_hat` (#8).
- Dynamic search directories when loading the compiled `_C` extension.

### Changed

- Development status raised to Pre-Alpha.

### Fixed

- Formatting and typing inconsistencies reported by ruff and mypy.

## [0.0.1] - 2026-07-10

### Added

- Initial scaffolding: build system, operator registration and stubs for the planned alpha feature set (#1, #2).

[Unreleased]: https://github.com/vhrabar/serron/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/vhrabar/serron/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/vhrabar/serron/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/vhrabar/serron/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vhrabar/serron/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/vhrabar/serron/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/vhrabar/serron/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/vhrabar/serron/releases/tag/v0.0.1
