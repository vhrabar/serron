# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/vhrabar/serron/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/vhrabar/serron/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/vhrabar/serron/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vhrabar/serron/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/vhrabar/serron/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/vhrabar/serron/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/vhrabar/serron/releases/tag/v0.0.1
