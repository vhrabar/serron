# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CUDA backward kernels for `erode`/`dilate`, wired through new `_ErodeFunction`/`_DilateFunction` autograd bindings, 
    so gradients now flow through `erosion`, `dilatation`, and the composite ops (`opening`, `closing`, `gradient`, `top_hat`,
    `black_hat`) into both the input and the structuring element, including the learnable layers (`Erosion2d`, `Dilation2d`,
    `Opening2d`, `Closing2d`), which can now be optimized E2E (#13, #15). CUDA only for now; CPU tensors still raise on `.backward()`.


## [0.1.1] - 2026-07-25

### Added

- CPU backend for `erode` and `dilate`, registered on the `CPU` dispatch key so CPU tensors no longer require a CUDA build (#17).
- CPU-only and CUDA 13.x build paths in the installation instructions (#17).
- CPU backend for `erode` and `dilate`, registered on the `CPU` dispatch key so CPU tensors no longer require a CUDA build (#17).
- CPU-only and CUDA 13.x build paths in the installation instructions (#17).

## [0.1.0] - 2026-07-18

### Added

- CUDA grayscale erosion and dilation kernels, including the shared-memory tiled fast path (#4, #8).
- Boundary handling shared by both kernels: `REFLECT`, `REPLICATE` and `CONSTANT` (#4).
- Structuring-element builders (#6).
- Python API: `erosion`, `dilatation`, `opening`, `closing`, `gradient`, `top_hat` and `black_hat` (#8).
- Dynamic search directories when loading the compiled `_C` extension.

### Changed

- Development status raised to Pre-Alpha.

### Fixed

- Formatting and typing inconsistencies reported by ruff and mypy.

## [0.0.1] - 2026-07-10

### Added

- Initial scaffolding: build system, operator registration and stubs for the planned alpha feature set (#1, #2).

[Unreleased]: https://github.com/vhrabar/serron/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/vhrabar/serron/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/vhrabar/serron/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/vhrabar/serron/releases/tag/v0.0.1
