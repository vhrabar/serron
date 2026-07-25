# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CPU backend for `erode` and `dilate`, registered on the `CPU` dispatch key so CPU tensors no longer require a CUDA build (#17).
- CPU-only and CUDA 13.x build paths in the installation instructions (#17).

[Unreleased]: https://github.com/vhrabar/serron/compare/v0.1.0...HEAD
