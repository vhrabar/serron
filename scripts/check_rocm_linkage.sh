#!/usr/bin/env bash
set -euo pipefail

target="${1:?usage: $0 <_C_rocm.so | wheel>}"
if [[ "$target" == *.whl ]]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    python3 -m zipfile -e "$target" "$tmp"
    target="$(find "$tmp" -name '_C_rocm*.so' | head -n 1)"
    [[ -n "$target" ]] || { echo "no _C_rocm library in the wheel" >&2; exit 1; }
fi

status=0
needed="$(readelf -d "$target" | grep NEEDED || true)"
if grep -E 'libamdhip64|libhiprtc|libhsa-runtime|librocm' <<<"$needed"; then
    echo "error: $target has a NEEDED entry for a ROCm library; the HIP runtime must come from torch" >&2
    status=1
fi

undefined="$(nm -D --undefined-only "$target" | awk '{print $NF}')"
if grep -E '^(hip|__hip|hsa_)' <<<"$undefined"; then
    echo "error: $target has undefined HIP symbols; they are resolved at run time" >&2
    status=1
fi
if c++filt <<<"$undefined" | grep -E 'c10::|at::|torch::'; then
    echo "error: $target uses the torch C++ API outside the stable ABI" >&2
    status=1
fi

# Each kernel file is embedded as one clang offload bundle per architecture and one portable image.
bundles="$({ grep -a -o '__CLANG_OFFLOAD_BUNDLE__' "$target" || true; } | wc -l)"
portable="$({ grep -a -o 'hip-spirv64-amd-amdhsa--amdgcnspirv' "$target" || true; } | wc -l)"
architectures="$({ strings -a "$target" | grep -o 'hipv4-amdgcn-amd-amdhsa--gfx[0-9a-f]*' || true; } | sed 's/.*--//' | sort -u | tr '\n' ' ')"
if (( portable < 2 )); then
    echo "error: $target has $portable portable images, expected one for each of the forward and backward kernels" >&2
    status=1
fi
if [[ -z "$architectures" ]]; then
    echo "error: $target has no native kernel images" >&2
    status=1
fi

[[ $status -eq 0 ]] && echo "ok: $target links no HIP runtime, uses no torch C++ API, and holds $bundles kernel images (${architectures}+ portable)"
exit $status
