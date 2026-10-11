#!/usr/bin/env bash
# Fails if _C_cuda links the CUDA runtime or driver.
set -euo pipefail

target="${1:?usage: $0 <_C_cuda.so | wheel>}"
if [[ "$target" == *.whl ]]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    python3 -m zipfile -e "$target" "$tmp"
    target="$(find "$tmp" -name '_C_cuda*.so' | head -n 1)"
    [[ -n "$target" ]] || { echo "no _C_cuda library in the wheel" >&2; exit 1; }
fi

status=0
needed="$(readelf -d "$target" | grep NEEDED || true)"
if grep -E 'libcudart\.so|libcuda\.so' <<<"$needed"; then
    echo "error: $target has a NEEDED entry for the CUDA runtime or driver" >&2
    status=1
fi
undefined="$(nm -D --undefined-only "$target" | awk '{print $NF}')"
if grep -E '^(cu[A-Z]|cuda|__cuda)' <<<"$undefined"; then
    echo "error: $target has undefined CUDA symbols" >&2
    status=1
fi

[[ $status -eq 0 ]] && echo "ok: $target links neither the CUDA runtime nor the driver"
exit $status
