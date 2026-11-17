#!/usr/bin/env python3
"""Compile the CUDA kernel sources with hipRTC for AMD architectures. Needs no GPU.

    python3 scripts/rocm/compile_device.py [--arch gfx942 ...] [--csrc DIR]

Uses libhiprtc from a ROCm build of torch if there is one (run it with that torch's python), else HIPRTC_LIB, else
the system ROCm (ROCM_PATH, default /opt/rocm). A system hipRTC can lack the libc headers the torch headers include,
so prefer torch's. The torch headers themselves come from whichever torch is installed.
"""

import argparse
import ctypes
import importlib.util
import os
import sys
from ctypes import byref, c_char_p, c_size_t, c_void_p
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ("forward_kernels.cu", "backward_kernels.cu")
ARCHS = ("gfx90a", "gfx942", "gfx1100", "gfx1201")
TORCH_TARGET = "0x020d000000000000ULL"  # torch 2.13, as in CMakeLists.txt


def torch_dir() -> Path:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        sys.exit("torch is not installed; it is needed for its headers")
    return Path(spec.origin).parent


def load_hiprtc() -> ctypes.CDLL:
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    candidates = [
        os.environ.get("HIPRTC_LIB", ""),
        str(torch_dir() / "lib" / "libhiprtc.so"),
        str(rocm / "lib" / "libhiprtc.so"),
        "libhiprtc.so",
    ]
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    sys.exit("no libhiprtc found; set HIPRTC_LIB or ROCM_PATH")


def compile_one(rtc: ctypes.CDLL, source: Path, arch: str, includes: list[str]) -> str:
    """Returns the compiler log if the compile fails, else an empty string."""
    prog = c_void_p()
    rtc.hiprtcCreateProgram(byref(prog), source.read_text().encode(), source.name.encode(), 0, None, None)
    opts = [
        f"--offload-arch={arch}",
        "-std=c++20",
        "-DTORCH_STABLE_ONLY",
        f"-DTORCH_TARGET_VERSION={TORCH_TARGET}",
        *[f"-I{d}" for d in includes],
    ]
    arr = (c_char_p * len(opts))(*[o.encode() for o in opts])
    failed = rtc.hiprtcCompileProgram(prog, len(opts), arr)
    size = c_size_t()
    rtc.hiprtcGetProgramLogSize(prog, byref(size))
    log = ctypes.create_string_buffer(size.value)
    rtc.hiprtcGetProgramLog(prog, log)
    rtc.hiprtcDestroyProgram(byref(prog))
    return log.value.decode(errors="replace") if failed else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", nargs="+", default=list(ARCHS))
    parser.add_argument("--csrc", type=Path, default=ROOT / "packages/serron/csrc")
    args = parser.parse_args()

    includes = [str(torch_dir() / "include"), str(args.csrc), str(args.csrc / "compat")]
    rocm_include = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "include"

    rtc = load_hiprtc()
    status = 0
    for name in KERNELS:
        for arch in args.arch:
            source = args.csrc / "gpu" / "morphology" / name
            log = compile_one(rtc, source, arch, includes)
            if "hip/hip_runtime.h' file not found" in log and rocm_include.is_dir():
                log = compile_one(rtc, source, arch, [*includes, str(rocm_include)])
            print(f"{'FAIL' if log else 'ok  '} {name} {arch}")
            if log:
                status = 1
                errors = [ln for ln in log.splitlines() if "error" in ln]
                print("\n".join("     " + ln for ln in errors[:3]))
    return status


if __name__ == "__main__":
    sys.exit(main())
