"""
Runs the separable-path cases under whatever path ``SERRON_SEPARABLE_MIN_K`` selects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import serron
from serron import BorderMode

CASES = [
    ("erosion", BorderMode.REFLECT, (1, 2, 12, 14), (21, 21)),
    ("erosion", BorderMode.REPLICATE, (1, 2, 12, 14), (21, 21)),
    ("erosion", BorderMode.CONSTANT, (1, 2, 12, 14), (21, 21)),
    ("dilation", BorderMode.REFLECT, (1, 2, 12, 14), (21, 21)),
    ("dilation", BorderMode.REPLICATE, (1, 2, 12, 14), (21, 21)),
    ("dilation", BorderMode.CONSTANT, (1, 2, 12, 14), (21, 21)),
    ("erosion", BorderMode.REPLICATE, (2, 3, 17, 23), (21, 3)),
    ("dilation", BorderMode.REFLECT, (2, 3, 17, 23), (3, 21)),
    ("erosion", BorderMode.CONSTANT, (1, 1, 5, 7), (31, 31)),
    ("dilation", BorderMode.REPLICATE, (1, 1, 5, 7), (31, 31)),
]


def run_cases(device: torch.device | str) -> dict[str, torch.Tensor]:
    """Run every case in :data:`CASES` and collect its forward and gradients.

    :param device: device to run on.
    :returns: ``{"<case index>.<forward|grad_input|grad_kernel>": tensor}``, on CPU.
    """
    out: dict[str, torch.Tensor] = {}
    for i, (op, border, shape, ksize) in enumerate(CASES):
        gen = torch.Generator(device="cpu").manual_seed(0x5E770 + i)
        x = torch.randn(*shape, generator=gen, dtype=torch.float64).to(device).requires_grad_(True)
        kernel = torch.zeros(shape[1], *ksize, dtype=torch.float64, device=device).requires_grad_(True)
        go = torch.randn(*shape, generator=gen, dtype=torch.float64).to(device)

        forward = getattr(serron.functional, op)(x, kernel, border=border)
        forward.backward(go)
        assert x.grad is not None and kernel.grad is not None

        out[f"{i}.forward"] = forward.detach().cpu()
        out[f"{i}.grad_input"] = x.grad.cpu()
        out[f"{i}.grad_kernel"] = kernel.grad.cpu()
    return out


def main() -> None:
    """Write :func:`run_cases` results to the path given by ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.save(run_cases(args.device), args.out)


if __name__ == "__main__":
    main()
