"""
Shared markdown-table rendering
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def markdown_table(header: Sequence[str], rows: Iterable[Sequence[str]], aligns: str) -> str:
    """Render a markdown table, padded so the raw text stays readable.

    :param header: column titles.
    :param rows: one sequence of cells per row.
    :param aligns: one character per column, ``"l"`` (left) or ``"r"`` (right).
    :returns: the table, without a trailing newline.
    """
    body = [list(row) for row in rows]
    widths = [max(len(cell) for cell in (head, *(row[i] for row in body))) for i, head in enumerate(header)]

    def render(cells: Sequence[str]) -> str:
        padded = [c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(cells, widths, aligns, strict=True)]
        return "| " + " | ".join(padded) + " |"

    rules = ["-" * (w - 1) + ":" if a == "r" else "-" * w for w, a in zip(widths, aligns, strict=True)]
    return "\n".join([render(header), "|" + "|".join(f" {r} " for r in rules) + "|", *(render(row) for row in body)])
