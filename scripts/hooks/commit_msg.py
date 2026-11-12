#!/usr/bin/env python3
"""
Commit-msg hook: the subject line must be a conventional commit.

``type(scope): subject``, where the scope is optional and ``type`` is one of the types below. Merge, revert and
fixup commits that git generates are let through.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TYPES = ("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert")
MAX_SUBJECT = 120

_SUBJECT = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([^()\r\n]+\))?!?: \S")
_GENERATED = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")


def problems(message: str) -> list[str]:
    """
    What is wrong with a commit message.

    :param message: the whole message as git hands it to the hook, comments included.
    :returns: one sentence per problem, empty when the message is fine.
    """
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["the commit message is empty"]

    subject = lines[0]
    if subject.startswith(_GENERATED):
        return []

    found: list[str] = []
    if not _SUBJECT.match(subject):
        found.append(f"the subject must look like 'type(scope): what changed', with type one of {', '.join(TYPES)}")
    if len(subject) > MAX_SUBJECT:
        found.append(f"the subject is {len(subject)} characters, keep it to {MAX_SUBJECT} or fewer")
    if subject.rstrip() != subject:
        found.append("the subject ends in whitespace")
    if subject.endswith("."):
        found.append("the subject ends in a period")
    if len(lines) > 1 and lines[1].strip():
        found.append("leave a blank line between the subject and the body")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: commit_msg.py MESSAGE_FILE", file=sys.stderr)
        return 2
    found = problems(Path(argv[1]).read_text(encoding="utf-8"))
    if not found:
        return 0
    print("Commit message rejected:", file=sys.stderr)
    for problem in found:
        print(f"  - {problem}", file=sys.stderr)
    print("Example: feat(autotune): add per-problem algorithm tuning #66", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
