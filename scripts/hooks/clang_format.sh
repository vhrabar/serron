#!/usr/bin/env bash

set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

root="$(git rev-parse --show-toplevel)"
version="$(sed -n 's/^CLANG_FORMAT_VERSION=//p' "$root/scripts/lint.sh")"
if [ -z "$version" ]; then
    echo "clang_format.sh: could not read CLANG_FORMAT_VERSION from scripts/lint.sh" >&2
    exit 2
fi

cd "$root"
exec uvx "clang-format@${version}" -i "$@"
