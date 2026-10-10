#!/usr/bin/env bash
#
# Lint and static-check pipeline.


set -uo pipefail

CLANG_FORMAT_VERSION=18.1.8

cd "$(dirname "$0")/.."

in_ci() { [ -n "${GITHUB_ACTIONS:-}" ]; }

# Tracked C++/CUDA sources, empty when there are none.
cpp_files() { git ls-files | grep -E '\.(cpp|cc|cxx|h|hpp|cu|cuh)$' || true; }

stage_ruff_format() {
    echo '== Ruff format =='
    uv run --no-sync ruff format --check .
}

stage_ruff_check() {
    echo '== Ruff check =='
    if in_ci; then
        uv run --no-sync ruff check --output-format=github .
    else
        uv run --no-sync ruff check .
    fi
}

stage_pyrefly() {
    echo '== Pyrefly =='
    # Paths and excludes come from [tool.pyrefly] in pyproject.toml, so this takes no path argument.
    if in_ci; then
        uv run --no-sync pyrefly check --output-format=full-text-with-github
    else
        uv run --no-sync pyrefly check
    fi
}

stage_clang_format() {
    echo '== clang-format =='
    local files
    files="$(cpp_files)"
    if [ -z "$files" ]; then
        echo 'No C++/CUDA files to check.'
        return 0
    fi
    # shellcheck disable=SC2086
    uvx "clang-format@${CLANG_FORMAT_VERSION}" --dry-run --Werror $files
}

stage_torch_floor() {
    echo '== Torch floor =='
    python3 scripts/check_torch_floor.py
}

stage_fix() {
    local rc=0 files

    echo '== Ruff check --fix =='
    uv run --no-sync ruff check --fix . || rc=1

    echo '== Ruff format =='
    uv run --no-sync ruff format . || rc=1

    echo '== clang-format -i =='
    files="$(cpp_files)"
    if [ -z "$files" ]; then
        echo 'No C++/CUDA files.'
    # shellcheck disable=SC2086
    elif uvx "clang-format@${CLANG_FORMAT_VERSION}" -i $files; then
        echo 'Formatted C++/CUDA files.'
    else
        rc=1
    fi

    echo 'Note: pyrefly is type-check only and has no autofix.'
    return $rc
}

ALL_STAGES=(ruff-format ruff-check pyrefly clang-format torch-floor)

run_stage() {
    case "$1" in
    ruff-format) stage_ruff_format ;;
    ruff-check) stage_ruff_check ;;
    pyrefly) stage_pyrefly ;;
    clang-format) stage_clang_format ;;
    torch-floor) stage_torch_floor ;;
    fix) stage_fix ;;
    *)
        # A bad stage name is a usage error
        echo "lint.sh: unknown stage '$1'" >&2
        echo "stages: ${ALL_STAGES[*]} fix all" >&2
        exit 2
        ;;
    esac
}

stages=("$@")
if [ "${#stages[@]}" -eq 0 ] || [ "${stages[0]}" = all ]; then
    stages=("${ALL_STAGES[@]}")
fi

rc=0
for stage in "${stages[@]}"; do
    run_stage "$stage" || rc=1
done
exit $rc
