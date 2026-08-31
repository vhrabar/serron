# syntax=docker/dockerfile:1.7
#
#
#   docker build -t serron .
#   docker run --rm --gpus all serron                       # full suite: tests + benchmarks
#   docker run --rm --gpus all serron pytest tests/ -q      # tests only
#   docker run --rm -it  --gpus all serron bash             # interactive shell
#


ARG CUDA_VERSION=13.3.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu26.04

ARG CUDA_ARCH=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ENV UV_PYTHON_INSTALL_DIR=/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    PYTHONUNBUFFERED=1

ENV CMAKE_ARGS="${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=}${CUDA_ARCH}"

WORKDIR /app

# STage I: Deps
COPY pyproject.toml uv.lock ./
COPY packages/serron/pyproject.toml packages/serron/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cu132 --no-install-workspace

# Stage II: Build
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cu132 --no-editable

RUN useradd --create-home --uid 1000 app && chown -R app:app /app /opt/venv
USER app

ENTRYPOINT ["uv", "run", "--no-sync"]
CMD ["pytest", "tests/", "benchmarks/"]
