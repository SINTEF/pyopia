# syntax=docker/dockerfile:1.6
#
# PyOPIA runtime container
#
# Builds an image with the PyOPIA CLI (`pyopia process <config.toml>`) as
# the entrypoint. Data and config are expected to be mounted via volumes.
# See README or the docker-compose example for the 1:1 host-path mounting
# convention that lets existing pyopia configs run unchanged.
#
#   docker build -t pyopia:local \
#       --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
#
#   docker run --rm \
#       -v /data/silcam:/data/silcam:ro \
#       -v /data/output:/data/output \
#       -v $(pwd)/config.toml:/data/output/config.toml:ro \
#       pyopia:local process /data/output/config.toml

FROM python:3.12-slim-bookworm AS base

# uv, as a static binary from the official image. Pin the version to
# match astral-sh/setup-uv@v5 in .github/workflows/build-and-test.yml
# so Docker builds behave identically to CI.
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

# System runtime deps:
#   libgl1, libglib2.0-0  — cv2 / scikit-image bindings
#   libhdf5-dev           — h5py / h5netcdf output (stats files)
#   ca-certificates       — PyPI TLS + model downloads via gdown
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libhdf5-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv in-Docker hygiene:
#   COMPILE_BYTECODE=1  precompile .pyc for faster first import
#   LINK_MODE=copy      avoid hardlink errors across build-context fs boundaries
#   SYSTEM_PYTHON=1     target the base image's Python (no venv needed)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

# Non-root user. UID/GID configurable at build time so mounted volumes
# don't get root-owned files on the host. Defaults match most Linux
# desktops; GitHub Actions runners use UID 1001 but we normalise to 1000
# for local use.
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID pyopia && useradd -m -u $UID -g $GID -s /bin/bash pyopia

WORKDIR /workspace

# Copy source and install. Installing from source (rather than
# `pyopia==<tag>` from PyPI) avoids the race between this workflow and
# pypi.yml on release, and guarantees the image matches the repo state
# at the tagged commit. 
COPY --chown=pyopia:pyopia . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[classification-torch,classification]"

USER pyopia

# PyOPIA's CLI is a typer app registered as the `pyopia` entry point
# (see pyproject.toml: [project.scripts] pyopia = "pyopia.cli:app").
# Everything after the image name in `docker run` is passed straight to
# the CLI, e.g. `docker run … pyopia:local process /config/config.toml`.
ENTRYPOINT ["pyopia"]
CMD ["--help"]
