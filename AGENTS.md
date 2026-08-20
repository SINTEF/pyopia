# AGENTS.md

Operational guidance for AI coding agents working in this repository. See the
README's "Using AI tools" section for the human-facing policy this supports:
AI tools are welcome as an aid, but whoever submits a PR is responsible for
the code in it.

## Setup

```bash
uv sync --all-extras --dev
```

## Testing

```bash
uv run pytest -m "not slow"    # fast local loop, no network/model downloads
uv run pytest                  # everything CI runs, including the DINOv2 training notebook
```

Tests are marked `slow` (real network downloads and/or real model inference)
where relevant; excluded only from the fast local loop above, still run in
routine CI.

## Linting

```bash
uv run flake8 pyopia
```

autopep8 is fine for autoformatting, but verify behavior is unchanged before
committing.

## Conventions

- Docstrings: NumPy style.
- Pipeline steps are config-driven classes operating on a shared `data` dict
  - see `pyopia/pipeline.py`.
- Version numbering is MAJOR.MINOR.PATCH - see the README's "Version
  numbering" section for what bumps which.

## Before submitting

All PRs need a human review and must pass CI - see the README's
Contributions section for the full list of guidelines this repo expects.
