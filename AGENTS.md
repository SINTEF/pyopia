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
uv run pytest                  # everything CI runs
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

- Docstrings: NumPy style, as concise as possible - describe current
  behaviour only, not the rationale or history for why it got that way.
  That belongs in the relevant Issue.
- Pipeline steps are config-driven classes operating on a shared `data` dict
  - see `pyopia/pipeline.py`.
- Version numbering is MAJOR.MINOR.PATCH - see the README's "Version
  numbering" section for what bumps which.

## Working practices

- Verify claims against real evidence - actual command output, logs, diffs -
  rather than asserting from assumption or memory. If something seems flaky
  or wrong, find the real cause before concluding.
- Don't add tests, error handling, or abstractions for scenarios that aren't
  real yet. Do add tests for genuine, demonstrated bugs or new logic.
- Prefer one unified, consistent mechanism over two parallel/special-cased
  implementations, even if it touches more call sites.
- Only stage/commit files that were actually intentionally edited - check
  `git status` before committing, since tool or test runs can leave stray
  files behind.
- Show draft text - commit messages, PR descriptions, Issue bodies, PR/Issue
  comments - to the human you're working with before creating or posting it,
  even when it feels too small to bother asking about.
- Every PR body must use `Closes #N`/`Fixes #N` for every issue it
  addresses, so merging auto-closes them. Note: GitHub only recognises these
  keywords when the PR's base is the repo's default branch - a PR stacked on
  a feature branch won't auto-link issues until it's retargeted.
- Commit messages and PR/Issue text: short and factual, explaining what
  changed and why. No hand-wringing, no defensive over-justification.
- PR descriptions for combined/staged work: headed bullet sections, one
  bullet per logical change, each linking to the commit that makes it.
- No emojis unless explicitly asked for.

## Before submitting

All PRs need a human review and must pass CI - see the README's
Contributions section for the full list of guidelines this repo expects.
