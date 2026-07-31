PyOPIA
===============================

A Python Ocean Particle Image Analysis toolbox

PyOPIA processes images of particles suspended in water (e.g. from SilCam, holographic, or UVP imaging systems) into particle size, shape, and concentration statistics.

# Quick tryout of PyOPIA

1) Install [uv](https://docs.astral.sh/uv/getting-started/installation)
2) Initialize a PyOPIA project with example data (`--example-data` downloads a small example image dataset and generates a matching `config.toml`), then run processing:
```bash
uvx --python 3.12 --from pyopia[classification] pyopia init-project pyopiatest --example-data
```
```bash
cd pyopiatest
```
```bash
uvx --python 3.12 --from pyopia[classification] pyopia process config.toml
```
3) Inspect the processed particle statistics in the processed/ folder
4) Merge the individual processed image STATS files into a single STATS.nc file, then create a montage of the processed STATS.nc

```bash
uvx --python 3.12 --from pyopia[classification] pyopia merge-mfdata processed
```

```bash
uvx --python 3.12 --from pyopia[classification] pyopia make-montage processed/pyopiatest-STATS.nc
```
5) This creates `montage.png` in the current folder - open it to see a single image made up of all the processed particle images.

See the documentation for more information on how to install and use PyOPIA.

# Running with Docker

A prebuilt container image is published to GitHub Container Registry on every release. Docker is worth reaching for if you'd rather not install PyOPIA's dependencies directly: it avoids the install overhead of heavier optional dependencies (e.g. TensorFlow/PyTorch for classification), guarantees a consistent, reproducible environment regardless of your host OS, and is well suited to running on servers or HPC systems.

## One-off invocation

From any directory that contains a `config.toml`:

```bash
docker run --rm \
    --user $(id -u):$(id -g) \
    -v "$PWD:$PWD" -w "$PWD" \
    ghcr.io/sintef/pyopia:latest \
    process config.toml
```

The 1:1 volume mount (`-v "$PWD:$PWD"`) makes the container see your current directory at the same path as the host, so an existing `config.toml` with absolute paths under `$PWD` works unchanged.

**If your config references paths outside `$PWD`** — most commonly a classifier weights file — add a matching bind mount for each of them.
The target (right of the colon) must equal the source (left) so the path in the config resolves unchanged. 
For a config like

```toml
[steps.classifier]
pipeline_class = "pyopia.classify_torch.Classify"
model_path = "/home/you/models/classifier.pt"
```

the invocation becomes

```bash
docker run --rm \
    --user $(id -u):$(id -g) \
    -v "$PWD:$PWD" -w "$PWD" \
    -v /home/you/models:/home/you/models:ro \
    ghcr.io/sintef/pyopia:latest \
    process config.toml
```

Any PyOPIA CLI command works, e.g. `docker run --rm ghcr.io/sintef/pyopia:latest --help`.

## Using docker compose

Each GitHub release attaches a `compose.yaml` as an asset. 
Download it once per project:

```bash
curl -LO https://github.com/SINTEF/pyopia/releases/latest/download/compose.yaml
docker compose run --rm pyopia                       # runs `pyopia process config.toml`
docker compose run --rm pyopia --help                # or any other CLI argument
```

The compose file has commented-out stubs for paths that live outside `$PWD` — in particular the classifier weights file. 
Edit `compose.yaml` to uncomment the line and point it at your actual host path, e.g.

```yaml
    volumes:
      - ${PWD}:${PWD}
      - /home/you/models:/home/you/models:ro    # classifier weights
```

The target must match the source so the config-referenced path resolves unchanged. 
Override the default config filename with `PYOPIA_CONFIG`:

```bash
PYOPIA_CONFIG=my_run.toml docker compose run --rm pyopia
```

# Documentation:

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pyopia.readthedocs.io) [![Documentation](https://readthedocs.org/projects/pyopia/badge/?version=latest)](https://pyopia.readthedocs.io/en/latest/?badge=latest)

[pyopia.readthedocs.io](https://pyopia.readthedocs.io)
# Current status:

- Under development. See/register issues, [here](https://github.com/SINTEF/pyopia/issues)

----
# Design principles

- PyOPIA is instrument-agnostic at its core: SilCam, holographic, and UVP support are all built as pluggable instrument modules on top of a shared `Pipeline`.
- Processing is config-driven: a `Pipeline` is built from a TOML/dict `settings` object describing an ordered list of steps, each mapping to a Python class. Steps update a shared `data` dict as the pipeline runs.
- Heavy, optional dependencies (e.g. TensorFlow/PyTorch for classification) are kept out of the core install and available via extras (`pyopia[classification]`, `pyopia[classification-torch]`).
- Multi-file, multi-core processing is supported via chunked/parallel processing (`pyopia process --num-chunks`).

## Contributions

We welcome additions and improvements to the code! We request that you follow a few guidelines. These are in place to make sure the code improves over time.

1. All code changes must be submitted as pull requests, either from a branch or a fork.
2. Good documentation of the code is needed for PyOPIA to succeed and so please include up-to-date docstrings as you make changes, so that the auto-build on readthedocs is complete and useful for users. (A version of the new docs will compile when you make a pull request and a link to this can be found in the pull request checks)
3. All pull requests are required to pass all tests before merging. Please do not disable or remove tests just to make your branch pass the pull request.
4. All pull requests must be reviewed by a person. The benefits from code review are plenty, but we like to emphasise that code reviews help spreading the awareness of code changes. Please note that code reviews should be a pleasant experience, so be pleasant, polite and remember that there is a human being with good intentions on the other side of the screen.
5. All contributions are linted with flake8. We recommend that you run flake8 on your code while developing to fix any issues as you go. We recommend using autopep8 to autoformat your Python code (but please check the code behaviour is not affected by autoformatting before pushing). This makes flake8 happy, and makes it easier for us all to maintain a consistent and readable code base.

## Docstrings

Use the NumPy style in docstrings. See style guide [here](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes)

## Testing

PyOPIA's test suite lives in `pyopia/tests/` and runs via `pytest` (see `uv run pytest` below). A few things are useful to know before running or adding to it:

**Markers**: some tests are tagged with pytest markers to indicate how expensive they are.

- `@pytest.mark.slow` - tests that do real network downloads and/or real model inference (e.g. downloading the example classifier model and running real predictions on it). These run in routine CI, but you can skip them for a fast local feedback loop:
  ```bash
  uv run pytest -m "not slow"
  ```
- `@pytest.mark.training` - tests that train a model from scratch (currently, a notebook that trains a DINOv2-based classifier). These never run in routine CI - only manually, or on a schedule - since they involve a real, uncapped multi-epoch training run rather than a check of PyOPIA's own correctness:
  ```bash
  uv run pytest -m training
  ```

Unmarked tests are fast and have no external dependencies; they always run.

**Shared fixtures**: tests that need real example data (an example image, the trained classifier model, the classifier training database, an example hologram) get it from session-scoped fixtures defined in `pyopia/tests/conftest.py`, rather than each downloading their own copy. The download happens once per test run and is shared across every test file that needs it.

**Notebooks**: `pyopia/tests/test_notebooks.py` executes the notebooks in `notebooks/` and `docs/notebooks/` to check they still run against the current codebase. Each notebook is its own parametrized test (`test_notebook[<name>.ipynb]`), tagged `slow`/`training` as above where relevant. Not every notebook is included - a couple depend on state produced by another notebook, or by a user's own prior processing run, and would fail if executed standalone; see the comments in `test_notebooks.py` for which ones and why. These tests also get a longer timeout (1800s) and up to 2 automatic retries on failure, since running a Jupyter kernel via nbconvert has shown real, platform-specific flakiness on macOS CI runners rather than a reproducible bug.

Please do not disable or remove tests just to make a pull request pass - see Contributions guideline 3 above.

# Installing

## For users

Users are expected to be familiar with Python. Please refer to the recommended installation instructions provided on the documentation pages, [here](https://pyopia.readthedocs.io/en/latest/intro.html#installing)

## For developers from source


Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

1. Navigate to the folder where you want to install pyopia using the 'cd' command.

If you use git:
Download repository from github, and move into the new directory:

```bash
git clone https://github.com/SINTEF/pyopia.git
cd pyopia
```

For the next steps, you need to be located in the PyOPIA root directory that contains the file 'pyproject.toml'.

2. Install all requirements with

```bash
uv sync --all-extras --dev
```

3. (optional) Run local tests (see the Testing section above for markers and how to run a fast subset):

```bash
uv run pytest
```

#### Version numbering

The version number of PyOPIA is split into three sections: MAJOR.MINOR.PATCH

* MAJOR: Changes in high-level pipeline use and/or data output that are not backwards-compatible.
* MINOR: New features that are backwards-compatible.
* PATCH: Backwards-compatible bug fixes or enhancements to existing functionality

## Build docs locally

```bash
uv sync --extra classification --group docs
uv run sphinx-build -b html docs/ docs/_build/html
```

----
# License

PyOPIA is licensed under the BSD3 license. See LICENSE. All contributors should be recognised & acknowledged.
