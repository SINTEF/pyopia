'''
Executes notebooks end-to-end to check they still run against the current codebase.

Each notebook is its own parametrized test (test_notebook[<name>.ipynb]), so a failure
in one notebook is reported individually instead of as one monolithic pass/fail across
all of them. Notebooks are executed with their own directory as the working directory,
since several of them load a local config.toml, or write/read files using paths that
are relative to where the notebook itself lives.

See issue #403 for why this replaced a single un-parametrized test_notebooks() function,
and #237 for the docs/notebooks coverage this adds.
'''

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parents[2]

# Real network downloads and/or real pipeline runs, but no model training: acceptable
# to run in routine CI, just excluded from a fast local `pytest -m "not slow"` loop.
SLOW_NOTEBOOKS = [
    REPO_ROOT / 'notebooks' / 'single-image-stats.ipynb',
    REPO_ROOT / 'notebooks' / 'pipeline-holo.ipynb',
    REPO_ROOT / 'notebooks' / 'single-image-stats-holo.ipynb',
    REPO_ROOT / 'notebooks' / 'pyopia-classifier' / 'pyopia-default-classifier.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'background_correction.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'montaging.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'stats.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'exploring_pipeline_data.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'pipeline_step_by_step.ipynb',
]

# No network/model dependency: fast enough to run alongside the rest of the suite.
FAST_NOTEBOOKS = [
    REPO_ROOT / 'docs' / 'notebooks' / 'cli.ipynb',
]

# Markdown-only notebooks with no code cells to execute. Included for completeness
# (#237) but they provide no code-execution coverage on their own - there's nothing in
# them that could fail due to a code/API regression.
DOCS_ONLY_NOTEBOOKS = [
    REPO_ROOT / 'docs' / 'notebooks' / 'toml_config.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'processing_raw_data.ipynb',
    REPO_ROOT / 'docs' / 'notebooks' / 'big_datasets.ipynb',
]

# Trains a real model from scratch (a DINOv2 backbone + 30 real training epochs, no
# reduced-epoch CI mode). Never runs in routine CI; run manually or on a schedule only,
# with `pytest -m training`.
TRAINING_NOTEBOOKS = [
    REPO_ROOT / 'notebooks' / 'pyopia-classifier' / 'pyopia-torch-dinov2-classifier-train.ipynb',
]

# Not included: docs/notebooks/STATSnc.ipynb loads a pre-existing 'test-STATS.nc' file
# that no notebook produces at that same relative path when run in isolation - it's
# designed to be read by a user who already has their own processed stats file sitting
# in their working directory, not to be run standalone. Adding it here would need
# either chaining it after stats.ipynb in a shared working directory (fragile, couples
# two supposedly-independent tests) or changing the notebook itself to generate its own
# input first, which is a documentation content change rather than a test-infra one.

NOTEBOOK_PARAMS = (
    [pytest.param(nb, id=nb.name, marks=pytest.mark.slow) for nb in SLOW_NOTEBOOKS]
    + [pytest.param(nb, id=nb.name) for nb in FAST_NOTEBOOKS]
    + [pytest.param(nb, id=nb.name) for nb in DOCS_ONLY_NOTEBOOKS]
    + [pytest.param(nb, id=nb.name, marks=[pytest.mark.slow, pytest.mark.training]) for nb in TRAINING_NOTEBOOKS]
)

# Overrides pyproject.toml's 600s default for every test in this module. Confirmed on a
# real CI run (SINTEF/pyopia PR #404): pipeline-holo.ipynb - an ordinary, non-training
# notebook - completed in ~2 minutes on Ubuntu and Windows, but exceeded 600s on macOS.
# The traceback showed it stuck waiting on the Jupyter kernel's socket, which points to
# nbconvert/Jupyter-kernel execution being disproportionately slow on macOS CI runners
# specifically, not the underlying computation actually taking longer. Applies to all
# notebook tests here, not just the training one, since any of them could hit it.
#
# Also auto-retries a failing notebook test up to twice (with a short delay between
# attempts) before letting it fail for real, since the underlying issue looks like CI
# infra flakiness (a stuck kernel socket) rather than a reproducible bug. Scoped to this
# module only - retrying the rest of the suite could mask a real, reproducible failure.
pytestmark = [pytest.mark.timeout(1800), pytest.mark.flaky(reruns=2, reruns_delay=10)]


@pytest.mark.parametrize('notebook_path', NOTEBOOK_PARAMS)
def test_notebook(notebook_path):
    with open(notebook_path, encoding='utf8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor()
    ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
