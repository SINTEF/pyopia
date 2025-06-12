"""
A high level test for executing the ipynb notebooks in the notebooks folder.

Can only be run from top-level directory (i.e. with 'poetry run pytest -v')

"""

from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from pathlib import Path


def test_notebooks():
    notebooks = sorted(Path("notebooks/").rglob("*.ipynb"))
    notebooks.append("docs/notebooks/background_correction.ipynb")
    for notebook_filename in notebooks:
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor()
            print("running", notebook_filename)
            ep.preprocess(nb, {"metadata": {"path": "notebooks/"}})
            print(notebook_filename, "complete")
