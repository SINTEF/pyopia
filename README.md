PyOPIA
===============================

A Python Ocean Particle Image Analysis toolbox

# Quick tryout of PyOPIA

1) Install [uv](https://docs.astral.sh/uv/getting-started/installation)
2) Initialize PyOPIA project with a small example image dataset and run processing
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
uvx --python 3.12 --from pyopia[classification] pyopia make-montage processed\pyopiatest-STATS.nc
```
5) Visualise the monatge of all processed singular particle images in one
```bash
montage.png
```
Will show you a montage of all the processed particle images in one.

See the documentation for more information on how to install and use PyOPIA.

# Documentation:

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pyopia.readthedocs.io) [![Documentation](https://readthedocs.org/projects/pyopia/badge/?version=latest)](https://pyopia.readthedocs.io/en/latest/?badge=latest)

[pyopia.readthedocs.io](https://pyopia.readthedocs.io)
# Current status:

- Under development. See/regester issues, [here](https://github.com/SINTEF/pyopia/issues)

----
# Development targets for PyOpia:

1) Allow nonfamiliar users to install and use PyOpia, and to contribute & commit code changes
2) Not hardware specific
3) Smaller dependency list than PySilCam -Eventual optional dependencies (e.g. for classification)
4) Can be imported by pysilcam or other hardware-specific tools
5) Work on a single-image basis (...primarily, with options for multiprocess to be considered later)
6) No use of settings/config files within the core code - pass arguments directly. Eventual use of settings/config files should be handled by high-level wrappers that provide arguments to functions.
7) Github workflows
8) Tests

Normal functions within PyOpia should:

1) take inputs
2) return new outputs
3) don't modify state of input
4) minimum possible disc IO during processing

## Contributions

We welcome additions and improvements to the code! We request that you follow a few guidelines. These are in place to make sure the code improves over time.

1. All code changes must be submitted as pull requests, either from a branch or a fork.
2. Good documentation of the code is needed for PyOpia to succeed and so please include up-to-date docstrings as you make changes, so that the auto-build on readthedocs is complete and useful for users. (A version of the new docs will complie when you make a pull request and a link to this can be found in the pull request checks)
3. All pull requests are required to pass all tests before merging. Please do not disable or remove tests just to make your branch pass the pull request.
4. All pull requests must be reviewed by a person. The benefits from code review are plenty, but we like to emphasise that code reviews help spreading the awarenes of code changes. Please note that code reviews should be a pleasant experience, so be plesant, polite and remember that there is a human being with good intentions on the other side of the screen.
5. All contributions are linted with flake8. We recommend that you run flake8 on your code while developing to fix any issues as you go. We recommend using autopep8 to autoformat your Python code (but please check the code behaviour is not affected by autoformatting before pushing). This makes flake8 happy, and makes it easier for us all to maintain a consistent and readable code base.

## Docstrings

Use the NumPy style in docstrings. See style guide [here](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes)

# Installing

## For users

Users are expected to be familiar with Python. Please refer to the recommended installation instructions provided on the documentation pages, [here](https://pyopia.readthedocs.io/en/latest/intro.html#installing)

## For developers from source


Install (uv)[https://docs.astral.sh/uv/getting-started/installation/]

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
uv sync --all-extras
```

3. (optional) Run local tests:

```bash
uv run pytest
```

#### Version numbering

The version number of PyOPIA is split into three sections: MAJOR.MINOR.PATCH

* MAJOR: Changes in high-level pipeline use and/or data output that are not backwards-compatible.
* MINOR: New features that are backwards-compatible.
* PATCH: Backwards-compatible bug fixes or enhancements to existing functionality

## Build docs locally

```
sphinx-apidoc -f -o docs/source docs/build --separate

sphinx-build -b html ./docs/ ./docs/build
```

----
# License

PyOpia is licensed under the BSD3 license. See LICENSE. All contributors should be recognised & aknowledged.
