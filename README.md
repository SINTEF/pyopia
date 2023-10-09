PyOpia
===============================

A Python Ocean Particle Image Analysis toolbox

# Documentation:

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pyopia.readthedocs.io) [![Documentation](https://readthedocs.org/projects/pyopia/badge/?version=latest)](https://pyopia.readthedocs.io/en/latest/?badge=latest)

[pyopia.readthedocs.io](https://pyopia.readthedocs.io)
# Current status:

- Under development. See/regester issues, [here](https://github.com/SINTEF/pyopia/issues)

----

# Installing for users

Users are expected to be familiar with Python, and have [Python](https://github.com/conda-forge/miniforge/#download), [pip](https://pypi.org/project/pip/). You can then install Pyopia like this:

```
pip install pyopia
```

We would usually recommend installing within a virtual python environment, which you can read more about [here](https://jni.github.io/using-python-for-science/intro-to-environments.html).

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

## Installing from source for developers

Install [Python](https://github.com/conda-forge/miniforge/#download).

A prompt such as is provided by the [miniforge installation](https://github.com/conda-forge/miniforge/#download) may be used for the following:

1. Navigate to the folder where you want to install pyopia using the 'cd' command.

If you use git:
Download repository from github, and move into the new directory:

```bash
git clone https://github.com/SINTEF/pyopia.git
cd pyopia
```

If you do not want to use git, you can manually download the repository from github. To do so, click on the green "<> CODE button" (top right-hand side) and select "Download zip". Unzip the files into your target folder, and move into the new directory (note the foldername may contain the branch name, such as 'main').

```bash
cd pyopia-main
```

For the next steps, you need to be located in the directory that contains the file 'environment.yml'.

1. (optional, but recommended) Create a virtual environment using the environment.yml. This will create an environment called pyopia, but with no dependencies installed. Dependencies are managed by poetry (in step 2):

```bash
conda env create -f environment.yml
```
and activate the environment:

```bash
conda activate pyopia
```

2. Install dependencies using poetry:

```bash
poetry install
```

3. (optional) Run local tests:

```bash
poetry run pytest
```

## Build docs locally

```
sphinx-apidoc -f -o docs/source docs/build --separate

sphinx-build -b html ./docs/ ./docs/build
```

----
# License

PyOpia is licensed under the BSD3 license. See LICENSE. All contributors should be recognised & aknowledged.
