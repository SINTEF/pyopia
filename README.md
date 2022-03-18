PyOpia
===============================

A Python Ocean Particle Image Analysis toolbox

Current status:
------------
Moving code from PySilCam


Development requirements for PyOpia:
------------
1) Allow nonfamiliar users to install and use PyOpia, and to contribute & commit code changes
2) Not hardware specific
3) Smaller dependency list than PySilCam -Eventual optional dependencies (e.g. for classification)
4) Can be imported by pysilcam or other hardware-specific tools
5) Work on a single-image basis (...primarily, with options for multiprocess to be considered later)
6) No use of settings/config files - pass arguments directly. Eventual use of settings/config files should be handled by other repos that import PyOpia
7) Github workflows
8) Tests

Normal functions within PyOpia should:
------------
1) take inputs
2) return new outputs
3) don't modify state of input
4) minimum possible disc IO

Contributions
-------------

We welcome additions and improvements to the code! We request that you follow a few guidelines. These are in place to make sure the code improves over time.

1. All code changes must be submitted as pull requests, either from a branch or a fork.
2. All pull requests are required to pass all tests. Please do not disable or remove tests just to make your branch pass the pull request.
3. All pull requests must be reviewed by a person. The benefits from code review are plenty, but we like to emphasise that code reviews help spreading the awarenes of code changes. Please note that code reviews should be a pleasant experience, so be plesant, polite and remember that there is a human being with good intentions on the other side of the screen.
4. All contributions are linted with flake8. We recommend that you run flake8 on your code while developing to fix any issues as you go.
5. We recommend using autopep8 to autoformat your Python code (but please check the code behaviour is not affected by autoformatting before pushing). This makes flake8 happy, and makes it easier for us all to maintain a consistent and readable code base.

Installing
----------

Install [Python](https://github.com/conda-forge/miniforge/#download)

A prompt such as is provided by the [miniforge installation](https://github.com/conda-forge/miniforge/#download) may be used for the following:

Create a virtual environment using the environment.yml (will create an environment called pyopia)

```bash
conda env create -f environment.yml
```

to update, we recommend a forced re-install:

```bash
conda env create -f environment.yml --force
```

to activate:

```bash
conda activate pyopia
```

Test that it works with

```bash
python setup.py develop
```

License
-------

PyOpia is licensed under the BSD3 license. See LICENSE. All contributors should be recognised & aknowledged.
