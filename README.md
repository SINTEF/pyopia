PyOpia
===============================

A Python Ocean Particle Image Analysis toolbox

Development requirements for PyOpia:
------------
-Allow nonfamiliar users to install and use PyOpia, and to contribute & commit code changes
-Not hardware specific
-Smaller dependency list than PySilCam -Eventual optional dependencies (e.g. for classification)
-Can be imported by pysilcam or other hardware-specific tools
-Work on a single-image basis (...primarily, with options for multiprocess to be considered later)
-No use of settings/config files - pass arguments directly. Eventual use of settings/config files should be handled by other repos that import PyOpia
-Github workflows
-Tests

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

License
-------

PyOpia is licensed under the BSD3 license. See LICENSE. All contributors should be recognised & aknowledged.