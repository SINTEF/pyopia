# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
from setuptools.command.develop import develop
import distutils.cmd


with open("requirements.txt") as f:
    requirements = f.readlines()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)


class PyTestNoSkip(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        # use pytest plugin to force error if a test is skipped
        self.test_args = ['--error-for-skips']
        self.test_suite = True

    def run_tests(self):
        import pytest
        params = {"args": self.test_args}
        params["args"] += ["--junitxml", "test-report/output.xml"]
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        params = {"args": self.test_args}
        params["args"] += ["-s", "--junitxml", "test-report/output.xml"]
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


class Documentation(distutils.cmd.Command):
    description = '''Build the documentation with Sphinx.
                   sphinx-apidoc is run for automatic generation of the sources.
                   sphinx-build then creates the html from these sources.'''
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        command = 'sphinx-apidoc -f -o docs/source pyopia/ --separate'
        os.system(command)
        command = 'sphinx-build -b html ./docs/source ./docs/build'
        os.system(command)
        if not os.environ.get('READTHEDOCS'):
            sys.exit()


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


if __name__ == "__main__":
    setup(install_requires=requirements,
          packages=['pyopia', 'pyopia.instrument'],
          cmdclass={'test': PyTest,
                    'test_noskip': PyTestNoSkip,
                    'develop': PostDevelopCommand,
                    'docbuild': Documentation}
          )
