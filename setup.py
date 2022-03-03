# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand
from setuptools.command.install import install
from setuptools.command.develop import develop
import distutils.cmd


# modifications to develop and install, based on post here:
# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        Documentation.run(self)


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
        params["args"] += ["--junitxml", "test-report/output.xml"]
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content


setup(
    name='PyOpia',
    description='A Python Ocean Particle Image Analysis toolbox',
    long_description=read('README.md'),
    author='Emlyn Davies',
    author_email='emlyn.davies@sintef.no',
    zip_safe=False,
    keywords='Ocean Particle Image Analysis',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    packages=['pyopia'],
    cmdclass={'test': PyTest,
              'test_noskip': PyTestNoSkip,
              'develop': PostDevelopCommand,
              'install': PostInstallCommand}
)
