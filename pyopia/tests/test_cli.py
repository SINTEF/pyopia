'''
Tests for the pyopia.cli command line interface, using typer's CliRunner.
'''

import os
import tempfile

import toml
from typer.testing import CliRunner

from pyopia.cli import app

runner = CliRunner()


def test_generate_config_silcam():
    with tempfile.TemporaryDirectory() as tempdir:
        result = _invoke_in(tempdir, [
            'generate-config', 'silcam', 'images/*.silc', 'model.keras', 'proc', 'test'
        ])

        assert result.exit_code == 0
        config_path = os.path.join(tempdir, 'silcam-config.toml')
        assert os.path.isfile(config_path)

        toml_settings = toml.load(config_path)
        assert toml_settings['general']['raw_files'] == 'images/*.silc'
        assert toml_settings['steps']['classifier']['model_path'] == 'model.keras'


def test_generate_config_holo():
    with tempfile.TemporaryDirectory() as tempdir:
        result = _invoke_in(tempdir, [
            'generate-config', 'holo', 'images/*.pgm', 'model.keras', 'proc', 'test'
        ])

        assert result.exit_code == 0
        config_path = os.path.join(tempdir, 'holo-config.toml')
        assert os.path.isfile(config_path)

        toml_settings = toml.load(config_path)
        assert toml_settings['general']['raw_files'] == 'images/*.pgm'
        assert toml_settings['steps']['initial']['pipeline_class'] == 'pyopia.instrument.holo.Initial'


def test_generate_config_uvp():
    with tempfile.TemporaryDirectory() as tempdir:
        result = _invoke_in(tempdir, [
            'generate-config', 'uvp', 'images/*.png', 'model.keras', 'proc', 'test'
        ])

        assert result.exit_code == 0
        config_path = os.path.join(tempdir, 'uvp-config.toml')
        assert os.path.isfile(config_path)

        toml_settings = toml.load(config_path)
        assert toml_settings['general']['raw_files'] == 'images/*.png'


def test_modify_config():
    with tempfile.TemporaryDirectory() as tempdir:
        generate_result = _invoke_in(tempdir, [
            'generate-config', 'silcam', 'images/*.silc', 'model.keras', 'proc', 'test'
        ])
        assert generate_result.exit_code == 0

        modify_result = _invoke_in(tempdir, [
            'modify-config', 'silcam-config.toml', 'modified-config.toml',
            '--raw-files', 'other_images/*.silc',
            '--pixel-size', '12.5',
            '--step-name', 'segmentation',
            '--modify-arg', 'threshold',
            '--modify-value', '0.85',
        ])
        assert modify_result.exit_code == 0

        modified_path = os.path.join(tempdir, 'modified-config.toml')
        assert os.path.isfile(modified_path)

        toml_settings = toml.load(modified_path)
        assert toml_settings['general']['raw_files'] == 'other_images/*.silc'
        assert toml_settings['general']['pixel_size'] == 12.5
        assert toml_settings['steps']['segmentation']['threshold'] == 0.85


def _invoke_in(tempdir, args):
    '''Invoke the CLI app with the working directory set to tempdir, restoring
    the original working directory afterwards.
    '''
    original_cwd = os.getcwd()
    os.chdir(tempdir)
    try:
        return runner.invoke(app, args)
    finally:
        os.chdir(original_cwd)
