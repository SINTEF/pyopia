'''
Tests for the pyopia.cli command line interface, using typer's CliRunner.

Commands that need real particle statistics/ROI images to operate on (process,
merge-mfdata, convert-raw-images, make-montage, export-to-ecotaxa) share a single
real end-to-end pipeline run against the downloaded SilCam example image, via the
session-scoped `silcam_cli_project` / `silcam_cli_merged_stats` fixtures defined in
conftest.py. This mirrors the real-download, real-processing approach already used
in test_pipeline.py and test_classify.py, rather than mocking the pipeline itself.
'''

import json
import pathlib
import zipfile

import pytest
import toml
import xarray

import pyopia
import pyopia.cli
from pyopia.cli import app

from pyopia.tests.conftest import invoke_in, runner


def test_generate_config_silcam(tmp_path):
    result = invoke_in(tmp_path, [
        'generate-config', 'silcam', 'images/*.silc', 'model.keras', 'proc', 'test'
    ])

    assert result.exit_code == 0
    config_path = tmp_path / 'silcam-config.toml'
    assert config_path.is_file()

    toml_settings = toml.load(config_path)
    assert toml_settings['general']['raw_files'] == 'images/*.silc'
    assert toml_settings['steps']['classifier']['model_path'] == 'model.keras'


def test_generate_config_holo(tmp_path):
    result = invoke_in(tmp_path, [
        'generate-config', 'holo', 'images/*.pgm', 'model.keras', 'proc', 'test'
    ])

    assert result.exit_code == 0
    config_path = tmp_path / 'holo-config.toml'
    assert config_path.is_file()

    toml_settings = toml.load(config_path)
    assert toml_settings['general']['raw_files'] == 'images/*.pgm'
    assert toml_settings['steps']['initial']['pipeline_class'] == 'pyopia.instrument.holo.Initial'


def test_generate_config_uvp(tmp_path):
    result = invoke_in(tmp_path, [
        'generate-config', 'uvp', 'images/*.png', 'model.keras', 'proc', 'test'
    ])

    assert result.exit_code == 0
    config_path = tmp_path / 'uvp-config.toml'
    assert config_path.is_file()

    toml_settings = toml.load(config_path)
    assert toml_settings['general']['raw_files'] == 'images/*.png'


def test_modify_config(tmp_path):
    generate_result = invoke_in(tmp_path, [
        'generate-config', 'silcam', 'images/*.silc', 'model.keras', 'proc', 'test'
    ])
    assert generate_result.exit_code == 0

    modify_result = invoke_in(tmp_path, [
        'modify-config', 'silcam-config.toml', 'modified-config.toml',
        '--raw-files', 'other_images/*.silc',
        '--pixel-size', '12.5',
        '--step-name', 'segmentation',
        '--modify-arg', 'threshold',
        '--modify-value', '0.85',
    ])
    assert modify_result.exit_code == 0

    modified_path = tmp_path / 'modified-config.toml'
    assert modified_path.is_file()

    toml_settings = toml.load(modified_path)
    assert toml_settings['general']['raw_files'] == 'other_images/*.silc'
    assert toml_settings['general']['pixel_size'] == 12.5
    assert toml_settings['steps']['segmentation']['threshold'] == 0.85


def test_docs_launches_readthedocs_url(monkeypatch):
    launched = {}
    monkeypatch.setattr(pyopia.cli.typer, 'launch', lambda url: launched.setdefault('url', url))

    result = runner.invoke(app, ['docs'])

    assert result.exit_code == 0
    assert launched['url'] == 'https://pyopia.readthedocs.io'


def test_version_flag_prints_package_version():
    result = runner.invoke(app, ['--version'])

    assert result.exit_code == 0
    assert f'PyOPIA version: {pyopia.__version__}' in result.output


@pytest.mark.slow
def test_init_project_creates_expected_structure(tmp_path):
    result = invoke_in(tmp_path, ['init-project', 'myproj'])

    assert result.exit_code == 0, result.output

    proj_folder = tmp_path / 'myproj'
    assert (proj_folder / 'images').is_dir()
    assert (proj_folder / 'auxillarydata' / 'auxillary_data.csv').is_file()
    assert (proj_folder / 'README').is_file()
    assert (proj_folder / 'metadata.json').is_file()
    assert (proj_folder / 'config.toml').is_file()

    config = toml.load(proj_folder / 'config.toml')
    assert config['general']['raw_files'] == 'images/*.silc'
    assert config['steps']['output']['project_metadata_file'] == 'metadata.json'
    assert config['steps']['output']['auxillary_data_file'] == 'auxillarydata/auxillary_data.csv'

    output_datafile = config['steps']['output']['output_datafile']
    assert (proj_folder / pathlib.Path(output_datafile).parent).is_dir()

    metadata = json.loads((proj_folder / 'metadata.json').read_text())
    assert metadata['project_name'] == 'myproj'
    assert metadata['instrument'] == 'silcam'


def test_init_project_refuses_to_overwrite_existing_folder(tmp_path):
    existing_project = tmp_path / 'myproj'
    existing_project.mkdir()

    result = invoke_in(tmp_path, ['init-project', 'myproj'])

    assert result.exit_code == 0
    assert 'ERROR' in result.output
    assert not (existing_project / 'config.toml').exists()


def test_check_chunks_rejects_less_than_one_chunk():
    with pytest.raises(RuntimeError, match='at least 1 chunk'):
        pyopia.cli.check_chunks(0, {'steps': {'output': {}}})


def test_check_chunks_rejects_multiple_chunks_when_append_enabled():
    with pytest.raises(RuntimeError, match='append = false'):
        pyopia.cli.check_chunks(2, {'steps': {'output': {'append': True}}})


def test_check_chunks_allows_multiple_chunks_when_append_disabled():
    pyopia.cli.check_chunks(2, {'steps': {'output': {'append': False}}})


def test_process_requires_an_output_step(tmp_path):
    config_filename = tmp_path / 'config.toml'
    with open(config_filename, 'w') as fh:
        toml.dump({'general': {'raw_files': str(tmp_path / '*.silc')}, 'steps': {}}, fh)

    result = invoke_in(tmp_path, ['process', str(config_filename)])

    assert result.exit_code != 0
    assert 'output' in str(result.exception)


@pytest.mark.slow
def test_process_produces_real_particle_stats_and_roi_export(silcam_cli_project):
    for stats_file in silcam_cli_project['stats_files']:
        with xarray.open_dataset(stats_file) as stats:
            stats.load()

        assert len(stats.major_axis_length) == 870
        assert any(name.startswith('probability_') for name in stats.data_vars)

    assert (silcam_cli_project['project_dir'] / 'filelist.txt').is_file()

    roi_files = list(silcam_cli_project['roi_folder'].glob('*.h5'))
    assert len(roi_files) == len(silcam_cli_project['stats_files']) == 2


def test_process_realtime_requires_an_output_step(tmp_path):
    config_filename = tmp_path / 'config.toml'
    with open(config_filename, 'w') as fh:
        toml.dump({'general': {}, 'steps': {}}, fh)

    result = invoke_in(tmp_path, ['process-realtime', str(config_filename)])

    assert result.exit_code != 0
    assert 'output' in str(result.exception)


def test_process_realtime_requires_output_datafile_setting(tmp_path):
    config_filename = tmp_path / 'config.toml'
    with open(config_filename, 'w') as fh:
        toml.dump({
            'general': {},
            'steps': {'output': {'pipeline_class': 'pyopia.io.StatsToDisc'}},
        }, fh)

    result = invoke_in(tmp_path, ['process-realtime', str(config_filename)])

    assert result.exit_code != 0
    assert 'output_datafile' in str(result.exception)


def test_process_realtime_prepares_output_folder_and_calls_run_realtime(tmp_path, monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        pyopia.cli.pyopia.realtime, 'run_realtime',
        lambda pipeline_config, watch_folder=None: recorded.update(
            pipeline_config=pipeline_config, watch_folder=watch_folder
        )
    )

    output_datafile = str(tmp_path / 'proc' / 'test')
    config_filename = tmp_path / 'config.toml'
    with open(config_filename, 'w') as fh:
        toml.dump({
            'general': {'raw_files': str(tmp_path / 'images' / '*.silc')},
            'steps': {'output': {'pipeline_class': 'pyopia.io.StatsToDisc', 'output_datafile': output_datafile}},
        }, fh)

    result = invoke_in(tmp_path, [
        'process-realtime', str(config_filename), '--watch-folder', str(tmp_path / 'images')
    ])

    assert result.exit_code == 0, result.output
    assert (tmp_path / 'proc').is_dir()
    assert recorded['watch_folder'] == str(tmp_path / 'images')
    assert recorded['pipeline_config']['steps']['output']['output_datafile'] == output_datafile


@pytest.mark.slow
def test_merge_mfdata_combines_per_image_stats(silcam_cli_project, silcam_cli_merged_stats):
    with xarray.open_dataset(silcam_cli_merged_stats) as merged:
        merged.load()

    per_image_particle_counts = []
    for stats_file in silcam_cli_project['stats_files']:
        with xarray.open_dataset(stats_file) as original:
            original.load()
            per_image_particle_counts.append(len(original.major_axis_length))

    assert len(merged.major_axis_length) == sum(per_image_particle_counts) == 1740


@pytest.mark.slow
def test_convert_raw_images_creates_png(silcam_cli_project, tmp_path):
    result = invoke_in(tmp_path, ['convert-raw-images', str(silcam_cli_project['config_filename'])])

    assert result.exit_code == 0, result.output

    converted = list((tmp_path / 'images_converted').glob('*.png'))
    assert len(converted) == 2


@pytest.mark.slow
def test_make_montage_creates_real_montage_image(silcam_cli_merged_stats, tmp_path):
    montage_path = tmp_path / 'montage.png'

    result = invoke_in(tmp_path, [
        'make-montage', str(silcam_cli_merged_stats), '--output-filename', str(montage_path)
    ])

    assert result.exit_code == 0, result.output
    assert montage_path.is_file()
    assert montage_path.stat().st_size > 0


@pytest.mark.slow
def test_export_to_ecotaxa_creates_bundle_zip(silcam_cli_merged_stats, tmp_path):
    export_path = tmp_path / 'ecotaxa_export.zip'

    result = invoke_in(tmp_path, [
        'export-to-ecotaxa', str(silcam_cli_merged_stats), str(export_path)
    ])

    assert result.exit_code == 0, result.output
    assert export_path.is_file()

    with zipfile.ZipFile(export_path) as bundle:
        names = bundle.namelist()
        assert 'ecotaxa_particle_statistics.tsv' in names
        assert sum(name.endswith('.png') for name in names) == 1740
