'''
Shared pytest fixtures for pyopia.cli integration tests.
'''

import os
import shutil

import pytest
import toml
from typer.testing import CliRunner

import pyopia.exampledata
from pyopia.cli import app

runner = CliRunner()


def invoke_in(directory, args):
    '''Invoke the CLI app with the working directory set to `directory`,
    restoring the original working directory afterwards.
    '''
    original_cwd = os.getcwd()
    os.chdir(directory)
    try:
        return runner.invoke(app, args)
    finally:
        os.chdir(original_cwd)


@pytest.fixture(scope='session')
def silcam_example_files(tmp_path_factory):
    '''Download the real SilCam example image and classifier model once per test session.

    pyopia.pipeline.FilesToProcess.prepare_chunking() refuses to process a raw file
    list with fewer than 2 entries (`num_chunks > len(files) // 2`), even for the
    default single-chunk case. Only one distinct real raw frame is available via
    pyopia.exampledata, so the downloaded image is duplicated under a second,
    differently-timestamped filename to satisfy that minimum. The image content
    processed by the pipeline is still the real downloaded photo, just used twice;
    no pipeline/classification logic is faked.
    '''
    download_dir = tmp_path_factory.mktemp('silcam_example_data')
    image_filename = pyopia.exampledata.get_example_silc_image(str(download_dir))
    model_path = pyopia.exampledata.get_example_model(str(download_dir))

    original_path = download_dir / image_filename
    duplicate_path = download_dir / 'D20181101T142732.838206.silc'
    shutil.copy(original_path, duplicate_path)

    return {
        'download_dir': download_dir,
        'model_path': model_path,
    }


@pytest.fixture(scope='session')
def silcam_cli_project(tmp_path_factory, silcam_example_files):
    '''Run the real `pyopia process` CLI command against the real downloaded SilCam
    example image, producing genuine per-image STATS.nc output and exported ROI images.

    Background correction is deliberately left out of this config: with only one real
    raw frame available, CorrectBackgroundAccurate would build its background from
    that same frame, which self-cancels the image and yields zero particles. This
    mirrors the working config used in test_pipeline.py::test_silcam_pipeline.

    Session-scoped so the real download and model inference only happens once, and is
    shared by every CLI test that needs real particle stats/ROIs to operate on
    (process, merge-mfdata, convert-raw-images, make-montage, export-to-ecotaxa).
    '''
    project_dir = tmp_path_factory.mktemp('silcam_cli_project')
    outfolder = project_dir / 'proc'
    roi_folder = outfolder / 'roi'

    config_filename = project_dir / 'silcam-config.toml'
    pipeline_config = {
        'general': {
            'raw_files': str(silcam_example_files['download_dir'] / '*.silc'),
            'pixel_size': 28,
        },
        'steps': {
            'classifier': {
                'pipeline_class': 'pyopia.classify.Classify',
                'model_path': str(silcam_example_files['model_path']),
            },
            'load': {
                'pipeline_class': 'pyopia.instrument.silcam.SilCamLoad',
            },
            'imageprep': {
                'pipeline_class': 'pyopia.instrument.silcam.ImagePrep',
                'image_level': 'imraw',
            },
            'segmentation': {
                'pipeline_class': 'pyopia.process.Segment',
                'threshold': 0.85,
                'segment_source': 'im_minimum',
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats',
                'export_outputpath': str(roi_folder),
                'roi_source': 'im_minimum',
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsToDisc',
                'output_datafile': str(outfolder / 'test'),
                'append': False,
            },
        },
    }
    with open(config_filename, 'w') as fh:
        toml.dump(pipeline_config, fh)

    result = invoke_in(project_dir, ['process', str(config_filename)])
    assert result.exit_code == 0, result.output

    stats_files = sorted(outfolder.glob('*Image-D*-STATS.nc'))
    assert len(stats_files) == 2, f'Expected two per-image STATS files, found: {stats_files}'

    return {
        'project_dir': project_dir,
        'config_filename': config_filename,
        'outfolder': outfolder,
        'roi_folder': roi_folder,
        'stats_files': stats_files,
    }


@pytest.fixture(scope='session')
def silcam_cli_merged_stats(silcam_cli_project):
    '''Merge the per-image STATS.nc file(s) from `silcam_cli_project` via the real
    `merge-mfdata` CLI command, producing a combined STATS.nc shared by the
    make-montage and export-to-ecotaxa tests.
    '''
    outfolder = silcam_cli_project['outfolder']
    result = invoke_in(silcam_cli_project['project_dir'], ['merge-mfdata', str(outfolder)])
    assert result.exit_code == 0, result.output

    merged_path = outfolder / 'test-STATS.nc'
    assert merged_path.is_file()
    return merged_path
