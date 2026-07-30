'''
A high level test for the basic processing pipeline.

'''

from glob import glob
import os
import numpy as np
import pytest
import skimage.io

import pyopia.io
import pyopia.classify
from pyopia.pipeline import FilesToProcess, Pipeline
import pyopia.process
import pyopia.statistics
import pyopia.background  # noqa: F401
import xarray


@pytest.mark.slow
def test_holo_pipeline(tmp_path, holo_example_files):
    '''
    Runs a holo pipeline on a single image with a pre-created background file.
    This test is primarily to detect errors when running the pipeline.

    Asserts that the number of particles counted after analysis is as-expected for the settings used in the test
    (although based on a course step-size, for speed purposes)

    Note: This does not properly test the background creation, and loads a pre-created background
    '''
    import pyopia.instrument.holo  # noqa: F401
    tempdir_proc = tmp_path / 'proc'
    tempdir_proc.mkdir()

    holo_filename, holo_background_filename = holo_example_files
    datafile_prefix = os.path.join(tempdir_proc, 'test')

    # define the configuration to use in the processing pipeline - given as a dictionary - with some values defined above
    pipeline_config = {
        'general': {
            'raw_files': os.path.join(os.path.dirname(holo_filename), '*.pgm'),
            'pixel_size': 4.4  # pixel size in um
        },
        'steps': {
            'initial': {
                'pipeline_class': 'pyopia.instrument.holo.Initial',
                'wavelength': 658,  # laser wavelength in nm
                'n': 1.33,  # index of refraction of sample volume medium (1.33 for water)
                'offset': 27,  # offset to start of sample volume in mm
                'minZ': 0,  # minimum reconstruction distance within sample volume in mm
                'maxZ': 50,  # maximum reconstruction distance within sample volume in mm
                'stepZ': 0.5  # step size in mm
            },
            'load': {
                'pipeline_class': 'pyopia.instrument.holo.Load'
            },
            'correctbackground': {
                'pipeline_class': 'pyopia.background.CorrectBackgroundAccurate',
                'bgshift_function': 'accurate',
                'average_window': 1
            },
            'reconstruct': {
                'pipeline_class': 'pyopia.instrument.holo.Reconstruct',
                'stack_clean': 0.02,
                'forward_filter_option': 2,
                'inverse_output_option': 0
            },
            'focus': {
                'pipeline_class': 'pyopia.instrument.holo.Focus',
                'stacksummary_function': 'max_map',
                'threshold': 0.97,
                'focus_function': 'find_focus_sobel',
                'increase_depth_of_field': False,
                'merge_adjacent_particles': 2
            },
            'segmentation': {
                'pipeline_class': 'pyopia.process.Segment',
                'threshold': 0.97,
                'segment_source': 'im_focussed'
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats',
                'export_outputpath': str(tempdir_proc),
                'propnames': ['major_axis_length', 'minor_axis_length', 'equivalent_diameter',
                              'feret_diameter_max', 'equivalent_diameter_area'],
                'roi_source': 'im_focussed'
            },
            'mergeholostats': {
                'pipeline_class': 'pyopia.instrument.holo.MergeStats',
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsToDisc',
                'output_datafile': datafile_prefix
            }
        }
    }

    processing_pipeline = Pipeline(pipeline_config)

    # Manually initialize the background from a pre-computed and stored image
    background_img = skimage.io.imread(holo_background_filename)
    processing_pipeline.data['bgstack'] = [background_img]
    processing_pipeline.data['imbg'] = np.mean(processing_pipeline.data['bgstack'], axis=0)

    print('Run processing on: ', holo_filename)
    processing_pipeline.run(holo_filename)
    with xarray.open_dataset(datafile_prefix + '-STATS.nc') as stats:
        stats.load()

    print('stats header: ', stats.data_vars)
    print('Total number of particles: ', len(stats.major_axis_length))
    assert len(stats.major_axis_length) == 40, ('Number of particles expected in this test is 56 for main' +
                                                ' (or 40 for dev-1.2.)' +
                                                ' This test counted ' + str(len(stats.major_axis_length)) +
                                                ' Something has altered the number of particles detected')


@pytest.mark.slow
def test_silcam_pipeline(tmp_path, silcam_example_image_dir):
    '''
    Asserts that the number of images counted in the processed hdf5 stats is the same as the
    number of images that should have been downloaded for the test.

    This test is primarily to detect errors when running the pipeline.
    '''
    import pyopia.instrument.silcam
    tempdir_proc = tmp_path / 'proc'
    tempdir_proc.mkdir()

    files = glob(os.path.join(silcam_example_image_dir, '*.silc'))
    print('file list available for test:')
    print(files)

    datafile_prefix = os.path.join(tempdir_proc, 'test')

    pipeline_config = {
        'general': {
            'raw_files': files,
            'pixel_size': 28  # pixel size in um
        },
        'steps': {
            'load': {
                'pipeline_class': 'pyopia.instrument.silcam.SilCamLoad'
            },
            'imageprep': {
                'pipeline_class': 'pyopia.instrument.silcam.ImagePrep',
                'image_level': 'imraw'
            },
            'segmentation': {
                'pipeline_class': 'pyopia.process.Segment',
                'threshold': 0.85,
                'segment_source': 'im_minimum'
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats',
                'roi_source': 'im_minimum'
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsToDisc',
                'output_datafile': datafile_prefix
            }
        }
    }

    processing_pipeline = Pipeline(pipeline_config)

    for filename in files[:2]:
        stats = processing_pipeline.run(filename)

    with xarray.open_dataset(datafile_prefix + '-STATS.nc') as stats:
        stats.load()

    print('stats header: ', stats.data_vars)
    print('Total number of particles: ', len(stats.major_axis_length))
    num_images = pyopia.statistics.count_images_in_stats(stats)
    print('Number of raw images: ', num_images)
    assert num_images == 1, ('Number of images expected is 1.' +
                             'This test counted' + str(num_images))
    assert len(stats.major_axis_length) == 870, ('Number of particles expected in this test is 870.' +
                                                 'This test counted ' + str(len(stats.major_axis_length)) +
                                                 ' Something has altered the number of particles detected')


def test_calculate_image_stats_uses_configured_path_length():
    '''Verifies CalculateImageStats uses the path_length configured in general settings,
    rather than silently falling back to the default of 40mm.

    Regression test: data['settings']['general'] is a plain dict, so a previous
    implementation using getattr(general, 'path_length', 40) always returned the
    default, silently ignoring any configured path_length.
    '''
    import pandas as pd
    from pyopia.statistics import nc_vc_from_stats

    pixel_size = 28.0
    path_length = 123.0  # deliberately not the default of 40, to catch silent fallback
    imy, imx = 100, 200

    timestamp = pd.Timestamp('2026-01-01T00:00:00')
    stats = pd.DataFrame({
        'major_axis_length': [10.0],
        'minor_axis_length': [8.0],
        'equivalent_diameter': [9.0],
        'saturation': [1.0],
        'export_name': ['D20260101T000000.000000-PN0'],
        'timestamp': [timestamp],
    })

    data = {
        'settings': {'general': {'pixel_size': pixel_size, 'path_length': path_length}},
        'imraw': np.zeros((imy, imx, 3), dtype=np.uint8),
        'stats': stats,
        'timestamp': timestamp,
    }

    step = pyopia.process.CalculateImageStats()
    data = step(data)

    expected_nc, expected_vc, expected_sample_volume, expected_junge = nc_vc_from_stats(
        stats, pixel_size, path_length, imx=imx, imy=imy)

    result = data['image_stats'].loc[data['timestamp']]
    np.testing.assert_allclose(result['vc'], expected_vc)
    np.testing.assert_allclose(result['sample_volume'], expected_sample_volume)

    # Sanity check that this isn't coincidentally matching the old (broken) default of 40mm
    wrong_nc, wrong_vc, wrong_sample_volume, wrong_junge = nc_vc_from_stats(
        stats, pixel_size, 40, imx=imx, imy=imy)
    assert not np.isclose(result['vc'], wrong_vc)


def test_per_class_concentration(tmp_path):
    '''Verifies PerClassConcentration writes timestamp-indexed per-class
    number concentrations (numbers/L) to CSV across multiple images.
    '''
    import pandas as pd
    from pyopia.statistics import PerClassConcentration, get_sample_volume

    pixel_size = 28.0
    path_length = 40.0
    imy, imx = 2048, 2448
    sample_volume = get_sample_volume(pixel_size, path_length, imx=imx, imy=imy)

    output_csv = tmp_path / 'sub' / 'per_class.csv'

    step = PerClassConcentration(
        output_csv=str(output_csv),
        probability_threshold=0.5,
        overwrite=True,
    )

    # Image 1: 2 oil, 1 bubble, 1 below-threshold (unclassified)
    stats_1 = pd.DataFrame({
        'equivalent_diameter': [10.0, 12.0, 14.0, 9.0],
        'probability_oil': [0.9, 0.8, 0.1, 0.4],
        'probability_bubble': [0.05, 0.15, 0.85, 0.35],
        'probability_other': [0.05, 0.05, 0.05, 0.25],
    })
    ts_1 = pd.Timestamp('2026-05-07T12:00:00')

    # Image 2: empty (placeholder NaN row, like pyopia.process.extract_particles)
    stats_2 = pd.DataFrame({
        'equivalent_diameter': [np.nan],
        'probability_oil': [np.nan],
        'probability_bubble': [np.nan],
        'probability_other': [np.nan],
    })
    ts_2 = pd.Timestamp('2026-05-07T12:00:01')

    data = {
        'settings': {'general': {'pixel_size': pixel_size, 'path_length': path_length}},
        'imraw': np.zeros((imy, imx, 3), dtype=np.uint8),
    }

    data['stats'] = stats_1
    data['timestamp'] = ts_1
    step(data)

    data['stats'] = stats_2
    data['timestamp'] = ts_2
    step(data)

    result = pd.read_csv(output_csv, index_col='timestamp', parse_dates=True)

    # Expected concentrations (counts / sample_volume in numbers/L)
    assert result.shape[0] == 2
    assert list(result.index) == [ts_1, ts_2]
    np.testing.assert_allclose(result.loc[ts_1, 'oil'], 2.0 / sample_volume)
    np.testing.assert_allclose(result.loc[ts_1, 'bubble'], 1.0 / sample_volume)
    np.testing.assert_allclose(result.loc[ts_1, 'other'], 0.0)
    np.testing.assert_allclose(result.loc[ts_1, 'unclassified'], 1.0 / sample_volume)
    np.testing.assert_allclose(result.loc[ts_1, 'total'], 4.0 / sample_volume)
    np.testing.assert_allclose(result.loc[ts_1, 'sample_volume_L'], sample_volume)

    # Empty image: zero concentrations across the board
    for col in ['oil', 'bubble', 'other', 'unclassified', 'total']:
        assert result.loc[ts_2, col] == 0.0
    np.testing.assert_allclose(result.loc[ts_2, 'sample_volume_L'], sample_volume)


def test_files_to_process_raises_clear_error_for_no_matching_files(tmp_path):
    '''Regression test for #279: an empty/non-matching raw_files pattern used to surface
    as a misleading "Number of chunks exceeds..." RuntimeError instead of a clear
    "no files found" error.
    '''
    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    glob_pattern = str(empty_dir / '*.silc')

    raw_files = FilesToProcess(glob_pattern)

    with pytest.raises(RuntimeError, match='No raw files found'):
        raw_files.prepare_chunking(num_chunks=1, average_window=0, bgshift_function='pass')
