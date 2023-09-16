'''
A high level test for the basic processing pipeline.

'''

from glob import glob
import tempfile
import os

import pyopia.tests.testdata as testdata
import pyopia.io
import pyopia.classify
from pyopia.pipeline import Pipeline
import pyopia.process
import pyopia.statistics
import pyopia.background
import xarray


def test_holo_pipeline():
    '''
    Runs a holo pipeline on a single image with a pre-created background file.
    This test is primarily to detect errors when running the pipeline.

    Asserts that the number of particles counted after analysis is as-expected for the settings used in the test
    (although based on a course step-size, for speed purposes)

    Note: This does not properly test the background creation, and loads a pre-created background
    '''
    import pyopia.instrument.holo
    with tempfile.TemporaryDirectory() as tempdir:
        print('tmpdir created:', tempdir)
        os.makedirs(tempdir, exist_ok=True)
        tempdir_proc = os.path.join(tempdir, 'proc')
        os.makedirs(tempdir_proc, exist_ok=True)

        model_path = testdata.get_example_model(tempdir)
        print('model_path:', model_path)

        holo_filename, holo_background_filename = testdata.get_example_hologram_and_background(tempdir)
        datafile_prefix = os.path.join(tempdir_proc, 'test')

        # define the configuration to use in the processing pipeline - given as a dictionary - with some values defined above
        pipeline_config = {
            'general': {
                'raw_files': os.path.join(tempdir, '*.pgm'),
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
                'classifier': {
                    'pipeline_class': 'pyopia.classify.Classify',
                    'model_path': model_path
                },
                'createbackground': {
                    'pipeline_class': 'pyopia.background.CreateBackground',
                    'average_window': 10,
                    'instrument_module': 'holo'
                },
                'load': {
                    'pipeline_class': 'pyopia.instrument.holo.Load'
                },
                'correctbackground': {
                    'pipeline_class': 'pyopia.background.CorrectBackgroundAccurate',
                    'bgshift_function': 'accurate'
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
                    'threshold': 0.9,
                    'focus_function': 'find_focus_sobel',
                    'increase_depth_of_field': False,
                    'merge_adjacent_particles': 2
                },
                'segmentation': {
                    'pipeline_class': 'pyopia.process.Segment',
                    'threshold': 0.9
                },
                'statextract': {
                    'pipeline_class': 'pyopia.process.CalculateStats',
                    'export_outputpath': tempdir_proc,
                    'propnames': ['major_axis_length', 'minor_axis_length', 'equivalent_diameter',
                                  'feret_diameter_max', 'equivalent_diameter_area']
                },
                'mergeholostats': {
                    'pipeline_class': 'pyopia.instrument.holo.MergeStats',
                },
                'output': {
                    'pipeline_class': 'pyopia.io.StatsH5',
                    'output_datafile': datafile_prefix
                }
            }
        }

        processing_pipeline = Pipeline(pipeline_config)

        print('Run processing on: ', holo_filename)
        processing_pipeline.run(holo_filename)
        with xarray.open_dataset(datafile_prefix + '-STATS.nc') as stats:
            stats.load()

        print('stats header: ', stats.data_vars)
        print('Total number of particles: ', len(stats.major_axis_length))
        assert len(stats.major_axis_length) == 42, ('Number of particles expected in this test is 42.' +
                                                    'This test counted' + str(len(stats.major_axis_length)) +
                                                    'Something has altered the number of particles detected')


def test_silcam_pipeline():
    '''
    Asserts that the number of images counted in the processed hdf5 stats is the same as the
    number of images that should have been downloaded for the test.

    This test is primarily to detect errors when running the pipeline.
    '''
    import pyopia.instrument.silcam
    with tempfile.TemporaryDirectory() as tempdir:
        os.makedirs(tempdir, exist_ok=True)
        tempdir_proc = os.path.join(tempdir, 'proc')
        os.makedirs(tempdir_proc, exist_ok=True)

        model_path = testdata.get_example_model(tempdir)
        print('model_path:', model_path)

        filename = testdata.get_example_silc_image(tempdir)
        print('filename got:', filename)

        files = glob(os.path.join(tempdir, '*.silc'))
        print('file list available for test:')
        print(files)

        datafile_prefix = os.path.join(tempdir_proc, 'test')

        pipeline_config = {
            'general': {
                'raw_files': files,
                'pixel_size': 28  # pixel size in um
            },
            'steps': {
                'classifier': {
                    'pipeline_class': 'pyopia.classify.Classify',
                    'model_path': model_path
                },
                'load': {
                    'pipeline_class': 'pyopia.instrument.silcam.SilCamLoad'
                },
                'imageprep': {
                    'pipeline_class': 'pyopia.instrument.silcam.ImagePrep',
                    'image_level': 'imraw'
                },
                'segmentation': {
                    'pipeline_class': 'pyopia.process.Segment',
                    'threshold': 0.85
                },
                'statextract': {
                    'pipeline_class': 'pyopia.process.CalculateStats'
                },
                'output': {
                    'pipeline_class': 'pyopia.io.StatsH5',
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
                                                     'This test counted' + str(len(stats.major_axis_length)) +
                                                     'Something has altered the number of particles detected')


if __name__ == "__main__":
    pass