'''
A high level test for the basic processing pipeline.

'''

from glob import glob
import tempfile
import os

import pyopia.tests.testdata as testdata
import pyopia.io
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
                'raw_files': tempdir,
                'pixel_size': 4.4  # pixel size in um
            },
            'steps': {
                # start of steps run once on pipeline initialisation
                # initial step to setup hologram reconstruction kernel - arguments are hologram reconstruction settings
                'initial': {
                    'pipeline_class': 'pyopia.instrument.holo.Initial',
                    'wavelength': 658,  # laser wavelength in nm
                    'n': 1.33,  # index of refraction of sample volume medium (1.33 for water)
                    'offset': 27,  # offset to start of sample volume in mm
                    'minZ': 0,  # minimum reconstruction distance within sample volume in mm
                    'maxZ': 50,  # maximum reconstruction distance within sample volume in mm
                    'stepZ': 0.5  # step size in mm
                },
                # sets up classifier model, runs once on pipeline initialisation - argument is the path to the classification model to use from Step 03
                'classifier': {
                    'pipeline_class': 'pyopia.classify.Classify',
                    'model_path': model_path
                },
                # creates initial background, runs once on pipeline initialisation - arguments are number of files to use for initial background and which instrument loading function to use
                'createbackground': {
                    'pipeline_class': 'pyopia.background.CreateBackground',
                    'average_window': 10,
                    'instrument_module': 'holo'
                },
                # start of steps applied to every image
                # load the image using instrument-specific loading function 
                'load': {
                    'pipeline_class': 'pyopia.instrument.holo.Load'
                },
                # apply background correction - argument is which method to use:
                # 'accurate' - recommended method for moving background
                # 'fast' - faster method for realtime applications
                # 'pass' - omit background correction
                'correctbackground': {
                    'pipeline_class': 'pyopia.background.CorrectBackgroundAccurate',
                    'bgshift_function': 'accurate'
                },
                # hologram reconstruction step - arguments are:
                # stack_clean - is how much stack cleaning (% dimmest pixels to remove) to apply - set to 0 to omit cleaning
                # forward_filter_option - switch to control filtering in frequency domain (0=none,1=DC only,2=zero ferquency/default)
                # inverse_output_option - switch to control optional scaling of output intensity (0=square/default,1=linear)
                'reconstruct': {
                    'pipeline_class': 'pyopia.instrument.holo.Reconstruct',
                    'stack_clean': 0.02,
                    'forward_filter_option': 2,
                    'inverse_output_option': 0
                },
                # focussing step - arguments are:
                # which summarisation method to use:
                # 'std_map' (default) - takes standard deviation of values through stack
                # 'max_map' - takes maximum intensity value through stack
                # threshold is global segmentation threshold to apply to stack summary
                # which focus function to use:
                # 'find_focis_imax' (default) - finds focus using plane of maximum intensity
                # 'find_focus_sobel' - finds focus using edge sharpness
                # focus options are:
                # increase_depth_of_field (bool, default False) - finds max of planes adjacent to optimum focus plane
                # merge_adjacent_particles (int, default 0) - merges adjacent particles within stack summary using this pixel radius
                'focus': {
                    'pipeline_class': 'pyopia.instrument.holo.Focus',
                    'stacksummary_function': 'max_map',
                    'threshold': 0.9,
                    'focus_function': 'find_focus_sobel',
                    'increase_depth_of_field': False,
                    'merge_adjacent_particles': 2
                },
                # segmentation of focussed particles - argument is threshold to apply (can be different to Focus step)
                'segmentation': {
                    'pipeline_class': 'pyopia.process.Segment',
                    'threshold': 0.9
                },
                # extraction of particle statistics - arguments are:
                # export_outputpath - is output folder for image-specific outputs for montage creation (can be omitted)
                # propnames - is list of skimage regionprops to export to stats (optional - must contain default values that can be appended to)
                'statextract': {
                    'pipeline_class': 'pyopia.process.CalculateStats',
                    'export_outputpath': tempdir_proc, 
                    'propnames': ['major_axis_length', 'minor_axis_length', 'equivalent_diameter', 
                                  'feret_diameter_max', 'equivalent_diameter_area']
                },
                # step to merge hologram-specific information (currently focus depth & original filename) into output statistics file
                'mergeholostats': {
                    'pipeline_class': 'pyopia.instrument.holo.MergeStats',
                },
                # write the output HDF5 statistics file
                'output': {
                    'pipeline_class': 'pyopia.io.StatsH5',
                    'output_datafile': datafile_prefix
                }
            }
        }

        processing_pipeline = Pipeline(pipeline_config)

        processing_pipeline.run(holo_filename)
        with xarray.open_dataset(datafile_prefix + '-STATS.nc') as stats:
            stats.load()

        print('stats header: ', stats.columns)
        print('Total number of particles: ', len(stats))
        assert len(stats) == 42, ('Number of particles expected in this test is 42.' +
                                  'This test counted' + str(len(stats)) +
                                  'Something has altered the number of particles detected')


def test_silcam_pipeline():
    '''
    Asserts that the number of images counted in the processed hdf5 stats is the same as the
    number of images that should have been downloaded for the test.

    This test is primarily to detect errors when running the pipeline.
    '''
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
                'raw_files': tempdir,
                'pixel_size': 28  # pixel size in um
            },
            'steps': {
                # sets up classifier model, runs once on pipeline initialisation - argument is the path to the classification model to use from Step 03
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

        print('stats header: ', stats.columns)
        print('Total number of particles: ', len(stats))
        num_images = pyopia.statistics.count_images_in_stats(stats)
        print('Number of raw images: ', num_images)
        assert num_images == 1, ('Number of images expected is 1.' +
                                 'This test counted' + str(num_images))
        assert len(stats) == 870, ('Number of particles expected in this test is 870.' +
                                   'This test counted' + str(len(stats)) +
                                   'Something has altered the number of particles detected')


if __name__ == "__main__":
    pass
