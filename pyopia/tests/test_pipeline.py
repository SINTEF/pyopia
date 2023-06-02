'''
A high level test for the basic processing pipeline.

'''

import pandas as pd
from glob import glob
import tempfile
import os

from pyopia.classify import Classify
import pyopia.tests.testdata as testdata
import pyopia.io
from pyopia.pipeline import Pipeline
from pyopia.instrument.silcam import SilCamLoad, ImagePrep
import pyopia.instrument.holo as holo
import pyopia.process
import pyopia.statistics
import pyopia.background


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
        datafile_hdf = os.path.join(tempdir_proc, 'test')

        threshold = 0.9

        holo_initial_settings = {'pixel_size': 4.4,  # pixel size in um
                                 'wavelength': 658,  # laser wavelength in nm
                                 'n': 1.33,  # index of refraction of sample volume medium (1.33 for water)
                                 'offset': 27,  # offset to start of sample volume in mm
                                 'minZ': 22,  # minimum reconstruction distance in mm
                                 'maxZ': 60,  # maximum reconstruction distance in mm
                                 'stepZ': 5}  # step size in mm (use a very large step size for speed in running this test)

        steps = {'initial': holo.Initial(holo_filename, **holo_initial_settings),  # initialisation step to create reconstruction kernel
                 'classifier': Classify(model_path=model_path),
                 'create background': pyopia.background.CreateBackground([holo_background_filename],
                                                                         pyopia.instrument.holo.load_image),
                 'load': holo.Load(),
                 'correct background': pyopia.background.CorrectBackgroundAccurate(pyopia.background.shift_bgstack_accurate),
                 'reconstruct': holo.Reconstruct(stack_clean=0.02),
                 'focus': holo.Focus(pyopia.instrument.holo.std_map,
                                     threshold=threshold,
                                     focus_function=pyopia.instrument.holo.find_focus_sobel,
                                     increase_depth_of_field=True,
                                     merge_adjacent_particles=0),
                 'segmentation': pyopia.process.Segment(threshold=threshold),
                 'statextract': pyopia.process.CalculateStats(export_outputpath=tempdir_proc,
                                                              propnames=['major_axis_length', 'minor_axis_length', 'equivalent_diameter', 
                                                                         'feret_diameter_max', 'equivalent_diameter_area']),
                 'output': pyopia.io.StatsH5(datafile_hdf)
                 }

        processing_pipeline = Pipeline(steps)

        processing_pipeline.run(holo_filename)

        pyopia.io.show_h5_meta(datafile_hdf + '-STATS.h5')

        stats = pd.read_hdf(datafile_hdf + '-STATS.h5', 'ParticleStats/stats')
        print('stats header: ', stats.columns)
        print('Total number of particles: ', len(stats))
        assert len(stats) == 42, ('Number of particles expected in this test is 42.' +
                                  'This test counted' + len(stats) +
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

        datafile_hdf = os.path.join(tempdir_proc, 'test')

        threshold = 0.85

        steps = {'classifier': Classify(model_path=model_path),
                 'load': SilCamLoad(),
                 'imageprep': ImagePrep(image_level='imraw'),
                 'segmentation': pyopia.process.Segment(threshold=threshold),
                 'statextract': pyopia.process.CalculateStats(),
                 'output': pyopia.io.StatsH5(datafile_hdf)}

        processing_pipeline = Pipeline(steps, initial_steps=['classifier'])

        for filename in files[:2]:
            stats = processing_pipeline.run(filename)

        # display metadata in the h5
        pyopia.io.show_h5_meta(datafile_hdf + '-STATS.h5')

        # load the stats DataFrame from the h5 file
        stats = pd.read_hdf(datafile_hdf + '-STATS.h5', 'ParticleStats/stats')
        print('stats header: ', stats.columns)
        print('Total number of particles: ', len(stats))
        num_images = pyopia.statistics.count_images_in_stats(stats)
        print('Number of raw images: ', num_images)
        assert num_images == 1, ('Number of images expected is 1.' +
                                 'This test counted' + str(num_images))
        assert len(stats) == 870, ('Number of particles expected in this test is 870.' +
                                   'This test counted' + len(stats) +
                                   'Something has altered the number of particles detected')


if __name__ == "__main__":
    pass
