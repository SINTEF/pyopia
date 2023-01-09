'''
A high level test for the basic processing pipeline.

Asserts that the number of images counted in the processed hdf5 stats is the same as the
number of images that should have been downloaded for the test
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
import pyopia.process
import pyopia.statistics


def test_pipeline():
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
                 'imageprep': ImagePrep(),
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
                                 'This test sounted' + str(num_images))


if __name__ == "__main__":
    test_pipeline()
