'''
A high level test for the basic processing pipeline.

'''

from glob import glob
import tempfile
import os

import pyopia.exampledata as testdata
import pyopia.io
import pyopia.classify
from pyopia.pipeline import Pipeline
import pyopia.process
import pyopia.statistics
import pyopia.background  # noqa: F401
import xarray
import pandas as pd
import skimage.io
import numpy as np

# location of the training data
database_path = '/Users/emlynd/Downloads/pysilcam-testdata/unittest-data/silcam_classification_database'
model_path = '/Users/emlynd/Downloads/pysilcam-testdata/keras_model/keras_model.h5'

def test_classify():
    '''
    Basic check of classification prediction against the training database.
    Therefore, if correct positive matches are not high percentages, then something is wrong with the prediction.

    @todo include more advanced testing of the classification feks. assert values in a confusion matrix.
    '''

    # Load the trained tensorflow model and class names
    cl = pyopia.classify.Classify(model_path)
    class_labels = cl.class_labels
    #model, class_labels = pyopia.classify.Classify(model_path)

    # class_labels should match the training data
    classes = glob(os.path.join(database_path, '*'))

    # @todo write a quick check that classes and class_labels agree before doing the proper test.

    def correct_positives(category):
        '''
        calculate the percentage positive matches for a given category
        '''

        # list the files in this category of the training data
        files = glob(os.path.join(database_path, category, '*.tiff'))

        assert len(files) > 50, 'less then 50 files in test data.'

        # start a counter of incorrectly classified images
        failed = 0
        time_limit = len(files) * 0.01
        t1 = pd.Timestamp.now()

        # loop through the database images
        for file in files:
            img = skimage.io.imread(file)  # load ROI
            prediction = cl.proc_predict(img)  # run prediction from silcam_classify

            ind = np.argmax(prediction)  # find the highest score

            # check if the highest score matches the correct category
            if not class_labels[ind] == category:
                # if not, the add to the failure count
                failed += 1

        # turn failed count into a success percent
        success = 100 - (failed / len(files)) * 100

        t2 = pd.Timestamp.now()
        td = t2 - t1
        assert td < pd.to_timedelta(time_limit, 's'), 'Processing time too long.'

        return success

    # loop through each category and calculate the success percentage
    for cat in classes:
        name = os.path.split(cat)[-1]
        success = correct_positives(name)
        print(name, success)
        assert success > 96, (name + ' was poorly classified at only ' + str(success) + 'percent.')


if __name__ == "__main__":
    test_classify()
