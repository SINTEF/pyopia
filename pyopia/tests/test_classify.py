'''
A high level test for the basic processing pipeline.

'''

from glob import glob
import tempfile
import os
from tqdm import tqdm

import pyopia.exampledata as exampledata
import pyopia.io
import pyopia.classify
import pyopia.pipeline
import pyopia.process
import pyopia.statistics
import pyopia.background  # noqa: F401
import pandas as pd
import skimage.io
import numpy as np
import pyopia.instrument.silcam


ACCURACY = 39


def test_match_to_database():
    '''
    Basic check of classification prediction against the training database.
    Therefore, if correct positive matches are not high percentages, then something is wrong with the prediction.

    @todo include more advanced testing of the classification feks. assert values in a confusion matrix.
    '''

    # location of the training data
    with tempfile.TemporaryDirectory() as tempdir:
        # location of the training data
        database_path = os.path.join(tempdir, 'silcam_classification_database')

        exampledata.get_classifier_database_from_pysilcam_blob(database_path)
        os.makedirs(os.path.join(tempdir, 'model'), exist_ok=True)
        model_path = exampledata.get_example_model(os.path.join(tempdir, 'model'))

        # Load the trained tensorflow model and class names
        cl = pyopia.classify.Classify(model_path=model_path)
        class_labels = cl.class_labels

        # class_labels should match the training data
        classes = sorted(glob(os.path.join(database_path, '*')))

        # @todo write a quick check that classes and class_labels agree before doing the proper test.

        def correct_positives(category):
            '''
            calculate the percentage positive matches for a given category
            '''
            print('Checking', category)
            # list the files in this category of the training data
            files = glob(os.path.join(database_path, category, '*.tiff'))

            assert len(files) > 50, 'less then 50 files in test data.'

            # start a counter of incorrectly classified images
            failed = 0
            time_limit = len(files) * 0.01
            t1 = pd.Timestamp.now()

            # loop through the database images
            for file in tqdm(files):
                img = skimage.io.imread(file)  # load ROI
                img = np.float64(img) / 255
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
            assert success > ACCURACY, (name + ' was poorly classified at only ' + str(success) + 'percent.')


def test_pipeline_classification():
    '''Check that the pipeline doesn't change the outcome of the classification.
    Do this by putting rois of know classificion (which we know get correctly classified independintly),
    and then use the same model in a pipeline analysing the synthetic image.
    '''

    with tempfile.TemporaryDirectory() as tempdir:
        # location of the training data
        database_path = os.path.join(tempdir, 'silcam_classification_database')

        exampledata.get_classifier_database_from_pysilcam_blob(database_path)
        os.makedirs('model', exist_ok=True)
        model_path = exampledata.get_example_model('model')

        # Load the trained tensorflow model and class names
        cl = pyopia.classify.Classify(model_path=model_path)

        def get_good_roi(category):
            '''
            calculate the percentage positive matches for a given category
            '''

            print('category', category)
            # list the files in this category of the training data
            files = sorted(glob(os.path.join(database_path, category, '*.tiff')))
            print(len(files), 'files')

            found_match = 0
            # loop through the database images
            for file in tqdm(files):
                img = np.uint8(skimage.io.imread(file))  # load ROI
                img = np.float64(img) / 255
                prediction = cl.proc_predict(img)  # run prediction from silcam_classify

                if np.max(prediction) < (ACCURACY / 100):
                    continue

                ind = np.argmax(prediction)  # find the highest score

                # check if the highest score matches the correct category
                if cl.class_labels[ind] == category:
                    print('roi file', file)
                    return img, category
            assert found_match == 1, f'classifier not finding matching particle for {category}'

        canvas = np.ones((2048, 2448, 3), np.float64)

        rc_shift = int(2048/len(cl.class_labels)/1.5)
        rc = rc_shift

        classes = sorted(glob(os.path.join(database_path, '*')))

        categories = []

        for cat in classes:
            name = os.path.split(cat)[-1]
            img, category = get_good_roi(name)
            categories.append(category)
            img_shape = np.shape(img)
            rc += rc_shift
            canvas[rc:rc + img_shape[0], rc: rc + img_shape[1], :] = np.float64(img)

        settings = {'general': {'raw_files': None,
                                'pixel_size': 24},
                    'steps': {'note': 'non-standard pipeline.'}
                    }

        # Initialise the pipeline class without running anything
        MyPipeline = pyopia.pipeline.Pipeline(settings=settings, initial_steps='')

        # Get the example trained model
        model_path = pyopia.exampledata.get_example_model(os.getcwd())

        # Add the classifier step description to settings (i.e. metadata)
        MyPipeline.settings['steps'].update({'classifier':
                                            {'pipeline_class': 'pyopia.classify.Classify',
                                             'model_path': model_path}})

        # Execute the classifier step we defined above
        MyPipeline.run_step('classifier')
        # This is the same as running:
        # MyPipeline.data['cl'] = pyopia.classify.Classify(model_path=model_path)
        # Note: the classifier step is special in that it's output is specifically data['cl'], rather than other new keys in data

        MyPipeline.data['imraw'] = canvas
        MyPipeline.data['timestamp'] = pd.Timestamp.now()
        MyPipeline.data['filename'] = ''

        # Add the imageprep step description
        MyPipeline.settings['steps'].update({'imageprep':
                                            {'pipeline_class': 'pyopia.instrument.silcam.ImagePrep',
                                             'image_level': 'imraw'}})
        # Run the step
        MyPipeline.run_step('imageprep')
        # This is the same as running:
        # ImagePrep = pyopia.instrument.silcam.ImagePrep(image_level='imraw')
        # MyPipeline.data = ImagePrep(MyPipeline.data)

        # Add the segmentation step description
        MyPipeline.settings['steps'].update({'segmentation':
                                            {'pipeline_class': 'pyopia.process.Segment',
                                             'threshold': 1,
                                             'segment_source': 'im_minimum'}})
        # Run the step
        MyPipeline.run_step('segmentation')
        # This is the same as running:
        # Segment = pyopia.process.Segment(threshold=settings['steps']['segmentation']['threshold'])
        # data = Segment(data)

        # Add the segmentation step description
        MyPipeline.settings['steps'].update({'statextract':
                                            {'pipeline_class': 'pyopia.process.CalculateStats',
                                             'roi_source': 'imref'}}
                                            )

        # Run the step
        MyPipeline.run_step('statextract')
        # This is the same as running:
        # CalculateStats = pyopia.process.CalculateStats()
        # data = CalculateStats(data)

        stats = pyopia.statistics.add_best_guesses_to_stats(MyPipeline.data['stats'])

        out = [x[12:] for x in stats['best guess'].values]

        print('classes input', categories)
        print('classes measured', out)

        assert categories == out, 'Classes returned from classifier do not match what was given to the pipeline'


if __name__ == "__main__":
    test_match_to_database()
    test_pipeline_classification()
