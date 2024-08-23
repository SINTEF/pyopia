'''
Module containing tools for classifying particle ROIs
'''

import os

import numpy as np
import pandas as pd
from PIL import Image

# import tensorflow here. It must be imported on the processor where it will be used!
# import is therefore here instead of at the top of file.
# consider # noqa: E(?) for flake8 / linting
try:
    from tensorflow import keras
    import tensorflow as tf
except ImportError:
    info_str = 'ERROR: Could not import Keras. Classify will not work'
    info_str += ' until you install tensorflow.\n'
    info_str += 'Use: pip install pyopia[classification]\n'
    info_str += ' or: pip install pyopia[classification-arm64]'
    info_str += ' for tensorflow-macos (silicon chips)'
    raise ImportError(info_str)


class Classify():
    '''
    A classifier class for PyOPIA workflow.
    This is intended as a parent class that can be used as a template for flexible classification methods

    Args:
        model_path=model_path (str)        : path to particle-classifier e.g.
                                '/testdata/model_name/particle_classifier.h5'

    Example:

    .. code-block:: python

        cl = Classify(model_path='/testdata/model_name/particle_classifier.h5')

        prediction = cl.proc_predict(roi) # roi is an image roi to be classified

    Note that :meth:`Classify.load_model()`
    is run when the :class:`Classify` class is initialised.
    If this is used in combination with multiprocessing then the model must be loaded
    on the process where it will be used and not passed between processers
    (i.e. cl must be initialised on that process).

    The config setup looks like this:

    .. code-block:: python

        [steps.classifier]
        pipeline_class = 'pyopia.classify.Classify'
        model_path = 'keras_model.h5' # path to trained nn model

    If `[steps.classifier]`is not defined, the classification will be skipped and no probabilities reported.

    If you want to use an example trained model for SilCam data
    (no guarantee of accuracy for other applications), you can get it using `exampledata`
    within the notebooks folder (https://github.com/SINTEF/pyopia/blob/main/notebooks/exampledata.py):

    .. code-block:: python

        model_path = exampledata.get_example_model()

    '''
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.load_model()

    def __call__(self):
        return self

    def load_model(self):
        '''
        Load the trained tensorflow keras model into the Classify class. example here based on the pysilcam network setup

        model (tf model object) : loaded tf.keras model from load_model()
        '''
        model_path = self.model_path

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        keras.backend.clear_session()

        path, filename = os.path.split(model_path)
        header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
        self.class_labels = header.columns
        self.model = keras.models.load_model(model_path)
        return

    def preprocessing(self, img_input):
        '''
        Preprocess ROI ready for prediction. example here based on the pysilcam network setup

        Args:
            img_input (float)        : a particle ROI before preprocessing with range 0-1

        Returns:
            img_preprocessed (float) : a particle ROI with range 0.-255., corrected and preprocessed, ready for prediction
        '''
        imsize = 128

        # convert back to 0-255 scaling (because of this layer in the network:
        # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
        # This is useful because it allows training to use tf.keras.utils.image_dataset_from_directory, which loads images in 0-255 range
        img = keras.utils.img_to_array(img_input * 255)

        # resize to match the dimentions expected by the network
        img = tf.image.resize(img, [imsize, imsize],
                              method=tf.image.ResizeMethod.BILINEAR,
                              preserve_aspect_ratio=True)

        # This would be the alternative to the above two lines if loading an image directly from disc
        # img = tf.keras.utils.load_img(f, target_size=(img_height, img_width))

        img_array = tf.keras.utils.img_to_array(img)
        img_preprocessed = tf.expand_dims(img_array, 0)  # Create a batch
        return img_preprocessed

    def predict(self, img_preprocessed):
        '''
        Use tensorflow model to classify particles. example here based on the pysilcam network setup.

        Args:
            img_preprocessed (float) : a particle ROI arry, corrected and preprocessed using :meth:`Classify.preprocessing`,
                                       ready for prediction using :meth:`Classify.predict`

        Returns:
            prediction (array)       : the probability of the roi belonging to each class
        '''

        prediction = self.model.predict(img_preprocessed, verbose=0)
        prediction = tf.nn.softmax(prediction[0])
        return prediction

    def proc_predict(self, img_input):
        '''
        Run pre-processing (:meth:`Classify.preprocessing`) and prediction (:meth:`Classify.predict`)
        using tensorflow model to classify particles. example here based on the pysilcam network setup.

        Args:
            img_input (float)  : a particle ROI with range 0-1 before preprocessing

        Returns:
            prediction (array) : the probability of the roi belonging to each class
        '''

        img_preprocessed = self.preprocessing(img_input)
        prediction = self.predict(img_preprocessed)

        return prediction
