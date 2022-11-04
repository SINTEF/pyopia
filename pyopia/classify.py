from PIL import Image
import numpy as np
import os
import pandas as pd


class Classify():
    '''
    A classifier class for PyOpia workflow.
    This is intended as a parent class that can be used as a template for flexible classification methods

    Args:
        model_path (str)        : path to particle-classifier e.g.
                                '/testdata/model_name/particle_classifier.h5'

    example:

    .. code-block:: python

        cl = Classify(model_path='/testdata/model_name/particle_classifier.h5')
        cl.load_model()

        prediction = cl.proc_predict(roi) # roi is an image roi to be classified

    '''
    def __init__(self, model_path=None):
        self.model_path = model_path
        pass

    def __call__(self):
        return self

    def load_model(self):
        '''
        Load the trained tensorflow keras model into the Classify class. example here based on the pysilcam network setup

        model (tf model object) : loaded tf.keras model from load_model()
        '''
        model_path = self.model_path

        # import tensorflow here. It must be imported on the processor where it will be used!
        # import is therefore here instead of at the top of file.
        # consider # noqa: E(?) for flake8 / linting
        try:
            from tensorflow import keras
        except ImportError:
            info_str = 'WARNING: Could not import Keras, Classify will not work'
            info_str += ' until you install tensorflow (pip install tensorflow-cpu)'
            print(info_str)
            self.model = lambda x: None
            self.class_labels = []
            return

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
            img_input (uint8)        : a particle ROI before preprocessing

        Returns:
            img_preprocessed (uint8) : a particle ROI, corrected and preprocessed, ready for prediction
        '''
        # Scale it to 32x32
        img_preprocessed = Image.fromarray(img_input)
        img_preprocessed = img_preprocessed.resize((32, 32), Image.BICUBIC)
        img_preprocessed = np.array(img_preprocessed)

        # Apply temporary fix for image preprocessing that matches the TFL model conversions in the pysilcam
        # Tensorflow model that has been adapted for faster use with keras.
        # Eventual plan is to switch to pytorch and then this can be removed,
        # but we could not get the same accurancy with pytorch as tensorflow
        img_preprocessed = (img_preprocessed - 195.17760394934288) / 56.10742134506719
        return img_preprocessed

    def predict(self, img_preprocessed):
        '''
        Use tensorflow model to classify particles. example here based on the pysilcam network setup.

        Args:
            img_preprocessed (uint8) : a particle ROI, corrected and preprocessed, ready for prediction

        Returns:
            prediction (array)       : the probability of the roi belonging to each class
        '''

        prediction = self.model(np.expand_dims(img_preprocessed, 0))

        return prediction

    def proc_predict(self, img_input):
        '''
        Use tensorflow model to classify particles. example here based on the pysilcam network setup.

        Args:
            img_input (uint8)  : a particle ROI before preprocessing

        Returns:
            prediction (array) : the probability of the roi belonging to each class
        '''

        img_preprocessed = self.preprocessing(img_input)
        prediction = self.predict(img_preprocessed)

        return prediction
