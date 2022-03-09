from PIL import Image
import numpy as np
import os
import pandas as pd


class Classify():
    '''
    A classifier class for PyOpia workflow.
    This is intended as a parent class that can be used as a template for flexible classification methods
    '''
    def __init__(self):
        pass

    def load_model(self, model_path):
        '''
        Load the trained tensorflow keras model. example here based on the pysilcam network setup

        Args:
            model_path (str)        : path to particle-classifier e.g.
                                    '/testdata/model_name/particle_classifier.h5'

        Returns:
            model (tf model object) : loaded tf.keras model from load_model()
        '''
        # import tensorflow here. It must be imported on the processor where it will be used!
        # import is therefore here instead of at the top of file.
        # consider # noqa: E(?) for flake8 / linting
        from tensorflow import keras
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
