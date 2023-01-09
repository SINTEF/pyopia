'''
Module containing SilCam specific tools to enable compatability with the :mod:`pyopia.pipeline`
'''

import os

import numpy as np
import pandas as pd


def timestamp_from_filename(filename):
    '''get a pandas timestamp from a silcam filename

    Args:
        filename (string): silcam filename (.silc)

    Returns:
        timestamp: timestamp from pandas.to_datetime()
    '''

    # get the timestamp of the image (in this case from the filename)
    timestamp = pd.to_datetime(os.path.splitext(os.path.basename(filename))[0][1:])
    return timestamp


def load_image(filename):
    img = np.load(filename, allow_pickle=False)
    return img


class SilCamLoad():
    '''PyOpia pipline-compatible class for loading a single silcam image

    Parameters
    ----------
    filename : string
        silcam filename (.silc)

    Returns
    -------
    timestamp : timestamp
        timestamp from timestamp_from_filename()
    img : np.array
        raw silcam image
    '''

    def __init__(self):
        pass

    def __call__(self, data):
        timestamp = timestamp_from_filename(data['filename'])
        img = load_image(data['filename'])
        data['timestamp'] = timestamp
        data['img'] = img
        return data


class ImagePrep():

    def __init__(self):
        pass

    def __call__(self, data):
        im_corrected = data['imc']
        data['imref'] = im_corrected
        imc = np.float64(im_corrected)

        # simplify processing by squeezing the image dimensions into a 2D array
        # min is used for squeezing to represent the highest attenuation of all wavelengths
        imc = np.min(imc, axis=2)
        imc -= np.min(imc)
        imc /= np.max(imc)

        data['imc'] = imc
        return data
