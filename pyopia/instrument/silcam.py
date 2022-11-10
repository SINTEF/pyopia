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

    def __init__(self, filename):
        self.filename = filename

    def __call__(self, common):
        timestamp = timestamp_from_filename(self.filename)
        img = np.load(self.filename, allow_pickle=False)
        common['timestamp'] = timestamp
        common['img'] = img
        return common


class ImagePrep():

    def __init__(self):
        pass

    def __call__(self, common):
        # @todo
        # #imbg = common['imbg']
        # background correction
        print('WARNING: Background correction not implemented!')
        imraw = common['img']
        imc = np.float64(imraw)

        # simplify processing by squeezing the image dimensions into a 2D array
        # min is used for squeezing to represent the highest attenuation of all wavelengths
        imc = np.min(imc, axis=2)
        imc -= np.min(imc)
        imc /= np.max(imc)

        common['imc'] = imc
        return common
