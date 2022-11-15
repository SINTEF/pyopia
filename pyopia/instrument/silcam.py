'''
Module containing SilCam specific tools to enable compatability with the :mod:`pyopia.pipeline`
'''

import os
from dataclasses import dataclass
from typing import Any

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


@dataclass
class LoadData:
    timestamp: Any
    img: Any


class SilCamLoad():
    '''PyOpia pipline-compatible class for loading a single silcam image

    Requires pipeline data: :class:`LoadData`

    Parameters
    ----------
    filename : string
        silcam filename (.silc)

    Returns
    -------
        data.img : :class:`LoadData`
        data.timestamp : :class:`LoadData`
    '''

    def __init__(self, filename):
        self.filename = filename

    def __call__(self, _) -> LoadData:
        timestamp = timestamp_from_filename(self.filename)
        img = np.load(self.filename, allow_pickle=False)
        data = LoadData(timestamp=timestamp, img=img) # doing this will cause an error in pyopia.process.CalculateStats() by removing data.cl that should have been made by the first step: Classify(model_path=model_path)
        return data


@dataclass
class ImagePrepData(LoadData):
    imc: np.ndarray


class ImagePrep():
    '''Simplify processing by squeezing a 3-channel image into a 2D array

    min is used for squeezing to represent the highest attenuation of all wavelengths

    Requires pipeline data: :class:`ImagePrepData`

    Returns
    -------
        data.imc : :class:`ImagePrepData`
    '''

    def __init__(self):
        pass

    def __call__(self, data: LoadData) -> ImagePrepData:
        # @todo
        # #imbg = data.imbg
        # background correction
        print('WARNING: Background correction not implemented!')
        imraw = data.img
        imc = np.float64(imraw)

        # simplify processing by squeezing the image dimensions into a 2D array
        # min is used for squeezing to represent the highest attenuation of all wavelengths
        imc = np.min(imc, axis=2)
        imc -= np.min(imc)
        imc /= np.max(imc)

        data.imc = imc
        return data
