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
    '''load a .silc file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image
    '''
    img = np.load(filename, allow_pickle=False).astype(np.float64)
    return img


class SilCamLoad():
    '''PyOpia pipline-compatible class for loading a single silcam image
    using :func:`pyopia.instrument.silcam.load_image`
    and extracting the timestamp using
    :func:`pyopia.instrument.silcam.timestamp_from_filename`

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`
        containing the following keys:

        :attr:`pyopia.pipeline.Data.filename`

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.timestamp`

        :attr:`pyopia.pipeline.Data.img`
    '''

    def __init__(self):
        pass

    def __call__(self, data):
        timestamp = timestamp_from_filename(data['filename'])
        img = load_image(data['filename'])
        data['timestamp'] = timestamp
        data['imraw'] = img
        return data


class ImagePrep():
    '''PyOpia pipline-compatible class for preparing silcam images for further analysis

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`
        containing the following keys:

        :attr:`pyopia.pipeline.Data.img`

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.imc`
    '''
    def __init__(self, image_level='imc'):
        self.image_level = image_level
        pass

    def __call__(self, data):
        image = data[self.image_level]
        data['imref'] = image
        imc = np.float64(image)

        # simplify processing by squeezing the image dimensions into a 2D array
        # min is used for squeezing to represent the highest attenuation of all wavelengths
        imc = np.min(imc, axis=2)
        imc /= 255

        data['imc'] = imc
        return data


def generate_config(raw_files: str, model_path: str, outfolder: str, output_prefix: str):
    '''Generate example silcam config.toml as a dict

    Parameters
    ----------
    raw_files : str
        raw_files
    model_path : str
        model_path
    outfolder : str
        outfolder
    output_prefix : str
        output_prefix

    Returns:
    --------
    dict
        pipeline_config toml dict
    '''
    # define the configuration to use in the processing pipeline - given as a dictionary - with some values defined above
    pipeline_config = {
        'general': {
            'raw_files': raw_files,
            'pixel_size': 28  # pixel size in um
        },
        'steps': {
            'classifier': {
                'pipeline_class': 'pyopia.classify.Classify',
                'model_path': model_path
            },
            'load': {
                'pipeline_class': 'pyopia.instrument.silcam.SilCamLoad'
            },
            'imageprep': {
                'pipeline_class': 'pyopia.instrument.silcam.ImagePrep',
                'image_level': 'imraw'
            },
            'segmentation': {
                'pipeline_class': 'pyopia.process.Segment',
                'threshold': 0.85
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats'
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsH5',
                'output_datafile': os.path.join(outfolder, output_prefix)
            }
        }
    }
    return pipeline_config
