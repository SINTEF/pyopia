'''
Module containing UVP specific tools to enable compatability with the :mod:`pyopia.pipeline`
'''

import os
import numpy as np
import pandas as pd
import skimage.io


def timestamp_from_filename(filename):
    '''get a pandas timestamp from a UVP vignette image filename

    Args:
        filename (string): UVP filename (.png)

    Returns:
        timestamp: timestamp from pandas.to_datetime()
    '''

    # get the timestamp of the image (in this case from the filename)
    timestr = os.path.split(filename)[-1].strip('.png')
    timestamp = pd.to_datetime(timestr)
    return timestamp


def load_image(filename):
    '''load a UVP .png file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image float between 0-1, inverted so that particles are dark on a light background
    '''
    img = (255 - skimage.io.imread(filename).astype(np.float64)) / 255
    return img


class UVPLoad():
    '''PyOpia pipline-compatible class for loading a single UVP image
    using :func:`pyopia.instrument.uvp.load_image`
    and extracting the timestamp using
    :func:`pyopia.instrument.uvp.timestamp_from_filename`

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


def generate_config(raw_files: str, model_path: str, outfolder: str, output_prefix: str):
    '''Generate example uvp config.toml as a dict

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
            'pixel_size': 80  # pixel size in um
        },
        'steps': {
            'classifier': {
                'pipeline_class': 'pyopia.classify.Classify',
                'model_path': model_path
            },
            'load': {
                'pipeline_class': 'pyopia.instrument.uvp.UVPLoad'
            },
            'segmentation': {
                'pipeline_class': 'pyopia.process.Segment',
                'threshold': 0.95,
                'segment_source': 'imraw'
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats',
                'roi_source': 'imraw'
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsH5',
                'output_datafile': os.path.join(outfolder, output_prefix)
            }
        }
    }
    return pipeline_config
