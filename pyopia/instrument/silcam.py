'''
Module containing SilCam specific tools to enable compatability with the :mod:`pyopia.pipeline`
'''

import os
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
import skimage.io


def timestamp_from_filename(filename):
    '''get a pandas timestamp from a silcam filename

    Parameters
    ----------
        filename (string): silcam filename (.silc)

    Returns
    -------
        timestamp: timestamp from pandas.to_datetime()
    '''

    # get the timestamp of the image (in this case from the filename)
    timestamp = pd.to_datetime(os.path.splitext(os.path.basename(filename))[0][1:])
    return timestamp


def load_mono8(filename):
    '''load a mono8 .msilc file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image float between 0-1
    '''
    im_mono = np.load(filename, allow_pickle=False).astype(np.float64) / 255
    #img = np.zeros((np.shape(im_mono)[0], np.shape(im_mono)[1], 3), dtype=np.float64)
    #for channel in range(3):
    img = im_mono[:, :, 0]
    return img


def load_rgb8(filename):
    '''load an RGB .silc file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image float between 0-1
    '''
    img = np.load(filename, allow_pickle=False).astype(np.float64) / 255
    return img


def load_image(filename):
    '''load an RGB .silc file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image float between 0-1
        
    .. deprecated:: 2.4.6
          :func:`pyopia.instrument.silcam.load_image` will be removed in version 3.0.0, it is replaced by
          :func:`pyopia.instrument.silcam.load_rgb8` because this is more explicit to that image type.
    '''
    
    return load_rgb8(filename)

class SilCamLoad():
    '''PyOpia pipline-compatible class for loading a single silcam image
    using :func:`pyopia.instrument.silcam.load_image`
    and extracting the timestamp using
    :func:`pyopia.instrument.silcam.timestamp_from_filename`

    Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.filename`

    Parameters
    ----------
    image_format : str, optional
        .silc file format. Can be either 'infer', 'rgb8' or 'mono8'.
        'infer' uses the file extension to determine the image format using the following convention:
         - '.silc' for RGB8
         - '.msilc' for MONO8
         - '.bmp' for using skimage.io.imread
        , by default 'infer'

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.timestamp`

        :attr:`pyopia.pipeline.Data.img`
    '''

    def __init__(self, image_format='infer'):
        self.image_format = image_format
        self.extension_load = {'.silc': load_rgb8,
                               '.msilc': load_mono8,
                               '.bmp': skimage.io.imread}
        self.format_load = {'RGB8': load_rgb8,
                            'MONO8': load_mono8}

    def __call__(self, data):
        data['timestamp'] = timestamp_from_filename(data['filename'])
        data['imraw'] = self.load_image(data['filename'])
        return data
    
    def load_image(self, filename):
        if self.image_format == 'infer':
            file_extension = os.path.splitext(os.path.basename(filename))[-1]
            load_function = self.extension_load[file_extension]
        else:
            load_function = self.format_load[self.image_format]
        img = load_function(filename)
        return img


class ImagePrep():
    '''PyOpia pipline-compatible class for preparing silcam images for further analysis

    Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.img`

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.im_minimum`
    '''
    def __init__(self, image_level='im_corrected'):
        self.image_level = image_level
        pass

    def __call__(self, data):
        image = data[self.image_level]

        # simplify processing by squeezing the image dimensions into a 2D array
        # min is used for squeezing to represent the highest attenuation of all wavelengths
        data['im_minimum'] = np.min(image, axis=2)

        data['imref'] = rescale_intensity(image, out_range=(0, 1))
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

    Returns
    -------
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
                'threshold': 0.85,
                'segment_source': 'im_minimum'
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats',
                'roi_source': 'imref'
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsH5',
                'output_datafile': os.path.join(outfolder, output_prefix)
            }
        }
    }
    return pipeline_config
