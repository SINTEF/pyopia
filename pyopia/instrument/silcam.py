'''
Module containing SilCam specific tools to enable compatability with the :mod:`pyopia.pipeline`

See:
Davies, E. J., Brandvik, P. J., Leirvik, F., & Nepstad, R. (2017). The use of wide-band transmittance imaging to size and
classify suspended particulate matter in seawater. Marine Pollution Bulletin, 115(1–2). https://doi.org/10.1016/j.marpolbul.2016.11.063
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
    timestamp: timestamp
        timestamp from pandas.to_datetime()
    '''

    # get the timestamp of the image (in this case from the filename)
    timestamp = pd.to_datetime(os.path.splitext(os.path.basename(filename))[0][1:])
    return timestamp


def load_mono8(filename):
    '''load a mono8 .msilc file from disc

    Assumes 8-bit mono image in range 0-255

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
    image_shape = np.shape(im_mono)
    if len(image_shape) > 2:
        if image_shape[2] == 1:
            img = im_mono[:, :, 0]
        else:
            raise RuntimeError('Invalid image dimension')
    return img


def load_bayer_rgb8(filename):
    '''load an RG8 .bsilc file from disc and convert it to RGB image

    Assumes 8-bit Bayer-RG (Red-Green) image in range 0-255

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image float between 0-1
    '''
    img_bayer = np.load(filename, allow_pickle=False).astype(np.int16)

    # Check the image dimension
    image_shape = np.shape(img_bayer)
    if len(image_shape) > 2:
        if image_shape[2] == 1:
            img_bayer = img_bayer[:, :, 0]
        else:
            raise RuntimeError('Invalid image dimension')

    M, N = img_bayer.shape[:2]   # Number of pixels in image height and width
    img_bayer_min, img_bayer_max = np.min(img_bayer), np.max(img_bayer)

    # img is a reconstructed RGB image
    img = np.zeros((M, N, 3), dtype=np.uint8)
    img[0:M:2, 0:N:2, 0] = img_bayer[0:M:2, 0:N:2]  # Red pixels
    img[0:M:2, 1:N:2, 1] = img_bayer[0:M:2, 1:N:2]  # Green pixels on the first row
    img[1:M:2, 0:N:2, 1] = img_bayer[1:M:2, 0:N:2]  # Green pixels on the second row
    img[1:M:2, 1:N:2, 2] = img_bayer[1:M:2, 1:N:2]  # Blue pixels

    # Boundary pixels interpolation in the first and last rows
    # ***Red pixels
    img[0, 2:N-1:2, 1] = (img_bayer[1, 2:N-1:2] + img_bayer[0, 1:N-2:2] + img_bayer[0, 3:N:2] + 1) // 3  # Interpolated G
    img[0, 2:N-1:2, 2] = (img_bayer[1, 1:N-2:2] + img_bayer[1, 3:N:2] + 1) // 2  # Interpolated B
    # ***Green pixels (odd columns)
    img[0, 1:N-2:2, 0] = (img_bayer[0, 0:N-3:2] + img_bayer[0, 2:N-1:2] + 1) // 2  # Interpolated R
    img[0, 1:N-2:2, 2] = img_bayer[1, 1:N-2:2]  # Interpolated B
    # ***Blue pixels
    img[M-1, 1:N-2:2, 1] = (img_bayer[M-2, 1:N-2:2]
                            + img_bayer[M-1, 0:N-3:2] + img_bayer[M-1, 2:N-1:2] + 1) // 3  # Interpolated G
    img[M-1, 1:N-2:2, 0] = (img_bayer[M-2, 0:N-3:2] + img_bayer[M-2, 2:N-1:2] + 1) // 2  # Interpolated R
    # ***Green pixels (even columns)
    img[M-1, 2:N-1:2, 2] = (img_bayer[M-1, 1:N-2:2] + img_bayer[M-1, 3:N:2] + 1) // 2  # Interpolated B
    img[M-1, 2:N-1:2, 0] = img_bayer[M-2, 2:N-1:2]  # Interpolated R

    # Boundary pixels interpolation in the first and last cols
    # ***Red pixels
    img[2:M-1:2, 0, 1] = (img_bayer[3:M:2, 0] + img_bayer[1:M-2:2, 0] + img_bayer[2:M-1:2, 1] + 1) // 3  # Interpolated G
    img[2:M-1:2, 0, 2] = (img_bayer[3:M:2, 1] + img_bayer[1:M-2:2, 1] + 1) // 2  # Interpolated B
    # ***Green pixels (odd columns)
    img[1:M-2:2, 0, 0] = (img_bayer[0:M-3:2, 0] + img_bayer[2:M-1:2, 0] + 1) // 2  # Interpolated R
    img[1:M-2:2, 0, 2] = img_bayer[1:M-2:2, 1]  # Interpolated B
    # ***Blue pixels
    img[1:M-2:2, N-1, 1] = (img_bayer[0:M-3:2, N-1]
                            + img_bayer[2:M-1:2, N-1] + img_bayer[1:M-2:2, N-2] + 1) // 3  # Interpolated G
    img[1:M-2:2, N-1, 0] = (img_bayer[0:M-3:2, N-2] + img_bayer[2:M-1:2, N-2] + 1) // 2  # Interpolated R
    # ***Green pixels (even columns)
    img[2:M-1:2, N-1, 0] = img_bayer[2:M-1:2, N-2]  # Interpolated R
    img[2:M-1:2, N-1, 2] = (img_bayer[1:M-2:2, N-1]+img_bayer[3:M:2, N-1] + 1)//2  # Interpolated B

    # Corner pixels interpolation
    # *** top-left
    img[0, 0, 1] = (img_bayer[1, 0] + img_bayer[0, 1] + 1) // 2  # Interpolated G
    img[0, 0, 2] = img_bayer[1, 1]  # Interpolated B
    # *** top-right
    img[0, N-1, 0] = img_bayer[0, N-2]  # Interpolated R
    img[0, N-1, 2] = img_bayer[1, N-1]  # Interpolated B
    # *** bottom-left
    img[M-1, 0, 0] = img_bayer[M-2, 0]  # Interpolated R
    img[M-1, 0, 2] = img_bayer[M-1, 1]  # Interpolated B
    # *** bottom-right
    img[M-1, N-1, 1] = (img_bayer[M-2, N-1] + img_bayer[M-1, N-2] + 1) // 2  # Interpolated G
    img[M-1, N-1, 0] = img_bayer[M-2, N-2]  # Interpolated R

    # Internal pixels interpolation
    # ***G pixel on odd row, even column
    img[1:M-2:2, 2:N-1:2, 0] = (img_bayer[0:M-3:2, 2:N-1:2] + img_bayer[2:M-1:2, 2:N-1:2] + 1) // 2  # Interpolated R
    img[1:M-2:2, 2:N-1:2, 2] = (img_bayer[1:M-2:2, 1:N-2:2] + img_bayer[1:M-2:2, 3:N:2] + 1) // 2  # Interpolated B
    # ***G pixel on even row, odd column
    img[2:M-1:2, 1:N-2:2, 0] = (img_bayer[2:M-1:2, 0:N-3:2] + img_bayer[2:M-1:2, 2:N-1:2] + 1) // 2  # Interpolated R
    img[2:M-1:2, 1:N-2:2, 2] = (img_bayer[1:M-2:2, 1:N-2:2] + img_bayer[3:M:2, 1:N-2:2] + 1) // 2  # Interpolated B
    # ***R pixel
    img[2:M-1:2, 2:N-1:2, 1] = (img_bayer[1:M-2:2, 2:N-1:2]
                                + img_bayer[3:M:2, 2:N-1:2] + img_bayer[2:M-1:2, 1:N-2:2]
                                + img_bayer[2:M-1:2, 3:N:2] + 2) // 4  # Interpolated G
    img[2:M-1:2, 2:N-1:2, 2] = (img_bayer[1:M-2:2, 1:N-2:2]
                                + img_bayer[3:M:2, 1:N-2:2] + img_bayer[3:M:2, 3:N:2]
                                + img_bayer[1:M-2:2, 3:N:2] + 2) // 4  # Interpolated B
    # ***B pixel
    img[1:M-2:2, 1:N-2:2, 0] = (img_bayer[0:M-3:2, 0:N-3:2]
                                + img_bayer[2:M-1:2, 0:N-3:2] + img_bayer[2:M-1:2, 2:N-1:2]
                                + img_bayer[0:M-3:2, 2:N-1:2] + 2) // 4  # Interpolated R
    img[1:M-2:2, 1:N-2:2, 1] = (img_bayer[0:M-3:2, 1:N-2:2]
                                + img_bayer[2:M-1:2, 1:N-2:2] + img_bayer[1:M-2:2, 0:N-3:2]
                                + img_bayer[1:M-2:2, 2:N-1:2] + 2) // 4  # Interpolated G
    img_min, img_max = np.min(img), np.max(img)

    if img_min < 0 or img_max > 255 or (img_max - img_bayer_max) != 0 or (img_min - img_bayer_min) != 0:
        raise ValueError(
            "The converted RGB image is not suitable for further analysis. Check the pixel intensity ranges of the input image."
            )

    img = img.astype(np.float64) / 255
    return img


def load_rgb8(filename):
    '''load an RGB .silc file from disc

    Assumes 8-bit RGB image in range 0-255

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
    '''.. deprecated:: 2.4.6
        :func:`pyopia.instrument.silcam.load_image` will be removed in version 3.0.0, it is replaced by
        :func:`pyopia.instrument.silcam.load_rgb8` because this is more explicit to that image type.

    Load an RGB .silc file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image float between 0-1
    '''

    return load_rgb8(filename)


class SilCamLoad():
    '''PyOpia pipline-compatible class for loading a single silcam image
    and extracting the timestamp using
    :func:`pyopia.instrument.silcam.timestamp_from_filename`

    Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.filename`

    Parameters
    ----------
    image_format : str, optional
        Image file format. Can be either 'infer', 'rgb8', 'bayer_rg8' or 'mono8', by default 'infer'.

    Note
    ----
        'infer' uses the file extension to determine the image format using the following convention:
         - '.silc' for RGB8
         - '.msilc' for MONO8
         - '.bsilc' for BAYER_RG8
         - '.bmp' for using skimage.io.imread

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
                               '.bsilc': load_bayer_rgb8,
                               '.bmp': skimage.io.imread}
        self.format_load = {'RGB8': load_rgb8,
                            'MONO8': load_mono8, 'BAYER_RG8': load_bayer_rgb8}

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
                'pipeline_class': 'pyopia.io.StatsToDisc',
                'output_datafile': os.path.join(outfolder, output_prefix)
            }
        }
    }
    return pipeline_config
