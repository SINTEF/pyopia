'''
Module containing hologram specific tools to enable compatability with the :mod:`pyopia.pipeline`
'''

import numpy as np
import pandas as pd
from scipy import fftpack
from skimage.io import imread
from skimage.filters import sobel
import pyopia.process

'''
This is an subpackage containing basic processing for reconstruction of in-line holographic images.

See (and references therein):
Davies EJ, Buscombe D, Graham GW & Nimmo-Smith WAM (2015)
'Evaluating Unsupervised Methods to Size and Classify Suspended Particles
Using Digital In-Line Holography'
Journal of Atmospheric and Oceanic Technology 32, (6) 1241-1256,
https://doi.org/10.1175/JTECH-D-14-00157.1
https://journals.ametsoc.org/view/journals/atot/32/6/jtech-d-14-00157_1.xml

2022-11-01 Alex Nimmo-Smith alex.nimmo.smith@plymouth.ac.uk
'''


class Initial():
    '''PyOpia pipline-compatible class for one-time setup of holograhic reconstruction

    Parameters
    ----------
    filename : string
        hologram filename to use for image size

    kernel settings : ....
        ....

    Returns
    -------
    @todo . This is an 'output' dict containing:
    kern : np.arry
        reconstruction kernel

    '''

    def __init__(self, filename, pixel_size, wavelength, minZ, maxZ, stepZ):
        self.filename = filename
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.minZ = minZ
        self.maxZ = maxZ
        self.stepZ = stepZ

    def __call__(self, data):
        print('Using first raw file to determine image dimensions')
        imtmp = imread(self.filename).astype(np.float64)
        print('Build kernel')
        kern = create_kernel(imtmp, self.pixel_size, self.wavelength, self.minZ, self.maxZ, self.stepZ)
        print('HoloInitial done', pd.datetime.now())
        data['kern'] = kern
        return data


def load_image(filename):
    '''load a hologram image file from disc

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    array
        raw image
    '''
    img = imread(filename).astype(np.float64)
    return img


class Load():
    '''PyOpia pipline-compatible class for loading a single holo image

    Parameters
    ----------
    filename : string
        hologram filename (.pgm)

    Returns
    -------
    timestamp : timestamp
        timestamp @todo
    imraw : np.arraym
        hologram
    '''

    def __init__(self):
        pass

    def __call__(self, data):
        print('WARNING: timestamp not implemented for holo data! using current time to test workflow.')
        print(data['filename'])
        timestamp = pd.datetime.now()
        im = imread(data['filename']).astype(np.float64)
        data['timestamp'] = timestamp
        data['imraw'] = im
        return data


class Reconstruct():
    '''PyOpia pipline-compatible class for reconstructing a single holo image

    Parameters
    ----------
    stack_clean : float
        defines amount of cleaning of stack (fraction of max value below which to zero)

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`
        containing the following keys:

        :attr:`pyopia.pipeline.Data.img`

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.im_stack`
    '''

    def __init__(self, stack_clean=0):
        self.stack_clean = stack_clean

    def __call__(self, data):
        imc = data['imc']
        kern = data['kern']

        im_fft = forward_transform(imc)
        im_stack = inverse_transform(im_fft, kern)
        data['im_stack'] = clean_stack(im_stack, self.stack_clean)

        return data


def forward_transform(im):
    '''Perform forward transform
    Remove the zero frequency components and then fftshift

    Parameters
    ----------
    im : np.array
        hologram (usually background-corrected)

    Returns
    -------
    im_fft : np.array
        im_fft
    '''
    # Perform forward transform
    im_fft = fftpack.fft2(im)

    # Remove the zero frequency components
    im_fft[:, 0] = 0
    im_fft[0, :] = 0

    # fftshift
    im_fft = fftpack.fftshift(im_fft)

    return im_fft


def create_kernel(im, pixel_size, wavelength, minZ, maxZ, stepZ):
    '''create reconstruction kernel

    Parameters
    ----------
    im : np.arry
        hologram
    pixel_size : float
        pixel_size in microns per pixel (i.e. usually 4.4 for lisst-holo type of resolution)
    wavelength : float
        laser wavelength in nm
    minZ : float
        minimum reconstruction distance in mm
    maxZ : float
        maximum reconstruction distance in mm
    stepZ : float
        step size in mm (i.e. resolution of reconstruction between minZ and maxZ)

    Returns
    -------
    np.array
        holographic reconstruction kernel (3D array of complex numbers)
    '''
    cx = im.shape[1] / 2
    cy = im.shape[0] / 2

    x = (np.arange(0, im.shape[1]) - cx) / cx
    y = (np.arange(0, im.shape[0]) - cy) / cy
    y.shape = (im.shape[0], 1)

    f1 = np.tile(x, (im.shape[0], 1))
    f2 = np.tile(y, (1, im.shape[1]))

    f = (np.pi / (pixel_size / 1e6)) * (f1**2 + f2**2)**0.5

    z = np.arange(minZ * 1e-3, maxZ * 1e-3, stepZ * 1e-3)

    wavelength_m = wavelength * 1e-9
    k = 2 * np.pi / wavelength_m

    kern = -1j * np.zeros((im.shape[0], im.shape[1], len(z)))
    for i, z_ in enumerate(z):

        kern[:, :, i] = np.exp(-1j * f**2 * z_ / (2 * k))
    return kern


def inverse_transform(im_fft, kern):
    '''create the reconstructed hologram stack of real images

    Parameters
    ----------
    im_fft : np.array
        calculated from forward_transform
    kern : np.array
        calculated from create_kernel

    Returns
    -------
    np.arry
        im_stack
    '''
    im_stack = np.zeros(np.shape(kern)).astype(np.float64)

    for i in range(np.shape(kern)[2]):
        im_stack[:, :, i] = (fftpack.ifft2(im_fft * kern[:, :, i]).real)**2

    return im_stack


def clean_stack(im_stack, stack_clean):
    '''clean the im_stack by removing low value pixels - set to 0 to disable

    Parameters
    ----------
    im_stack : np.array

    stack_clean : flaot
        pixels below this value will be zeroed

    Returns
    -------
    np.array
        cleaned version of im_stack
    '''
    if stack_clean > 0.0:
        im_max = np.amax(im_stack, axis=None)
        im_stack[im_stack < im_max * stack_clean] = 0
    return im_stack


def std_map(im_stack):
    '''_summary_

    Parameters
    ----------
    im_stack : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    std_map = np.std(im_stack, axis=2)
    return std_map


def max_map(im_stack):
    '''_summary_

    Parameters
    ----------
    im_stack : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    max_map = np.max(im_stack, axis=2)
    return max_map


def rescale_stack(im_stack):
    '''rescale the reconstructed stack so that particles look dark on a light background

    Parameters
    ----------
    im_stack : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    im_max = np.max(im_stack)
    im_min = np.min(im_stack)
    im_stack_inverted = 255 * (im_stack - im_min) / (im_max - im_min)
    im_stack_inverted = 255 - im_stack_inverted
    return im_stack_inverted


def rescale_image(im):
    '''rescale im (e.g. may be stack summary) to be dark particles on light background, 8 bit

    Parameters
    ----------
    im : image
        input image to be scaled

    Returns
    -------
    im : image
        scaled and inverted image
    '''
    im_max = np.max(im)
    im_min = np.min(im)
    im = 255 * (im - im_min) / (im_max - im_min)
    im = 255 - im
    return im


def find_focus_imax(im_stack, bbox):
    '''finds and returns the focussed image for the bbox region within im_stack
    using intensity of bbox area

    Parameters
    ----------
    im_stack : nparray
        image stack

    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col)

    Returns
    -------
    im : image
        focussed image for bbox

    ifocus: int
        index through stack of focussed image
    '''
    im_seg = im_stack[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    focus = np.sum(im_seg, axis=(0, 1))
    ifocus = np.argmax(focus)

    return im_seg[:, :, ifocus], ifocus


def find_focus_sobel(im_stack, bbox):
    '''finds and returns the focussed image for the bbox region within im_stack
    using edge magnitude of bbox area

    Parameters
    ----------
    im_stack : nparray
        image stack

    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col)

    Returns
    -------
    im : image
        focussed image for bbox

    ifocus: int
        index through stack of focussed image
    '''
    im_bbox = im_stack[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    im_seg = np.empty_like(im_bbox)
    for zi in range(im_seg.shape[2]):
        im_seg[:, :, zi] = sobel(im_bbox[:, :, zi])

    focus = np.sum(im_seg, axis=(0, 1))
    ifocus = np.argmax(focus)

    return im_seg[:, :, ifocus], ifocus


class Focus():
    '''PyOpia pipline-compatible class for creating a focussed image from an image stack

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.im_stack`

    Parameters
    ----------
    stacksummary_function : (function object, optional)
        Function used to summarise the stack
        Available functions are:

        :func:`pyopia.instrument.holo.max_map`

        :func:`pyopia.instrument.holo.std_map` (default)

    threshold : float
        threshold to apply during initial segmentation

    focus_function : (function object, optional)
        Function used to focus particles within the stack
        Available functions are:

        :func:`pyopia.instrument.holo.find_focus_imax` (default)

        :func:`pyopia.instrument.holo.find_focus_sobel`

    Returns
    -------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.imc`

        :attr:`pyopia.pipeline.Data.imss`

        :attr:`pyopia.pipeline.Data.stack_rp`

        :attr:`pyopia.pipeline.Data.stack_ifocus`
    '''

    def __init__(self, stacksummary_function=std_map, threshold=0.9, focus_function=find_focus_imax):
        self.stacksummary_function = stacksummary_function
        self.threshold = threshold
        self.focus_function = focus_function
        pass

    def __call__(self, data):
        im_stack = data['im_stack']
        imss = self.stacksummary_function(im_stack)
        imss = rescale_image(imss)
        data['imss'] = imss

        # segment imss to find particle x-y locations
        imssbw = pyopia.process.segment(imss, self.threshold)
        # identify particles
        region_properties = pyopia.process.measure_particles(imssbw)
        # loop through bounding boxes to focus each particle and add to output imc
        imc = np.zeros_like(im_stack[:, :, 0])
        ifocus = []
        for rp in region_properties:
            focus_result = self.focus_function(im_stack, rp.bbox)
            im_focus = 255 - focus_result[0]
            ifocus.append(focus_result[1])
            imc[rp.bbox[0]:rp.bbox[2], rp.bbox[1]:rp.bbox[3]] = im_focus

        data['imc'] = imc
        data['stack_rp'] = region_properties
        data['stack_ifocus'] = ifocus
        return data


class MergeStats():
    '''PyOpia pipline-compatible class for merging holo-specific statistics into output stats

    Parameters
    ----------

    Returns
    -------
    updated stats

    '''

    def __init__(self):
        pass

    def __call__(self, data):
        stats = data['stats']
        stack_rp = data['stack_rp']
        stack_ifocus = data['stack_ifocus']

        bbox = np.empty((0, 4), int)
        for rp in stack_rp:
            bbox = np.append(bbox, [rp.bbox], axis=0)

        ifocus = []
        for idx, minr in enumerate(stats.minr):
            total_diff = (abs(bbox[:, 0] - stats.minr[idx]) + abs(bbox[:, 1] - stats.minc[idx])
                        + abs(bbox[:, 2] - stats.maxr[idx]) + abs(bbox[:, 3] - stats.maxc[idx]))
            ifocus.append(stack_ifocus[np.argmin(total_diff)])

        stats['ifocus'] = ifocus
        data['stats'] = stats
        return data
