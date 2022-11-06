import numpy as np
import pandas as pd
from scipy import fftpack
from skimage.io import imread
import pyopia.background

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


class Common():
    '''PyOpia pipline-compatible class for one-time setup of holograhic reconstruction

    Parameters
    ----------
    filename : string
        background hologram filename (.pgm)

    kernel settings : ....
        ....

    Returns
    -------
    @todo . This is an 'output' dict containing:
    imbg : np.array
        background image
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

    def __call__(self):
        print('Load background from disc. \n WARNING: Proper background calculation not implemented here.')
        imbg = imread(self.filename).astype(np.float64)
        print('Build kernel')
        kern = create_kernel(imbg, self.pixel_size, self.wavelength, self.minZ, self.maxZ, self.stepZ)
        print('HoloCommon done', pd.datetime.now())
        output = {'imbg': imbg,
                  'kern': kern}
        return output


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
    img : np.arraym (@todo - check this)
        hologram
    '''

    def __init__(self, filename):
        self.filename = filename

    def __call__(self):
        print('WARNING: timestamp not implemented for holo data! using current time to test workflow.')
        timestamp = pd.datetime.now()
        im = imread(self.filename).astype(np.float64)
        return timestamp, im


class Reconstruct():
    '''PyOpia pipline-compatible class for reconstructing a single holo image

    Parameters
    ----------
    im : np.arry
        hologram image

    Returns
    -------
    image : np.array
        image ready for further segmentation and analysis
    '''

    def __init__(self, stack_clean):
        self.stack_clean = stack_clean

    def __call__(self, imraw, common):
        imbg = common['imbg']
        kern = common['kern']

        print('correct background')
        imc = pyopia.background.subtract_background(imbg, imraw)
        print('forward transform')
        im_fft = forward_transform(imc)
        print('inverse transform')
        im_stack = inverse_transform(im_fft, kern)
        print('clean stack')
        im_stack = clean_stack(im_stack, self.stack_clean)
        print('summarise stack')
        stack_max = max_map(im_stack)
        stack_max = np.max(stack_max) - stack_max
        stack_max -= np.min(stack_max)
        stack_max /= np.max(stack_max)
        # im_stack_inv = holo.rescale_stack(im_stack)
        return stack_max


def forward_transform(im):

    # Perform forward transform
    im_fft = fftpack.fft2(im)

    # Remove the zero frequency components
    im_fft[:, 0] = 0
    im_fft[0, :] = 0

    # fftshift
    im_fft = fftpack.fftshift(im_fft)

    return im_fft


def create_kernel(im, pixel_size, wavelength, minZ, maxZ, stepZ):

    cx = im.shape[1] / 2
    cy = im.shape[0] / 2

    x = (np.arange(0, im.shape[1]) - cx) / cx
    y = (np.arange(0, im.shape[0]) - cy) / cy
    y.shape = (im.shape[0], 1)

    f1 = np.tile(x, (im.shape[0], 1))
    f2 = np.tile(y, (1, im.shape[1]))

    f = (np.pi / (pixel_size / 1e6)) * (f1**2 + f2**2)**0.5

    z = np.arange(minZ * 1e-3, maxZ * 1e-3, stepZ * 1e-3)

    l = wavelength * 1e-9
    k = 2 * np.pi / l

    kern = -1j * np.zeros((im.shape[0], im.shape[1], len(z)))
    for i, z_ in enumerate(z):

        kern[:, :, i] = np.exp(-1j * f**2 * z_ / (2 * k))
    return kern


def inverse_transform(im_fft, kern):
    im_stack = np.zeros(np.shape(kern)).astype(np.float64)

    for i in range(np.shape(kern)[2]):
        im_stack[:, :, i] = (fftpack.ifft2(im_fft * kern[:, :, i]).real)**2

    return im_stack


def clean_stack(im_stack, stack_clean):
    im_max = np.amax(im_stack, axis=None)
    im_stack[im_stack < im_max * stack_clean] = 0
    return im_stack


def std_map(im_stack):
    std_map = np.std(im_stack, axis=2)
    return std_map


def max_map(im_stack):
    max_map = np.max(im_stack, axis=2)
    return max_map


def rescale_stack(im_stack):
    im_max = np.max(im_stack)
    im_min = np.min(im_stack)
    im_stack_inverted = 255 * (im_stack - im_min) / (im_max - im_min)
    im_stack_inverted = 255 - im_stack_inverted
    return im_stack
