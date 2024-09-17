# -*- coding: utf-8 -*-
'''
Particle plotting functionality for standardised figures
e.g. image presentation, size distributions, montages etc.
'''

import matplotlib.pyplot as plt
import numpy as np


def show_image(image, pixel_size):
    '''Plots a scaled figure (in mm) of an image

    Parameters
    ----------
    image : float
        Image (usually a corrected image, such as im_corrected)
    pixel_size : float
        the pixel size (um) of the imaging system used
    '''
    r, c = np.shape(image[:, :, 0])

    plt.imshow(image,
               extent=[0, c * pixel_size / 1000, 0, r * pixel_size / 1000],
               interpolation='nearest')
    plt.xlabel('mm')
    plt.ylabel('mm')

    return


def montage_plot(montage, pixel_size):
    '''
    Plots a SilCam particle montage with a 1mm scale reference

    Parameters
    ----------
    montage : uint8
        a montage created with scpp.make_montage
    pixel_size : float
        the pixel size (um) of the imaging system used
    '''
    msize = np.shape(montage)[0]
    ex = pixel_size * np.float64(msize) / 1000.

    ax = plt.gca()
    ax.imshow(montage, extent=[0, ex, 0, ex], cmap='grey')
    ax.set_xticks([1, 2], [])
    ax.set_xticklabels(['    1mm', ''])
    ax.set_yticks([], [])
    ax.xaxis.set_ticks_position('bottom')
