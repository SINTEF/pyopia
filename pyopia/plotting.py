# -*- coding: utf-8 -*-
'''
Particle plotting functionality for standardised figures
e.g. image presentation, size distributions, montages etc.
'''

import matplotlib.pyplot as plt
import numpy as np


def show_imc(imc, pixel_size):
    '''
    Plots a scaled figure (in mm) of an image

    Args:
        imc (uint8 or float) : Image (usually a corrected image, such as imc)
        pixel_size (float) : the pixel size (um) of the imaging system used
    '''
    r, c = np.shape(imc[:, :, 0])

    plt.imshow(imc,
               extent=[0, c * pixel_size / 1000, 0, r * pixel_size / 1000],
               interpolation='nearest')
    plt.xlabel('mm')
    plt.ylabel('mm')

    return


def montage_plot(montage, pixel_size):
    '''
    Plots a SilCam particle montage with a 1mm scale reference

    Args:
        montage (uint8)    : a SilCam montage created with scpp.make_montage
        pixel_size (float) : the pixel size of the SilCam used, obtained from settings.PostProcess.pix_size in the
                             config ini file
    '''
    msize = np.shape(montage)[0]
    ex = pixel_size * np.float64(msize) / 1000.

    ax = plt.gca()
    ax.imshow(montage, extent=[0, ex, 0, ex], cmap='grey')
    ax.set_xticks([1, 2], [])
    ax.set_xticklabels(['    1mm', ''])
    ax.set_yticks([], [])
    ax.xaxis.set_ticks_position('bottom')
