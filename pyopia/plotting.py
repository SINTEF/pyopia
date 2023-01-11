# -*- coding: utf-8 -*-
'''
Particle plotting functionality: PSD, D50, etc.
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_style('ticks')




def show_imc(imc, mag=2):
    '''
    Plots a scaled figure of for s SilCam image for medium or low magnification systems
    
    Args:
        imc (uint8) : SilCam image (usually a corrected image, such as imc)
        mag=2 (int) : mag=1 scales to the low mag SilCams; mag=2 (default) scales to the medium max SilCams
    '''
    PIX_SIZE = 35.2 / 2448 * 1000
    r, c = np.shape(imc[:, :, 0])

    if mag == 1:
        PIX_SIZE = 67.4 / 2448 * 1000

    plt.imshow(np.uint8(imc),
               extent=[0, c * PIX_SIZE / 1000, 0, r * PIX_SIZE / 1000],
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
    ax.imshow(montage, extent=[0, ex, 0, ex])
    ax.set_xticks([1, 2], [])
    ax.set_xticklabels(['    1mm', ''])
    ax.set_yticks([], [])
    ax.xaxis.set_ticks_position('bottom')
