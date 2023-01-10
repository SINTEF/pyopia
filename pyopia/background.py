'''
Background correction module (inherited from PySilCam)
'''

import numpy as np


def ini_background(bgfiles, load_function):
    '''
    Create and initial background stack and average image

    Args:
        bgfiles (list)                   : list of strings of filenames to be used in background creation
        load_function (function object)  : this function should take a filename and return an image,
                                           for example: :func:`pyopia.instrument.silcam.load_image`
    Returns:
        bgstack (list)              : list of all images in the background stack
        imbg (array)                : background image
    '''
    bgstack = []
    for f in bgfiles:
        im = load_function(f)
        bgstack.append(im)

    imbg = np.mean(bgstack, axis=0)  # average the images in the stack

    return bgstack, imbg


def pass_bgstack(bgstack, imbg, imraw):
    '''
    Pass through variables for use when wanted as static instead of shifting background

    Args:
        bgstack (list)                  : list of all images in the background stack
        imbg (uint8)                    : background image
        imraw (uint8)                   : raw image

    Returns:
        bgstack (list)                  : unmodified list of all images in the background stack
        imbg (uint8)                    : unmodified background averaged image
    '''
    return bgstack, imbg


def shift_bgstack_accurate(bgstack, imbg, imnew):
    '''
    Shifts the background by popping the oldest and added a new image

    The new background is calculated slowly by computing the mean of all images
    in the background stack.

    Args:
        bgstack (list)      : list of all images in the background stack
        imbg (uint8)        : background image
        imnew (unit8)       : new image to be added to stack

    Returns:
        bgstack (updated list of all background images)
        imbg (updated actual background image)
    '''
    bgstack.pop(0)  # pop the oldest image from the stack,
    bgstack.append(imnew)  # append the new image to the stack
    imbg = np.mean(bgstack, axis=0)
    return bgstack, imbg


def shift_bgstack_fast(bgstack, imbg, imnew):
    '''
    Shifts the background by popping the oldest and added a new image

    The new background is appoximated quickly by subtracting the old image and
    adding the new image (both scaled by the stacklength).
    This is close to a running mean, but not quite.

    Args:
        bgstack (list)      : list of all images in the background stack
        imbg (uint8)        : background image
        imnew (unit8)       : new image to be added to stack

    Returns:
        bgstack (updated list of all background images)
        imbg (updated actual background image)
    '''
    stacklength = len(bgstack)
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    # subtract the old image from the average (scaled by the average window)
    imbg -= (imold / stacklength)
    # add the new image to the average (scaled by the average window)
    imbg += (imnew / stacklength)
    bgstack.append(imnew)  # append the new image to the stack
    return bgstack, imbg


def correct_im_accurate(imbg, imraw):
    '''
    Corrects raw image by subtracting the background and scaling the output

    There is a small chance of clipping of imc in both crushed blacks an blown
    highlights if the background or raw images are very poorly obtained

    Args:
      imbg (uint8)  : background averaged image
      imraw (uint8) : raw image

    Returns:
      imc (uint8)   : corrected image
    '''

    imc = np.float64(imraw) - np.float64(imbg)
    imc[:, :, 0] += (255 / 2 - np.percentile(imc[:, :, 0], 50))
    imc[:, :, 1] += (255 / 2 - np.percentile(imc[:, :, 1], 50))
    imc[:, :, 2] += (255 / 2 - np.percentile(imc[:, :, 2], 50))
    imc += 255 - imc.max()

    imc[imc > 255] = 255
    imc[imc < 0] = 0
    imc = np.uint8(imc)

    return imc


def correct_im_fast(imbg, imraw):
    '''
    Corrects raw image by subtracting the background and clipping the ouput
    without scaling

    There is high potential for clipping of imc in both crushed blacks an blown
    highlights, especially if the background or raw images are not properly obtained

    Args:
      imbg (uint8)  : background averaged image
      imraw (uint8) : raw image

    Returns:
      imc (uint8)   : corrected image
    '''
    imc = imraw - imbg

    imc += 215
    imc[imc < 0] = 0
    imc[imc > 255] = 255
    imc = np.uint8(imc)

    return imc


def shift_and_correct(bgstack, imbg, imraw, stacklength, real_time_stats=False):
    '''
    Shifts the background stack and averaged image and corrects the new
    raw image.

    This is a wrapper for shift_bgstack and correct_im

    Args:
        bgstack (list)                  : list of all images in the background stack
        imbg (uint8)                    : background image
        imraw (uint8)                   : raw image
        stacklength (int)               : unsed int here - just there to maintain the same behaviour as
                                          shift_bgstack_fast()
        real_time_stats=False (Bool)    : if True use fast functions, if False use accurate functions

    Returns:
        bgstack (list)                  : list of all images in the background stack
        imbg (uint8)                    : background averaged image
        imc (uint8)                     : corrected image
    '''

    if real_time_stats:
        imc = correct_im_fast(imbg, imraw)
        bgstack, imbg = shift_bgstack_fast(bgstack, imbg, imraw, stacklength)
    else:
        imc = correct_im_accurate(imbg, imraw)
        bgstack, imbg = shift_bgstack_accurate(bgstack, imbg, imraw, stacklength)

    return bgstack, imbg, imc


def backgrounder(av_window, acquire, bad_lighting_limit=None,
                 real_time_stats=False):
    '''
    Generator which interacts with acquire to return a corrected image
    given av_window number of frame to use in creating a moving background

    Args:
        av_window (int)               : number of images to use in creating the background
        acquire (generator object)    : acquire generator object created by the Acquire class
        bad_lighting_limit=None (int) : if a number is supplied it is used for throwing away raw images that have a
                                        standard deviation in colour which exceeds the given value

    Yields:
        timestamp (timestamp)         : timestamp of when raw image was acquired
        imc (uint8)                   : corrected image ready for analysis or plotting
        imraw (uint8)                 : raw image

    Example:

    .. code-block:: python

        avwind = 10 # number of images used for background
        imgen = backgrounder(avwind,acquire,bad_lighting_limit) # setup generator

        n = 10 # acquire 10 images and correct them with a sliding background:
        for i in range(n):
            imc = next(imgen)
            print(i)
    '''

    # Set up initial background image stack
    bgstack, imbg = ini_background(av_window, acquire)
    stacklength = len(bgstack)

    # Aquire images, apply background correction and yield result
    for timestamp, imraw in acquire:

        if bad_lighting_limit is not None:
            bgstack_new, imbg_new, imc = shift_and_correct(bgstack, imbg,
                                                           imraw, stacklength, real_time_stats)

            # basic check of image quality
            r = imc[:, :, 0]
            g = imc[:, :, 1]
            b = imc[:, :, 2]
            s = np.std([r, g, b])
            # ignore bad images
            if s <= bad_lighting_limit:
                bgstack = bgstack_new
                imbg = imbg_new
                yield timestamp, imc, imraw
            else:
                print('bad lighting, std={0}'.format(s))
        else:
            bgstack, imbg, imc = shift_and_correct(bgstack, imbg, imraw,
                                                   stacklength, real_time_stats)
            yield timestamp, imc, imraw


def subtract_background(imbg, imraw):
    ''' simple background substraction

    Returns
    -------
    np.array : image
        image corrected by simple subtraction (imraw - imbw)
    '''
    return imraw - imbg


class CreateBackground():
    '''
    :class:`pyopia.pipeline` compatible class that calls: :func:`pyopia.background.ini_background`

    adds "bgstack" and "imbg" to the data dict.
    '''

    def __init__(self, bgfiles, load_function):
        self.bgfiles = bgfiles
        self.load_function = load_function
        pass

    def __call__(self, data):
        bgstack, imbg = ini_background(self.bgfiles, self.load_function)

        data['bgstack'] = bgstack
        data['imbg'] = imbg
        return data


class CorrectBackgroundAccurate():
    '''
    :class:`pyopia.pipeline` compatible class that calls: :func:`pyopia.background.correct_im_accurate`
    and will shift the background using a moving average function if given

    Pipeline input data:
    --------------------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.imc`

        :attr:`pyopia.pipeline.Data.bgstack`

        :attr:`pyopia.pipeline.Data.imraw`

        :attr:`pyopia.pipeline.Data.imbg`

    Parameters:
    -----------
    bgshift_function : (function object, optional)
        Function used to shift the background. Defaults to passing (i.e. static background)
        Available functions are:

        :func:`pyopia.background.shift_bgstack_accurate`

        :func:`pyopia.background.shift_bgstack_fast`

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.imc`

        :attr:`pyopia.pipeline.Data.bgstack`

        :attr:`pyopia.pipeline.Data.imbg`
    '''

    def __init__(self, bgshift_function=pass_bgstack):
        self.bgshift_function = bgshift_function
        pass

    def __call__(self, data):
        data['imc'] = correct_im_accurate(data['imbg'], data['imraw'])

        data['bgstack'], data['imbg'] = self.bgshift_function(data['bgstack'],
                                                              data['imbg'],
                                                              data['imraw'])
        return data
