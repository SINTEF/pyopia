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

    There is a small chance of clipping of imc in both crushed blacks and blown
    highlights if the background or raw images are very poorly obtained

    Args:
      imbg (uint8 or float64)  : background averaged image
      imraw (uint8 or float64) : raw image

    Returns:
      imc (uint8 or float64)   : corrected image, same type as input
    '''

    imc = np.float64(imraw) - np.float64(imbg)
    if imc.ndim == 3:
        imc[:, :, 0] += (255 / 2 - np.percentile(imc[:, :, 0], 50))
        imc[:, :, 1] += (255 / 2 - np.percentile(imc[:, :, 1], 50))
        imc[:, :, 2] += (255 / 2 - np.percentile(imc[:, :, 2], 50))
    else:
        imc += (255 / 2 - np.percentile(imc, 50))

    imc += 255 - imc.max()

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


class CorrectBackgroundAccurate():
    '''
    :class:`pyopia.pipeline` compatible class that calls: :func:`pyopia.background.correct_im_accurate`
    and will shift the background using a moving average function if given.

    Pipeline input data:
    --------------------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.bgstack`

        :attr:`pyopia.pipeline.Data.imraw`

        :attr:`pyopia.pipeline.Data.imbg`

    Parameters:
    -----------
    bgshift_function : (string, optional)
        Function used to shift the background. Defaults to passing (i.e. static background)
        Available options are 'accurate', 'fast', or 'pass' to apply a statick background correction:

        :func:`pyopia.background.shift_bgstack_accurate`

        :func:`pyopia.background.shift_bgstack_fast`

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.imc`

        :attr:`pyopia.pipeline.Data.bgstack`

        :attr:`pyopia.pipeline.Data.imbg`


    Example pipeline uses:
    ----------------------
    Apply moving average using :func:`pyopia.background.shift_bgstack_accurate` :

    .. code-block:: python

        [steps.correctbackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundAccurate'
        bgshift_function = 'accurate'

    Apply static background correction:

    .. code-block:: python

        [steps.correctbackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundAccurate'
        bgshift_function = 'pass'


    If you do not want to do background correction, leave this step out of the pipeline.
    Then you could use :class:`pyopia.pipeline.CorrectBackgroundNone` if you need to instead.
    '''

    def __init__(self, bgshift_function='pass', average_window=1):
        self.bgshift_function = bgshift_function
        self.average_window = average_window

        # Initialize data structures for background image and stack
        self.background_is_initialized = False


    def _build_background_step(self, data):
        '''Add one layer to the background stack from the raw image in data pipeline, and update the background image.'''
        if not 'bgstack' in data:
            data['bgstack'] = []

        data['bgstack'].append(data['imraw'])
        data['imbg'] = np.mean(data['bgstack'], axis=0)

        if len(data['bgstack']) == self.average_window:
            self.background_is_initialized = True


    def __call__(self, data):
        # Initialize background and skip correction step while required bgstack size not reached
        if not self.background_is_initialized:
            self._build_background_step(data)

        data['imc'] = correct_im_accurate(data['imbg'], data['imraw'])

        match self.bgshift_function:
            case 'pass':
                return data
            case 'accurate':
                data['bgstack'], data['imbg'] = shift_bgstack_accurate(data['bgstack'],
                                                                       data['imbg'],
                                                                       data['imraw'])
            case 'fast':
                data['bgstack'], data['imbg'] = shift_bgstack_fast(data['bgstack'],
                                                                   data['imbg'],
                                                                   data['imraw'])
        return data


class CorrectBackgroundStatic():
    '''
    :class:`pyopia.pipeline` compatible class that calls: :func:`pyopia.background.correct_im_accurate`
    and will use a static background for correction.

    Pipeline input data:
    --------------------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.imraw`

        :attr:`pyopia.pipeline.Data.imbg`

    Parameters:
    -----------
    background_image : (string)

        Full path to background image
        
    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.imc`

        :attr:`pyopia.pipeline.Data.bgstack`

        :attr:`pyopia.pipeline.Data.imbg`


    Example pipeline uses:
    ----------------------
    Apply moving average using :func:`pyopia.background.shift_bgstack_accurate` :

    .. code-block:: python

        [steps.correctbackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundAccurate'
        bgshift_function = 'accurate'

    Apply static background correction:

    .. code-block:: python

        [steps.correctbackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundAccurate'
        bgshift_function = 'pass'


    If you do not want to do background correction, leave this step out of the pipeline.
    Then you could use :class:`pyopia.pipeline.CorrectBackgroundNone` if you need to instead.
    '''

    def __init__(self, bgshift_function='pass', average_window=1):
        self.bgshift_function = bgshift_function
        self.average_window = average_window

        # Initialize data structures for background image and stack
        self.background_is_initialized = False

    def __call__(self, data):
        # Initialize background and skip correction step while required bgstack size not reached
        if not self.background_is_initialized:
            if not 'bgstack' in data:
                data['bgstack'] = []

            data['bgstack'].append(data['imraw'])
            data['imbg'] = np.mean(data['bgstack'], axis=0)

            if len(data['bgstack']) == self.average_window:
                self.background_is_initialized = True

        data['imc'] = correct_im_accurate(data['imbg'], data['imraw'])

        match self.bgshift_function:
            case 'pass':
                return data
            case 'accurate':
                data['bgstack'], data['imbg'] = shift_bgstack_accurate(data['bgstack'],
                                                                       data['imbg'],
                                                                       data['imraw'])
            case 'fast':
                data['bgstack'], data['imbg'] = shift_bgstack_fast(data['bgstack'],
                                                                   data['imbg'],
                                                                   data['imraw'])
        return data



class CorrectBackgroundNone():
    '''
    :class:`pyopia.pipeline` compatible class for use when no background correction is required.
    This simply makes `data['imc'] = data['imraw'] in the pipeline.

    Pipeline input data:
    --------------------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.imraw`

    Parameters:
    -----------
    none

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.imc`


    Example pipeline uses:
    ----------------------
    Don't apply any background correction after image load step :

    .. code-block:: python

        [steps.nobackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundNone'

    '''

    def __init__(self):
        pass

    def __call__(self, data):
        data['imc'] = data['imraw']

        return data
