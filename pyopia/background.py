'''
Background correction module (inherited from PySilCam)
'''
import numpy as np


def ini_background(bgfiles, load_function):
    '''
    Create and initial background stack and average image

    Parameters
    -----------
    bgfiles : list
        List of strings of filenames to be used in background creation
    load_function : object
        This function should take a filename and return an image, for example: :func:`pyopia.instrument.silcam.load_image`

    Returns
    -------
    bgstack : list
        list of all images in the background stack
    imbg : array
        background image
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

    Parameters
    -----------
    bgstack : list
        list of all images in the background stack
    imbg : array
        background image
    imnew : array
        new image to be added to stack

    Returns
    -------
    bgstack : list
        updated list of all background images
    imbg : array
        updated actual background image
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

    Parameters
    -----------
    bgstack : list
        list of all images in the background stack
    imbg : uint8
        background image
    imnew : unit8
        new image to be added to stack

    Returns
    -------
    bgstack : list
        updated list of all background images
    imbg : array
        updated actual background image
    '''
    stacklength = len(bgstack)
    imold = bgstack.pop(0)  # pop the oldest image from the stack,
    # subtract the old image from the average (scaled by the average window)
    imbg -= (imold / stacklength)
    # add the new image to the average (scaled by the average window)
    imbg += (imnew / stacklength)
    bgstack.append(imnew)  # append the new image to the stack
    return bgstack, imbg


def correct_im_accurate(imbg, imraw, divide_bg=False):
    '''
    Corrects raw image by subtracting or dividing the background and scaling the output

    For dividing method see: https://doi.org/10.1016/j.marpolbul.2016.11.063)

    There is a small chance of clipping of imc in both crushed blacks and blown
    highlights if the background or raw images are very poorly obtained

    Parameters
    -----------
    imbg : float64
        background averaged image
    imraw : float64
        raw image
    divide_bg : (bool, optional)
        If True, the correction will be performed by dividing the raw image by the background
        Default to False

    Returns
    -------
    im_corrected : float64
        corrected image, same type as input
    '''

    if divide_bg:
        imbg = np.clip(imbg, a_min=1/255, a_max=None)   # Clipping the zero_value pixels
        im_corrected = imraw / imbg
        im_corrected += (1 / 2 - np.percentile(im_corrected, 50))
        im_corrected -= im_corrected.min()   # Shift the negative values to zero
        im_corrected = np.clip(im_corrected, a_min=0, a_max=1)
    else:
        im_corrected = imraw - imbg
        im_corrected += (1 / 2 - np.percentile(im_corrected, 50))
        im_corrected += 1 - im_corrected.max()   # Shift the positive values exceeding unity to one

    return im_corrected


def correct_im_fast(imbg, imraw):
    '''
    Corrects raw image by subtracting the background and clipping the ouput
    without scaling

    There is high potential for clipping of imc in both crushed blacks an blown
    highlights, especially if the background or raw images are not properly obtained

    Parameters
    -----------
    imraw : array
        raw image
    imbg : array
        background averaged image

    Returns
    -------
    im_corrected : array
        corrected image
    '''
    im_corrected = imraw - imbg

    im_corrected += 215/255
    im_corrected = np.clip(im_corrected, 0, 1)

    return im_corrected


def shift_and_correct(bgstack, imbg, imraw, stacklength, real_time_stats=False):
    '''
    Shifts the background stack and averaged image and corrects the new
    raw image.

    This is a wrapper for shift_bgstack and correct_im

    Parameters
    -----------
    bgstack : list
        list of all images in the background stack
    imbg : float64
        background image
    imraw : float64
        raw image
    stacklength : int
        unused int here - just there to maintain the same behaviour as shift_bgstack_fast()
    real_time_stats : Bool, optional
        True use fast functions, if False use accurate functions., by default False

    Returns
    -------
    bgstack : list
        list of all images in the background stack
    imbg : float64
        background averaged image
    im_corrected : float64
        corrected image
    '''

    if real_time_stats:
        im_corrected = correct_im_fast(imbg, imraw)
        bgstack, imbg = shift_bgstack_fast(bgstack, imbg, imraw, stacklength)
    else:
        im_corrected = correct_im_accurate(imbg, imraw)
        bgstack, imbg = shift_bgstack_accurate(bgstack, imbg, imraw, stacklength)

    return bgstack, imbg, im_corrected


class CorrectBackgroundAccurate():
    '''
    :class:`pyopia.pipeline` compatible class that calls: :func:`pyopia.background.correct_im_accurate`
    and will shift the background using a moving average function if given.

    The background stack and background image are created during the first 'average_window' (int) calls
    to this class, and the skip_next_steps flag is set in the pipeline Data. No background correction
    is performed during these steps.

    Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.imraw`

    Parameters
    ----------
    bgshift_function : (string, optional)
        Function used to shift the background. Defaults to passing (i.e. static background)
        Available options are 'accurate', 'fast', or 'pass' to apply a static background correction:

        :func:`pyopia.background.shift_bgstack_accurate`

        :func:`pyopia.background.shift_bgstack_fast`

    average_window : int
        number of images to use in the background image stack

    image_source: (str, optional)
        The key in Pipeline.data of the image to be background corrected.
        Defaults to 'imraw'

    divide_bg : (bool)
        If True, it performs background correction by dividing the raw image by the background.
        Default to False.

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.im_corrected`
        :attr:`pyopia.pipeline.Data.im_corrected`

        :attr:`pyopia.pipeline.Data.bgstack`

        :attr:`pyopia.pipeline.Data.imbg`

    Examples
    --------
    Apply moving average using :func:`pyopia.background.shift_bgstack_accurate` :

    .. code-block:: toml

        [steps.correctbackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundAccurate'
        bgshift_function = 'accurate'
        average_window = 5

    Apply static background correction:

    .. code-block:: toml

        [steps.correctbackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundAccurate'
        bgshift_function = 'pass'
        average_window = 5

    If you do not want to do background correction, leave this step out of the pipeline.
    Then you could use :class:`pyopia.pipeline.CorrectBackgroundNone` if you need to instead.
    '''

    def __init__(self, bgshift_function='pass', average_window=1, image_source='imraw', divide_bg=False):
        self.bgshift_function = bgshift_function
        self.average_window = average_window
        self.image_source = image_source
        self.divide_bg = divide_bg

    def _build_background_step(self, data):
        '''Add one layer to the background stack from the raw image in data pipeline, and update the background image.'''
        if 'bgstack' not in data:
            data['bgstack'] = []

        init_complete = True
        if len(data['bgstack']) < self.average_window:
            data['bgstack'].append(data[self.image_source])
            data['imbg'] = np.mean(data['bgstack'], axis=0)
            init_complete = False

        return init_complete

    def __call__(self, data):
        # Initialize the background while required bgstack size not reached
        init_complete = self._build_background_step(data)

        # If we are still building the bgstack, return without doing image correction and bgstack update
        if not init_complete:
            # Flag to the pipeline that remaining steps should be skipped since we are still building the background
            data['skip_next_steps'] = True
            return data

        data['im_corrected'] = correct_im_accurate(data['imbg'], data[self.image_source], divide_bg=self.divide_bg)

        match self.bgshift_function:
            case 'pass':
                return data
            case 'accurate':
                data['bgstack'], data['imbg'] = shift_bgstack_accurate(data['bgstack'],
                                                                       data['imbg'],
                                                                       data[self.image_source])
            case 'fast':
                data['bgstack'], data['imbg'] = shift_bgstack_fast(data['bgstack'],
                                                                   data['imbg'],
                                                                   data[self.image_source])
        return data


class CorrectBackgroundNone():
    '''
    :class:`pyopia.pipeline` compatible class for use when no background correction is required.
    This simply makes `data['im_corrected'] = data['imraw'] in the pipeline.
    This simply makes `data['im_corrected'] = data['imraw'] in the pipeline.

    Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.imraw`

    Parameters
    -----------
    None

    Returns
    --------
    data : :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.im_corrected`


    Example pipeline uses:
    ----------------------
    Don't apply any background correction after image load step :

    .. code-block:: toml

        [steps.nobackground]
        pipeline_class = 'pyopia.background.CorrectBackgroundNone'

    '''

    def __init__(self):
        pass

    def __call__(self, data):
        data['im_corrected'] = data['imraw']

        return data
