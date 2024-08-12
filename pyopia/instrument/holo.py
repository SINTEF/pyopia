'''
Module containing hologram specific tools to enable compatability with the :mod:`pyopia.pipeline`
'''

import os
import numpy as np
import pandas as pd
from scipy import fft
from skimage.io import imread
from skimage.filters import sobel
from skimage.morphology import disk, erosion, dilation
import pyopia.process
import struct
from datetime import timedelta, datetime
from glob import glob

'''
This is a module containing basic processing for reconstruction of in-line holographic images.

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
    wavelength : float
        laser wavelength in nm
    n : float
        refractive index of medium
    offset : float
        offset of focal plane from hologram plane in mm
    minZ : float
        minimum reconstruction distance in mm
    maxZ : float
        maximum reconstruction distance in mm
    stepZ : float
        step size in mm (i.e. resolution of reconstruction between minZ and maxZ)

    Returns
    -------
    kern : np.arry
        reconstruction kernel
    im_stack : np.array
        pre-allocated array to receive reconstruction

    '''

    def __init__(self, wavelength, n, offset, minZ, maxZ, stepZ):
        self.wavelength = wavelength
        self.n = n
        self.offset = offset
        self.minZ = minZ
        self.maxZ = maxZ
        self.stepZ = stepZ

    def __call__(self, data):
        print('Using first raw file from list in general settings to determine image dimensions')
        raw_files = glob(data['settings']['general']['raw_files'])
        self.filename = raw_files[0]
        imtmp = load_image(self.filename)
        self.pixel_size = data['settings']['general']['pixel_size']
        print('Build kernel with pixel_size = ', self.pixel_size, 'um')
        kern = create_kernel(imtmp, self.pixel_size, self.wavelength, self.n, self.offset, self.minZ, self.maxZ, self.stepZ)
        im_stack = np.zeros(np.shape(kern)).astype(np.float64)
        print('HoloInitial done', datetime.now())
        data['kern'] = kern
        data['im_stack'] = im_stack
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
    img = imread(filename).astype(np.float64)/255
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
        print(data['filename'])
        try:
            timestamp = read_lisst_holo_info(data['filename'])
        except ValueError:
            timestamp = pd.to_datetime(os.path.splitext(os.path.basename(data['filename']))[0][1:])
        print(timestamp)
        im = load_image(data['filename'])
        data['timestamp'] = timestamp
        data['imraw'] = im
        return data


class Reconstruct():
    '''PyOpia pipline-compatible class for reconstructing a single holo image

    Parameters
    ----------
    stack_clean : float
        defines amount of cleaning of stack (fraction of max value below which to zero)
    forward_filter_option : int
        switch to control filtering in frequency domain (0=none,1=DC only,2=zero ferquency/default)
    inverse_output_option :  int
        switch to control optional scaling of output intensity (0=square/default,1=linear)

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`
        containing the following keys:

        :attr:`pyopia.pipeline.Data.imc`

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.im_stack`
    '''

    def __init__(self, stack_clean=0, forward_filter_option=0, inverse_output_option=0):
        self.stack_clean = stack_clean
        self.forward_filter_option = forward_filter_option
        self.inverse_output_option = inverse_output_option

    def __call__(self, data):
        imc = data['imc']
        kern = data['kern']
        im_stack = data['im_stack']

        im_fft = forward_transform(imc, self.forward_filter_option)
        im_stack = inverse_transform(im_fft, kern, im_stack, self.inverse_output_option)
        data['im_stack'] = clean_stack(im_stack, self.stack_clean)

        return data


def forward_transform(im, forward_filter_option=2):
    '''Perform forward transform with optional filtering

    Parameters
    ----------
    im : np.array
        hologram (usually background-corrected)
    forward_filter_option : int
        filtering in frequency domain (0=none/default,1=DC only,2=zero ferquency)

    Returns
    -------
    im_fft : np.array
        im_fft
    '''

    # Perform forward transform
    im_fft = fft.fft2(im, workers=os.cpu_count())

    # apply filtering if required
    match forward_filter_option:
        case 1:
            im_fft[0, 0] = 0
        case 2:
            im_fft[:, 0] = 0
            im_fft[0, :] = 0
        case _:
            pass

    # fftshift
    im_fft = fft.fftshift(im_fft)

    return im_fft


def create_kernel(im, pixel_size, wavelength, n, offset, minZ, maxZ, stepZ):
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

    z = (np.arange(minZ * 1e-3, (maxZ + stepZ) * 1e-3, stepZ * 1e-3) / n) + (offset * 1e-3)

    wavelength_m = wavelength * 1e-9
    k = 2 * np.pi / wavelength_m

    kern = -1j * np.zeros((im.shape[0], im.shape[1], len(z)))
    for i, z_ in enumerate(z):

        kern[:, :, i] = np.exp(-1j * f**2 * z_ / (2 * k))
    return kern


def inverse_transform(im_fft, kern, im_stack, inverse_output_option=0):
    '''create the reconstructed hologram stack of real images

    Parameters
    ----------
    im_fft : np.array
        calculated from forward_transform
    kern : np.array
        calculated from create_kernel
    im_stack : np.array
        pre-allocated array to receive output
    inverse_output_option: int
        optional scaling of output intensity (0=square/default,1=linear)

    Returns
    -------
    np.arry
        im_stack
    '''

    for i in range(np.shape(kern)[2]):
        im_tmp = np.multiply(im_fft, kern[:, :, i])
        match inverse_output_option:
            case 1:
                im_stack[:, :, i] = fft.ifft2(im_tmp, workers=os.cpu_count()).real
            case _:
                im_stack[:, :, i] = (fft.ifft2(im_tmp, workers=os.cpu_count()).real)**2

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
    im_stack_inverted = (im_stack - im_min) / (im_max - im_min)
    im_stack_inverted = 1 - im_stack_inverted
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
    im = (im - im_min) / (im_max - im_min)
    im = 1 - im
    return im


def find_focus_imax(im_stack, bbox, increase_depth_of_field):
    '''finds and returns the focussed image for the bbox region within im_stack
    using intensity of bbox area

    Parameters
    ----------
    im_stack : nparray
        image stack

    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col)

    increase_depth_of_field : bool
        set to True to use max values from planes either side of main focus plane to create focussed image (default False)

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

    if increase_depth_of_field:
        im_focus = np.max(im_seg[:, :, np.max([ifocus-1, 0]):np.min([ifocus+1, im_seg.shape[2]])], axis=2)
    else:
        im_focus = im_seg[:, :, ifocus]

    return im_focus, ifocus


def find_focus_sobel(im_stack, bbox, increase_depth_of_field):
    '''finds and returns the focussed image for the bbox region within im_stack
    using edge magnitude of bbox area

    Parameters
    ----------
    im_stack : nparray
        image stack

    bbox : tuple
        Bounding box (min_row, min_col, max_row, max_col)

    increase_depth_of_field : bool
        set to True to use max values from planes either side of main focus plane to create focussed image (default False)

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

    if increase_depth_of_field:
        im_focus = np.max(im_seg[:, :, np.max([ifocus-1, 0]):np.min([ifocus+1, im_seg.shape[2]])], axis=2)
    else:
        im_focus = im_seg[:, :, ifocus]

    return im_focus, ifocus


class Focus():
    '''PyOpia pipline-compatible class for creating a focussed image from an image stack

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.im_stack`

    Parameters
    ----------
    stacksummary_function : (string, optional)
        Function used to summarise the stack
        Available functions are:

        :func:`pyopia.instrument.holo.max_map`

        :func:`pyopia.instrument.holo.std_map` (default)

    threshold : float
        threshold to apply during initial segmentation

    focus_function : (string, optional)
        Function used to focus particles within the stack
        Available functions are:

        :func:`pyopia.instrument.holo.find_focus_imax` (default)

        :func:`pyopia.instrument.holo.find_focus_sobel`

    discard_end_slices : (bool, optional)
        set to True to discard particles that focus at either first or last slice

    increase_depth_of_field : (bool, optional)
        set to True to use max values from planes either side of main focus plane to create focussed image (default False)

    merge_adjacent_particles : (bool, optional)
        set to 0 (default) to deactivate, set to positive integer to give radius in pixels of smoothing of stack
        summary image to merge adjacent particles

    Returns
    -------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.imc`

        :attr:`pyopia.pipeline.Data.imss`

        :attr:`pyopia.pipeline.Data.stack_rp`

        :attr:`pyopia.pipeline.Data.stack_ifocus`
    '''

    def __init__(self, stacksummary_function='std_map', threshold=0.9, focus_function='find_focus_imax',
                 discard_end_slices=True, increase_depth_of_field=False, merge_adjacent_particles=0):
        self.stacksummary_function = stacksummary_function
        self.threshold = threshold
        self.focus_function = focus_function
        self.discard_end_slices = discard_end_slices
        self.increase_depth_of_field = increase_depth_of_field
        self.merge_adjacent_particles = merge_adjacent_particles

    def __call__(self, data):
        im_stack = data['im_stack']

        match self.stacksummary_function:
            case 'std_map':
                imss = std_map(im_stack)
            case 'max_map':
                imss = max_map(im_stack)
            case _:
                raise ValueError('stacksummary_function in pyopia.instrument.holo.Focus not recognised')

        imss = rescale_image(imss)
        if self.merge_adjacent_particles:
            se = disk(self.merge_adjacent_particles)
            imss = dilation(imss, se)
            imss = erosion(imss, se)
        data['imss'] = imss

        # segment imss to find particle x-y locations
        imssbw = pyopia.process.segment(imss, self.threshold)
        # identify particles
        region_properties = pyopia.process.measure_particles(imssbw)
        # loop through bounding boxes to focus each particle and add to output imc
        imc = np.zeros_like(im_stack[:, :, 0])
        ifocus = []
        rp_out = []
        for rp in region_properties:

            match self.focus_function:
                case 'find_focus_imax':
                    focus_result = find_focus_imax(im_stack, rp.bbox, self.increase_depth_of_field)
                case 'find_focus_sobel':
                    focus_result = find_focus_sobel(im_stack, rp.bbox, self.increase_depth_of_field)
                case _:
                    raise ValueError('focus_function in pyopia.instrument.holo.Focus not recognised')

            if self.discard_end_slices and (focus_result[1] == 0 or focus_result[1] == im_stack.shape[2]):
                continue
            im_focus = 1 - focus_result[0]
            ifocus.append(focus_result[1])
            rp_out.append(rp)
            imc[rp.bbox[0]:rp.bbox[2], rp.bbox[1]:rp.bbox[3]] = im_focus

        data['imc'] = imc
        data['stack_rp'] = rp_out
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
        stats['holo_filename'] = data['filename']
        data['stats'] = stats
        return data


def read_lisst_holo_info(filename):
    '''reads the non-image information (timestamp, etc) from LISST-HOLO holograms

    Parameters
    ----------
    filename : string
        filename to load

    Returns
    -------
    timestamp : timestamp
        timestamp
    '''
    f = open(filename, 'rb')
    assert f.readline().decode('ascii').strip() == 'P5'
    (width, height, bitdepth) = [int(i) for i in f.readline().split()]
    assert bitdepth <= 255
    f.seek(width * height, 1)

    timestamp = (pd.to_datetime(struct.unpack('i', f.read(4)), unit='s'))
    filenum = filename.rsplit('-', 1)[-1]
    filenum = int(filenum.rsplit('.', 1)[0])
    timestamp = timestamp + timedelta(microseconds=filenum)
    timestamp = timestamp[0]
    print(timestamp.strftime('D%Y%m%dT%H%M%S.%f'))
    f.close()

    return timestamp


def generate_config(raw_files: str, model_path: str, outfolder: str, output_prefix: str):
    '''Generaste example holo config.toml as a dict

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

    Returns:
    --------
    dict
        pipeline_config toml dict
    '''
    pipeline_config = {
        'general': {
            'raw_files': raw_files,
            'pixel_size': 4.4  # pixel size in um
        },
        'steps': {
            'initial': {
                'pipeline_class': 'pyopia.instrument.holo.Initial',
                'wavelength': 658,  # laser wavelength in nm
                'n': 1.33,  # index of refraction of sample volume medium (1.33 for water)
                'offset': 27,  # offset to start of sample volume in mm
                'minZ': 0,  # minimum reconstruction distance within sample volume in mm
                'maxZ': 50,  # maximum reconstruction distance within sample volume in mm
                'stepZ': 0.5  # step size in mm
            },
            'classifier': {
                'pipeline_class': 'pyopia.classify.Classify',
                'model_path': model_path
            },
            'createbackground': {
                'pipeline_class': 'pyopia.background.CreateBackground',
                'average_window': 10,
                'instrument_module': 'holo'
            },
            'load': {
                'pipeline_class': 'pyopia.instrument.holo.Load'
            },
            'correctbackground': {
                'pipeline_class': 'pyopia.background.CorrectBackgroundAccurate',
                'bgshift_function': 'accurate'
            },
            'reconstruct': {
                'pipeline_class': 'pyopia.instrument.holo.Reconstruct',
                'stack_clean': 0.02,
                'forward_filter_option': 2,
                'inverse_output_option': 0
            },
            'focus': {
                'pipeline_class': 'pyopia.instrument.holo.Focus',
                'stacksummary_function': 'max_map',
                'threshold': 0.9,
                'focus_function': 'find_focus_sobel',
                'increase_depth_of_field': False,
                'merge_adjacent_particles': 2
            },
            'segmentation': {
                'pipeline_class': 'pyopia.process.Segment',
                'threshold': 0.9
            },
            'statextract': {
                'pipeline_class': 'pyopia.process.CalculateStats',
                'export_outputpath': outfolder,
                'propnames': ['major_axis_length', 'minor_axis_length', 'equivalent_diameter',
                              'feret_diameter_max', 'equivalent_diameter_area']
            },
            'mergeholostats': {
                'pipeline_class': 'pyopia.instrument.holo.MergeStats',
            },
            'output': {
                'pipeline_class': 'pyopia.io.StatsH5',
                'output_datafile': os.path.join(outfolder, output_prefix)
            }
        }
    }
    return pipeline_config
