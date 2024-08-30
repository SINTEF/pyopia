# -*- coding: utf-8 -*-
'''
Module containing tools for processing particle image data
'''
import os
import numpy as np
from skimage import morphology
from skimage import segmentation
from skimage import measure
import pandas as pd
from scipy import ndimage as ndi
import skimage.exposure
import h5py
from skimage.io import imsave
from datetime import datetime
import pyopia.statistics

import logging
logger = logging.getLogger()


def image2blackwhite_accurate(input_image, greythresh):
    ''' converts corrected image (im_corrected) to a binary image
    using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

    Args:
        input_image (float)         : image. Usually a background-corrected image
        greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image
                                      histogram)

    Returns:
        imbw                        : segmented image (binary image)

    '''

    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = greythresh * np.percentile(input_image, 50)

    # create a segmented image using the crude threshold
    imbw1 = input_image < thresh

    # perform an adaptive historgram equalization to handle some
    # less-than-ideal lighting situations
    img_adapteq = skimage.exposure.equalize_adapthist(input_image,
                                                      clip_limit=(greythresh),
                                                      nbins=256)

    # use the equalised image to estimate a second semi-automated threshold
    newthresh = np.percentile(img_adapteq, 0.75) * greythresh

    # create a second segmented image using newthresh
    imbw2 = img_adapteq < newthresh

    # merge both segmentation methods by selecting regions where both identify
    # something as a particle (everything else is water)
    imbw = imbw1 & imbw2

    return imbw


def image2blackwhite_fast(input_image, greythresh):
    ''' converts an image (input_image) to a binary image
    using greythresh as the threshold value (fixed scaling of greythresh is done inside)

    Args:
        input_image (float)         : image. Usually a background-corrected image
        greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image
                                      histogram)

    Returns:
        imbw                        : segmented image (binary image)
    '''
    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = greythresh * np.percentile(input_image, 50)
    imbw = input_image < thresh  # segment the image

    return imbw


def clean_bw(imbw, minimum_area):
    ''' cleans up particles which are too small and particles touching the
    border

    Args:
        imbw                        : segmented image
        minimum_area                : minimum number of accepted pixels for a particle

    Returns:
        imbw (DataFrame)           : cleaned up segmented image

    '''

    # remove objects that are below the detection limit defined in the config
    # file.
    # this min_area is usually 12 pixels
    imbw_clean = morphology.remove_small_objects(imbw > 0, min_size=minimum_area)

    # remove particles touching the border of the image
    # because there might be part of a particle not recorded, and therefore
    # border particles will be incorrectly sized
    imbw_clean = segmentation.clear_border(imbw_clean, buffer_size=2)

    # remove objects smaller the min_area
    return imbw_clean


def concentration_check(imbw, max_coverage=30):
    ''' Check saturation level of the sample volume by comparing area of
    particles with settings.Process.max_coverage

    Args:
        imbw                        : segmented image
        max_coverage                : percentage of iamge allowed to be black

    Returns:
        sat_check                   : boolean on if the saturation is acceptable. True if the image is acceptable
        saturation                  : percentage of maximum acceptable saturation defined in
                                      settings.Process.max_coverage
    '''

    # calcualte the area covered by particles in the binary image
    covered_area = float(imbw.sum())

    # measure the image size for correct area calcaultion
    r, c = np.shape(imbw)

    # calculate the percentage of the image covered by particles
    covered_pcent = covered_area / (r * c) * 100

    # convert the percentage covered to a saturation based on the maximum
    # acceptable coverage defined in the config
    saturation = covered_pcent / max_coverage * 100

    logger.info(f'{saturation:.1f}% saturation')

    # check if the saturation is acceptable
    sat_check = saturation < 100

    return sat_check, saturation


def get_spine_length(imbw):
    ''' extracts the spine length of particles from a binary particle image
    (imbw is a binary roi)

    Args:
        imbw                : segmented particle ROI (assumes only one particle)

    Returns:
        spine_length        : spine length of particle (in pixels)
    '''
    skel = morphology.skeletonize(imbw)
    for i in range(2):
        skel = morphology.binary_dilation(skel)
    skel = morphology.skeletonize(skel)

    spine_length = np.sum(skel)
    return spine_length


def extract_roi(input_image, bbox):
    ''' given an image (im) and bounding box (bbox), this will return the roi

    Args:
        input_image         : any image, such as background-corrected image
        bbox                : bounding box from regionprops [r1, c1, r2, c2]

    Returns:
        roi                 : image cropped to region of interest
    '''
    # refer to skimage regionprops documentation on how bbox is structured
    roi = input_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    return roi


def write_segmented_images(imbw, input_image, settings, timestamp):
    '''writes binary images as bmp files to the same place as hdf5 files if loglevel is in DEBUG mode
    Useful for checking threshold and segmentation

    Args:
        imbw                        : segmented image
        settings                    : PySilCam settings
        timestamp                   : timestamp of image collection
    '''
    if (settings.General.loglevel == 'DEBUG') and settings.ExportParticles.export_images:
        fname = os.path.join(settings.ExportParticles.outputpath, timestamp.strftime('D%Y%m%dT%H%M%S.%f-SEG.bmp'))
        imbw_ = np.uint8(255 * imbw)
        imsave(fname, imbw_)
        fname = os.path.join(settings.ExportParticles.outputpath, timestamp.strftime('D%Y%m%dT%H%M%S.%f-IMC.bmp'))
        imsave(fname, input_image)


def put_roi_in_h5(export_outputpath, HDF5File, roi, filename, i):
    '''Adds rois to an open hdf file if export_outputpath is not None.
    For use within {func}`pyopia.process.export_particles`

    Parameters
    ----------
    export_outputpath : str
    HDF5File : h5 file object
    roi : uint8
    i : int
        particle number

    Returns
    -------
    str
        filename
    '''
    filename = filename + '-PN' + str(i)
    if export_outputpath is not None:
        HDF5File.create_dataset('PN' + str(i), data=roi)
    return filename


def extract_particles(imc, timestamp, Classification, region_properties,
                      export_outputpath=None, min_length=0, propnames=['major_axis_length', 'minor_axis_length',
                                                                       'equivalent_diameter']):
    '''extracts the particles to build stats and export particle rois to HDF5 files

    Args:
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        Classification              : initialised classification class from pyiopia.classify
        region_properties           : region properties object returned from regionprops (measure.regionprops(iml,
                                                                                                           cache=False))
        export_outputpath           : path for writing h5 output files. Defaults to None, which switches off file writing
        min_length                  : specifies minimum particle length in pixels to include
        propnames                   : specifies list of skimage regionprops to export to the output file - must contain
                                                                                    default values that can be appended to

    Returns:
        stats                       : (list of particle statistics for every particle, according to Partstats class)
    '''
    filenames = ['not_exported'] * len(region_properties)

    if Classification is not None:
        # pre-allocation
        predictions = np.zeros((len(region_properties),
                                len(Classification.class_labels)),
                               dtype='float64')
        predictions *= np.nan

    # obtain the original image filename from the timestamp
    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')

    if export_outputpath is not None:
        # check path exists and create if not
        isExist = os.path.exists(export_outputpath)
        if not isExist:
            os.makedirs(export_outputpath)
            logger.info(f'Export folder {export_outputpath} created.')

        # Make the HDF5 file
        hdf_filename = os.path.join(export_outputpath, filename + ".h5")
        HDF5File = h5py.File(hdf_filename, "w")
        # metadata
        meta = HDF5File.create_group('Meta')
        meta.attrs['Modified'] = str(datetime.now())
        meta.attrs['Timestamp'] = str(timestamp)
        meta.attrs['Raw image name'] = filename
        # @todo include more useful information in this meta data, e.g. possibly raw image location and background
        #  stack file list.
    else:
        HDF5File = None

    # pre-allocate some things
    data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)
    bboxes = np.zeros((len(region_properties), 4), dtype=np.float64)
    nb_extractable_part = 0

    for i, el in enumerate(region_properties):
        data[i, :] = [getattr(el, p) for p in propnames]
        bboxes[i, :] = el.bbox

        # Find particles that match export criteria
        # major_axis_length in pixels
        # minor length in pixels
        if ((data[i, 0] > min_length) & (data[i, 1] > 2)):

            nb_extractable_part += 1
            # extract the region of interest from the corrected colour image
            roi = extract_roi(imc, bboxes[i, :].astype(int))

            if Classification is not None:
                # run a prediction on what type of particle this might be
                prediction = Classification.proc_predict(roi)
                predictions[int(i), :] = prediction

            # add the roi to the HDF5 file
            filenames[int(i)] = put_roi_in_h5(export_outputpath, HDF5File, roi, filename, i)

    if export_outputpath is not None:
        # close the HDF5 file
        HDF5File.close()

    # build the column names for the outputed DataFrame
    column_names = np.hstack(([propnames, 'minr', 'minc', 'maxr', 'maxc']))

    # merge regionprops statistics with a seperate bounding box columns
    cat_data = np.hstack((data, bboxes))

    # put particle statistics into a DataFrame
    stats = pd.DataFrame(columns=column_names, data=cat_data)

    logger.info(f'EXTRACTING {nb_extractable_part} IMAGES from {len(stats["major_axis_length"])}')

    if Classification is not None:
        # add classification predictions to the particle statistics data
        for n, c in enumerate(Classification.class_labels):
            stats['probability_' + c] = predictions[:, n]

    # add the filenames of the HDF5 file and particle number tag to the
    # particle statistics data
    stats['export name'] = pd.Series(index=stats.index, data=filenames, dtype=str)

    return stats


def measure_particles(imbw, max_particles=5000):
    '''Measures properties of particles

    Args:
      imbw (full-frame binary image)
      max_particles

    Returns:
      region_properties

    '''
    # label the segmented image
    iml = morphology.label(imbw > 0)
    logger.info(f'  {iml.max()} particles found')

    # if there are too many particles then do no proceed with analysis
    if (iml.max() > max_particles):
        raise RuntimeError('Too many particles. Refer to documentation on max_particles parameter in measure_particles()')
        # @todo handle situation when too many particles are found

    region_properties = measure.regionprops(iml, cache=False)

    return region_properties


def segment(img, threshold=0.98, minimum_area=12, fill_holes=True):
    '''Create a binary image from a background-corrected image.

    Parameters
    ----------
    img : np.array
        background-corrected image
    threshold : float, optional
        segmentation threshold, by default 0.98
    minimum_area : int, optional
        minimum number of pixels to be considered a particle, by default 12
    fill_holes : bool, optional
        runs ndi.binary_fill_holes if True, by default True

    Returns
    -------
    imbw : np.array
        segmented image
    '''
    logger.info('segment')

    imbw = image2blackwhite_fast(img, threshold)

    logger.info('clean')

    # clean segmented image (small particles and border particles)
    imbw = clean_bw(imbw, minimum_area)

    if fill_holes:
        # fill holes in particles
        imbw = ndi.binary_fill_holes(imbw)

    imbw = imbw > 0
    return imbw


def statextract(imbw, timestamp, imc,
                Classification=None,
                max_coverage=30,
                max_particles=5000,
                export_outputpath=None,
                min_length=0,
                propnames=['major_axis_length', 'minor_axis_length', 'equivalent_diameter']):
    '''Extracts statistics of particles in a binary images (imbw)

    Args:
        imbw                        : segmented binary image
        img                         : background-corrected image
        timestamp                   : timestamp of image collection
        Classification              : initialised classification class from pyiopia.classify
        max_coverage                : maximum percentge of image that is acceptable as covered by particles.
                                      Image skipped if exceeded.
        max_particles               : maximum number of particles accepted in the image
                                      Image skipped if exceeded.
        region_properties           : region properties object returned from regionprops (measure.regionprops(iml,
                                                                                                           cache=False))
        export_outputpath           : path for writing h5 output files. Defaults to None, which switches off file writing
        min_length                  : specifies minimum particle length in pixels to include
        propnames                   : specifies list of skimage regionprops to export to the output file - must contain
                                                                                    default values that can be appended to

    Returns:
        stats                       : pandas DataFrame of particle statistics for every particle
        saturation                  : percentage saturation of image
    '''

    # check the converage of the image of particles is acceptable
    sat_check, saturation = concentration_check(imbw, max_coverage=max_coverage)
    if (sat_check is False):
        logger.warning('{timestamp}: Breached concentration limit! Skipping image. \
                       This can affect the accuary of concentration calculations!')
        imbw[:] = False  # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    logger.info('measure')
    # calculate particle statistics
    region_properties = measure_particles(imbw, max_particles=max_particles)

    # build the stats and export to HDF5
    if imc.ndim == 2:
        imc = np.stack([imc] * 3, axis=2)
        print('WARNING! Unexpected image dimension. extract_particles modified for 2-d images without color!')

    stats = extract_particles(imc, timestamp, Classification, region_properties,
                              export_outputpath=export_outputpath, min_length=min_length,
                              propnames=propnames)

    return stats, saturation


class Segment():
    '''PyOpia pipline-compatible class for calling segment

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.im_corrected`

    Parameters:
    ----------
    minimum_area : (int, optional)
        minimum number of pixels for particle detection. Defaults to 12.
    threshold : (float, optional)
        threshold for segmentation. Defaults to 0.98.
    fill_holes : (bool)
        runs ndi.binary_fill_holes if True. Defaults to True.
    segment_source: (str, optional)
        The key in Pipeline.data of the image to be segmented.
        Defaults to 'im_corrected'

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.imbw`
    '''

    def __init__(self,
                 minimum_area=12,
                 threshold=0.98,
                 fill_holes=True,
                 segment_source='im_corrected'):

        self.minimum_area = minimum_area
        self.threshold = threshold
        self.fill_holes = fill_holes
        self.segment_source = segment_source

    def __call__(self, data):
        data['imbw'] = segment(data[self.segment_source], threshold=self.threshold, fill_holes=self.fill_holes,
                               minimum_area=self.minimum_area)
        return data


class CalculateStats():
    '''PyOpia pipline-compatible class for calling statextract

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.imbw`

        :attr:`pyopia.pipeline.Data.timestamp`

        :attr:`pyopia.pipeline.Data.cl`

    Parameters:
    ----------
    max_coverage : (int, optional)
        percentage of the image that is allowed to be filled by particles. Defaults to 30.
    max_particles : (int, optional)
        maximum allowed number of particles in an image.
        Exceeding this will discard the image from analysis. Defaults to 5000.
    export_outputpath: (str, optional)
        Path to folder to put extracted particle ROIs (in h5 files).
        Required for making montages later.
    min_length: (int, optional)
        The minimum length of particles (in pixels) to includ in output ROIs
    propnames: (list, optional)
        Specifies properties wanted from skimage.regionprops.
        Defaults to ['major_axis_length', 'minor_axis_length', 'equivalent_diameter']
    roi_source: (str, optional)
        Key of an image in Pipeline.data that is used for outputting ROIs and passing to the classifier.
        Defaults to 'im_corrected'

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.stats`
    '''
    def __init__(self,
                 max_coverage=30,
                 max_particles=5000,
                 export_outputpath=None,
                 min_length=0,
                 propnames=['major_axis_length', 'minor_axis_length', 'equivalent_diameter'],
                 roi_source='im_corrected'):

        self.max_coverage = max_coverage
        self.max_particles = max_particles
        self.export_outputpath = export_outputpath
        self.min_length = min_length
        self.propnames = propnames
        self.roi_source = roi_source

        self.calc_image_stats = CalculateImageStats()

    def __call__(self, data):
        logger.info('statextract')
        stats, saturation = statextract(data['imbw'], data['timestamp'], data[self.roi_source],
                                        Classification=data['cl'],
                                        max_coverage=self.max_coverage,
                                        max_particles=self.max_particles,
                                        export_outputpath=self.export_outputpath,
                                        min_length=self.min_length,
                                        propnames=self.propnames)
        stats['timestamp'] = data['timestamp']
        stats['saturation'] = saturation

        data['stats'] = stats

        self.calc_image_stats(data)

        return data


class CalculateImageStats():
    '''PyOpia pipline-compatible class for collecting whole-image statistics

    Pipeline input data:
    ---------
    :class:`pyopia.pipeline.Data`

        containing the following keys:

        :attr:`pyopia.pipeline.Data.stats`

        :attr:`pyopia.pipeline.Data.timestamp`

    Parameters:
    ----------
    None

    Returns:
    --------
    :class:`pyopia.pipeline.Data`
        containing the following new keys:

        :attr:`pyopia.pipeline.Data.image_stats`
    '''
    def __init__(self):
        pass

    def __call__(self, data):
        logger.info('CalculateImageStats')

        if 'image_stats' not in data:
            data['image_stats'] = pd.DataFrame(columns=['filename', 'particle_count', 'saturation',
                                                        'd50', 'nc', 'vc', 'sample_volume', 'junge'])
            data['image_stats'] = data['image_stats'].astype({'particle_count': np.int64, 'saturation': np.float64})
            data['image_stats'].index.name = 'datetime'

        stats = data['stats']

        # Add image "global" statistics, separate from the particle stats above (stats)
        image_saturation = np.nan if stats.empty else stats['saturation'].values[0]
        data['image_stats'].loc[data['timestamp'], 'filename'] = getattr(data, 'filename', '')
        data['image_stats'].loc[data['timestamp'], 'particle_count'] = int(stats.shape[0])
        data['image_stats'].loc[data['timestamp'], 'saturation'] = image_saturation

        # Skip remaining calculations if no particles where found
        if data['stats'].size == 0:
            return data

        # Calculate D50, nc and vc stats
        pixel_size = data['settings']['general']['pixel_size']

        d50 = pyopia.statistics.d50_from_stats(data['stats'], pixel_size)
        data['image_stats'].loc[data['timestamp'], 'd50'] = d50

        path_length = getattr(data['settings']['general'], 'path_length', 40)
        if path_length is not None:
            nc_vc = pyopia.statistics.nc_vc_from_stats(data['stats'], pixel_size, path_length)
            for k, v in zip(['nc', 'vc', 'sample_volume', 'junge'], nc_vc):
                data['image_stats'].loc[data['timestamp'], k] = v

        return data
