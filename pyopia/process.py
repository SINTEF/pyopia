# -*- coding: utf-8 -*-
import time
import numpy as np
from skimage import morphology
from skimage import segmentation
from skimage import measure
import pandas as pd
from scipy import ndimage as ndi
import skimage.exposure
import h5py
import os
from skimage.io import imsave
import traceback
from datetime import datetime

'''
Module for processing particle image data
'''


def image2blackwhite_accurate(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

    Args:
        imc                         : background-corrected image
        greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image
                                      histogram)

    Returns:
        imbw                        : segmented image (binary image)

    '''
    img = np.copy(imc)  # create a copy of the input image (not sure why)

    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = np.uint8(greythresh * np.percentile(img, 50))

    # create a segmented image using the crude threshold
    imbw1 = img < thresh

    # perform an adaptive historgram equalization to handle some
    # less-than-ideal lighting situations
    img_adapteq = skimage.exposure.equalize_adapthist(img,
                                                      clip_limit=(1 - greythresh),
                                                      nbins=256)

    # use the equalised image to estimate a second semi-automated threshold
    newthresh = np.percentile(img_adapteq, 0.75) * greythresh

    # create a second segmented image using newthresh
    imbw2 = img_adapteq < newthresh

    # merge both segmentation methods by selecting regions where both identify
    # something as a particle (everything else is water)
    imbw = imbw1 & imbw2

    return imbw


def image2blackwhite_fast(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (fixed scaling of greythresh is done inside)

    Args:
        imc                         : background-corrected image
        greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image
                                      histogram)

    Returns:
        imbw                        : segmented image (binary image)
    '''
    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = np.uint8(greythresh * np.percentile(imc, 50))
    imbw = imc < thresh  # segment the image

    return imbw


def clean_bw(imbw, min_area):
    ''' cleans up particles which are too small and particles touching the
    border

    Args:
        imbw                        : segmented image
        min_area                    : minimum number of accepted pixels for a particle

    Returns:
        imbw (DataFrame)           : cleaned up segmented image

    '''

    # remove objects that are below the detection limit defined in the config
    # file.
    # this min_area is usually 12 pixels
    imbw = morphology.remove_small_objects(imbw > 0, min_size=min_area)

    # remove particles touching the border of the image
    # because there might be part of a particle not recorded, and therefore
    # border particles will be incorrectly sized
    imbw = segmentation.clear_border(imbw, buffer_size=2)

    # remove objects smaller the min_area
    return imbw


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

    print('{0:.1f}% saturation'.format(saturation))

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


def extract_roi(im, bbox):
    ''' given an image (im) and bounding box (bbox), this will return the roi

    Args:
        im                  : any image, such as background-corrected image (imc)
        bbox                : bounding box from regionprops [r1, c1, r2, c2]

    Returns:
        roi                 : image cropped to region of interest
    '''
    # refer to skimage regionprops documentation on how bbox is structured
    roi = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    return roi


def write_segmented_images(imbw, imc, settings, timestamp):
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
        imsave(fname, imc)


def extract_particles(imc, timestamp, Classification, region_properties,
                      export_outputpath=None, min_length=0):
    '''extracts the particles to build stats and export particle rois to HDF5 files

    Args:
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        Classification              : initialised classification class from pyiopia.classify
        region_properties           : region properties object returned from regionprops (measure.regionprops(iml,
                                                                                                           cache=False))
        export_outputpath           : path for writing h5 output files. Defaults to None, which switches off file writing
                                        Note: path must exist
    
    Returns:
        stats                       : (list of particle statistics for every particle, according to Partstats class)
    '''
    filenames = ['not_exported'] * len(region_properties)

    # pre-allocation
    predictions = np.zeros((len(region_properties),
                            len(Classification.class_labels)),
                           dtype='float64')
    predictions *= np.nan

    # obtain the original image filename from the timestamp
    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')

    if export_outputpath is not None:
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

    # define the geometrical properties to be calculated from regionprops
    propnames = ['major_axis_length', 'minor_axis_length',
                 'equivalent_diameter', 'solidity']

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

            # add the roi to the HDF5 file
            filenames[int(i)] = filename + '-PN' + str(i)
            if export_outputpath is not None:
                HDF5File.create_dataset('PN' + str(i), data=roi)
                # @todo also include particle stats here too.

            # run a prediction on what type of particle this might be
            prediction = Classification.proc_predict(roi)
            predictions[int(i), :] = prediction[0]

    if export_outputpath is not None:
        # close the HDF5 file
        HDF5File.close()

    # build the column names for the outputed DataFrame
    column_names = np.hstack(([propnames, 'minr', 'minc', 'maxr', 'maxc']))

    # merge regionprops statistics with a seperate bounding box columns
    cat_data = np.hstack((data, bboxes))

    # put particle statistics into a DataFrame
    stats = pd.DataFrame(columns=column_names, data=cat_data)

    print('EXTRACTING {0} IMAGES from {1}'.format(nb_extractable_part, len(stats['major_axis_length'])))

    # add classification predictions to the particle statistics data
    for n, c in enumerate(Classification.class_labels):
        stats['probability_' + c] = predictions[:, n]

    # add the filenames of the HDF5 file and particle number tag to the
    # particle statistics data
    stats['export name'] = filenames

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
    print('  {0} particles found'.format(iml.max()))

    # if there are too many particles then do no proceed with analysis
    if (iml.max() > max_particles):
        print('....that''s way too many particles! Skipping image.')
        imbw *= 0  # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    region_properties = measure.regionprops(iml, cache=False)

    return region_properties


def statextract(imc, timestamp, Classification,
                minimum_area=12, threshold=0.98, real_time_stats=False, max_coverage=30, max_particles=5000,
                extractparticles_function=extract_particles):
    '''extracts statistics of particles in imc (raw corrected image)

    Args:
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        Classification              : initialised classification class from pyiopia.classify
        measure_function            : function for measuring particles that must conform to measure_particles
                                        defaults to measure_particles

    Returns:
        stats                       : (list of particle statistics for every particle, according to Partstats class)
        imbw                        : segmented image
        saturation                  : percentage saturation of image
    '''
    print('segment')

    # simplify processing by squeezing the image dimensions into a 2D array
    # min is used for squeezing to represent the highest attenuation of all wavelengths
    img = np.uint8(np.min(imc, axis=2))

    if real_time_stats:
        imbw = image2blackwhite_fast(img, threshold)  # image2blackwhite_fast is less fancy but
    else:
        imbw = image2blackwhite_accurate(img, threshold)  # image2blackwhite_fast is less fancy but
    # image2blackwhite_fast is faster than image2blackwhite_accurate but might cause problems when trying to
    # process images with bad lighting

    print('clean')

    # clean segmented image (small particles and border particles)
    imbw = clean_bw(imbw, minimum_area)

    # fill holes in particles
    imbw = ndi.binary_fill_holes(imbw)

    # @todo re-implement: write_segmented_images(imbw, imc, settings, timestamp)

    # check the converage of the image of particles is acceptable
    sat_check, saturation = concentration_check(imbw, max_coverage=max_coverage)
    if (sat_check is False):
        print('....breached concentration limit! Skipping image.')
        imbw *= 0  # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    print('measure')
    # calculate particle statistics
    region_properties = measure_particles(imbw, max_particles=max_particles)

    # build the stats and export to HDF5
    stats = extractparticles_function(imc, timestamp, Classification, region_properties)

    return stats, imbw, saturation


def process_image(Classification, data,
                  statextract_function=statextract):
    '''
    Proceses image data into a stats formatted DataFrame

    Args:
        Classification      : initialised classification class from pyiopia.classify
        data (tuple)        : tuple contianing (i, timestamp, imc)
                              where i is an int referring to the image number
                              timestamp is the image timestamp obtained from passing the filename
                              imc is the background-corrected image
        statextract_function    : function for extracting particles that must conform to statextract behaviour
                                defaults to statextract

    Returns:
        stats (DataFrame) :  stats dataframe containing particle statistics
    '''
    try:
        i = data[0]
        timestamp = data[1]
        imc = data[2]

        # time the full acquisition and processing loop
        start_time = time.time()

        print('Processing time stamp {0}'.format(timestamp))

        # Calculate particle statistics
        stats, imbw, saturation = statextract_function(imc, timestamp, Classification)

        # if there are not particles identified, assume zero concentration.
        # This means that the data should indicate that a 'good' image was
        # obtained, without any particles. Therefore fill all values with nans
        # and add the image timestamp
        if len(stats) == 0:
            print('ZERO particles identified')
            z = np.zeros(len(stats.columns)) * np.nan
            stats.loc[0] = z
            # 'export name' should not be nan because then this column of the
            # DataFrame will contain multiple types, so label with string instead
            # padding end of string required for HDF5 writing
            stats['export name'] = 'not_exported'

        # add timestamp to each row of particle statistics
        stats['timestamp'] = timestamp

        # add saturation to each row of particle statistics
        stats['saturation'] = saturation

        # Time the particle statistics processing step
        proc_time = time.time() - start_time

        # Print timing information for this iteration
        infostr = '  Image {0} processed in {1:.2f} sec ({2:.1f} Hz). '
        infostr = infostr.format(i, proc_time, 1.0 / proc_time)
        print(infostr)

        # ---- END MAIN PROCESSING LOOP ----
        # ---- DO SOME ADMIN ----

    except Exception as e:
        print(e)
        traceback.print_exc()
        infostr = 'Failed to process frame {0}, skipping.'.format(i)
        print(infostr)
        return None

    return stats
