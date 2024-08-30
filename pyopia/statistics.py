'''
Module containing tools for handling particle image statistics after processing
'''

import os
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
import h5py
from tqdm import tqdm
from pyopia.io import write_stats, load_stats_as_dataframe


def d50_from_stats(stats, pixel_size):
    '''
    Calculate the d50 from the stats and settings

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        pixel_size                  : pixel size in microns per pixel

    Returns:
        d50 (float)                 : the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''

    # the volume distribution needs calculating first
    dias, vd = vd_from_stats(stats, pixel_size)

    # then the d50
    d50 = d50_from_vd(vd, dias)
    return d50


def d50_from_vd(vd, dias):
    '''
    Calculate d50 from a volume distribution

    Args:
        vd            : particle volume distribution calculated from vd_from_stats()
        dias          : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()

    Returns:
        d50 (float)                 : the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''

    # calculate cumulative sum of the volume distribution
    csvd = np.cumsum(vd / np.sum(vd))

    # find the 50th percentile and interpolate if necessary
    d50 = np.interp(0.5, csvd, dias)
    return d50


def get_size_bins():
    '''
    Retrieve log-spaced size bins for PSD analysis by doing the same binning as LISST-100x, but with 53 size bins

    Returns:
        dias (array)        : mid-points of size bins in microns
        bin_limits (array)  : limits of size bins in microns
    '''
    # pre-allocate
    bin_limits = np.zeros((53), dtype=np.float64)

    # define the upper limit of the smallest bin (same as LISST-100x type-c)
    bin_limits[0] = 2.72 * 0.91

    # loop through 53 size classes and calculate the bin limits
    for bin_number in np.arange(1, 53, 1):
        # each bin is 1.18 * larger than the previous
        bin_limits[bin_number] = bin_limits[bin_number - 1] * 1.180

    # pre-allocate
    dias = np.zeros((52), dtype=np.float64)

    # define the middle of the smallest bin (same as LISST-100x type-c)
    dias[0] = 2.72

    # loop through 53 size classes and calculate the bin mid-points
    for bin_number in np.arange(1, 52, 1):
        # each bin is 1.18 * larger than the previous
        dias[bin_number] = dias[bin_number - 1] * 1.180

    return dias, bin_limits


def crop_stats(stats, crop_stats):
    '''
    Filters stats file based on whether the particles are
    within a rectangle specified by crop_stats.

    Args:
        stats (df)    : silcam stats file
        crop_stats (tuple) : 4-tuple of lower-left (row, column) then upper-right (row, column) coord of crop

    Returns:
        stats (df)    : cropped silcam stats file
    '''

    r = np.array(((stats['maxr'] - stats['minr']) / 2) + stats['minr'])  # pixel row of middle of bounding box
    c = np.array(((stats['maxc'] - stats['minc']) / 2) + stats['minc'])  # pixel column of middle of bounding box

    pts = np.array([[(r_, c_)] for r_, c_ in zip(r, c)])
    pts = pts.squeeze()

    ll = np.array(crop_stats[:2])  # lower-left
    ur = np.array(crop_stats[2:])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    stats = stats[inidx]

    return stats


def vd_from_nd(count, psize, sv=1):
    '''
    Calculate volume concentration from particle count

    sv = sample volume size (litres)

    e.g:
    sample_vol_size=25*1e-3*(1200*4.4e-6*1600*4.4e-6); %size of sample volume in m^3
    sv=sample_vol_size*1e3; %size of sample volume in litres

    Args:
        count (array) : particle number distribution
        psize (float) : pixel size of the SilCam contained in settings.PostProcess.pix_size from the config ini file
        sv=1 (float)  : the volume of the sample which should be used for scaling concentrations

    Returns:
        vd (array)    : the particle volume distribution
    '''

    psize = psize * 1e-6  # convert to m
    pvol = 4 / 3 * np.pi * (psize / 2)**3  # volume in m^3
    tpvol = pvol * count * 1e9  # volume in micro-litres
    vd = tpvol / sv  # micro-litres / litre

    return vd


def nc_from_nd(count, sv):
    '''
    Calculate the number concentration from the count and sample volume

    Args:
        count (array) : particle number distribution
        sv=1 (float)  : the volume of the sample which should be used for scaling concentrations

    Returns:
        nc (float)    : the total number concentration in #/L
    '''
    nc = np.sum(count) / sv
    return nc


def nc_vc_from_stats(stats, pix_size, path_length):
    '''
    Calculates important summary statistics from a stats DataFrame

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        pix_size                    : size of pixels in microns (settings.PostProcess.pixel_size)
        path_length                 : path length of sample volume in mm

    Returns:
        nc (float)            : the total number concentration in #/L
        vc (float)            : the total volume concentration in uL/L
        sample_volume (float) : the total volume of water sampled in L
        junge (float)         : the slope of a fitted juge distribution between 150-300um
    '''
    # @todo take imx & imy as optional inputs in nc_vc_from_stats

    # calculate the sample volume per image
    sample_volume = get_sample_volume(pix_size, path_length, imx=2048, imy=2448)

    # count the number of images analysed
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images recorded
    sample_volume *= nims

    # calculate the number distribution
    dias, necd = nd_from_stats(stats, pix_size)

    # calculate the volume distribution from the number distribution
    vd = vd_from_nd(necd, dias, sample_volume)

    # calculate the volume concentration
    vc = np.sum(vd)

    # calculate the number concentration
    nc = nc_from_nd(necd, sample_volume)

    # convert nd to units of nc per micron per litre
    nd = nd_rescale(dias, necd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(nd > 0)
    nd[ind[0]] = np.nan

    # calcualte the junge distirbution slope
    junge = get_j(dias, nd)

    return nc, vc, sample_volume, junge


def nd_from_stats_scaled(stats, pix_size, path_length):
    ''' calcualte a scaled number distribution from stats.
    units of nd are in number per micron per litre

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        pix_size                    : size of pixels in microns
        path_length                 : path length of sample volume in mm

    Returns:
        dias                        : mid-points of size bins
        nd                          : number distribution in number/micron/litre
    '''
    # calculate the number distirbution (number per bin per sample volume)
    dias, necd = nd_from_stats(stats, pix_size)

    # calculate the sample volume per image
    sample_volume = get_sample_volume(pix_size, path_length)

    # count the number of images
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images
    sample_volume *= nims

    # re-scale the units of the number distirbution into number per micron per
    # litre
    nd = nd_rescale(dias, necd, sample_volume)

    # nan the first bin in measurement because it will always be part full
    ind = np.argwhere(nd > 0)
    nd[ind[0]] = np.nan

    return dias, nd


def nd_from_stats(stats, pix_size):
    ''' calcualte  number distirbution from stats
    units are number per bin per sample volume

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        pix_size (float)            : pixel size in microns

    Returns:
        dias                        : mid-points of size bins
        necd                        : number distribution in number/size-bin/sample-volume
    '''

    # convert the equiv diameter from pixels into microns
    ecd = stats['equivalent_diameter'] * pix_size

    # ignore nans
    ecd = ecd[~np.isnan(ecd.values)]

    # get the size bins into which particles will be counted
    dias, bin_limits_um = get_size_bins()

    # count particles into size bins
    necd, edges = np.histogram(ecd, bin_limits_um)

    # make it float so other operations are easier later
    necd = np.float64(necd)

    return dias, necd


def vd_from_stats(stats, pix_size):
    ''' calculate volume distribution from stats
    units of miro-litres per sample volume

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        pix_size (float)            : pixel size in microns

    Returns:
        dias                        : mid-points of size bins
        vd                          : volume distribution in micro-litres/sample-volume
    '''

    # obtain the number distribution
    dias, necd = nd_from_stats(stats, pix_size)

    # convert the number distribution to volume in units of micro-litres per
    # sample volume
    vd = vd_from_nd(necd, dias)

    return dias, vd


def make_montage(stats_file_or_df, pixel_size, roidir,
                 auto_scaler=500, msize=1024, maxlength=100000, crop_stats=None, brightness=1, eyecandy=True):
    '''
    makes nice looking montage from a directory of extracted particle images

    Args:
        stats_file_or_df           : either a str specifying the location of the STATS.nc file that comes from processing,
                                      or a stats dataframe
        pixel_size                  : pixel size of system defined by settings.PostProcess.pix_size
        roidir                      : location of roifiles usually defined by settings.ExportParticles.outputpath
        auto_scaler=500             : approximate number of particle that are attempted to be packed into montage
        msize=1024                  : size of canvas in pixels
        maxlength=100000            : maximum length in microns of particles to be included in montage
        crop_stats=None             : None or 4-tuple of lower-left then upper-right coord of crop
        brightness=1                : brighness of packaged particles used with eyecandy option
        eyecandy=True               : boolean which if True will explode the contrast of packed particles
                          (nice for natural particles, but not so good for oil and gas).

    Returns:
        montageplot (float64        : a nicely-made montage in the form of an image,
                                      which can be plotted using plotting.montage_plot(montage, settings.PostProcess.pix_size)
    '''

    if isinstance(stats_file_or_df, str):
        stats = load_stats_as_dataframe(stats_file_or_df)
    else:
        stats = stats_file_or_df

    if crop_stats is not None:
        stats = crop_stats(stats, crop_stats)

    # remove nans because concentrations are not important here
    stats = stats[~np.isnan(stats['major_axis_length'])]
    stats = stats[(stats['major_axis_length'] * pixel_size) < maxlength]

    # sort the particles based on their length
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)

    roifiles = gen_roifiles(stats, auto_scaler=auto_scaler)

    # pre-allocate an empty canvas
    montage = np.zeros((msize, msize, 3), dtype=np.float64())
    # pre-allocate an empty test canvas
    immap_test = np.zeros_like(montage[:, :, 0])
    print('making a montage - this might take some time....')

    # loop through each extracted particle and attempt to add it to the canvas
    for files in tqdm(roifiles):
        # get the particle image from the HDF5 file
        particle_image = roi_from_export_name(files, roidir)

        # measure the size of this image
        [height, width] = np.shape(particle_image[:, :, 0])

        # sanity-check on the particle image size
        if height >= msize:
            continue
        if width >= msize:
            continue

        if eyecandy:
            # contrast exploding:
            particle_image = explode_contrast(particle_image)

            # eye-candy normalization:
            peak = np.median(particle_image.flatten())
            bm = brightness - peak
            particle_image = particle_image + bm
        else:
            particle_image = particle_image
        particle_image[particle_image > 1] = 1

        # initialise a counter
        counter = 0

        # try five times to fit the particle to the canvas by randomly moving
        # it around
        while (counter < 5):
            r = np.random.randint(1, msize - height)
            c = np.random.randint(1, msize - width)
            test = np.max(immap_test[r:r + height, c:c + width, None] + 1)

            # if the new particle is overlapping an existing object in the
            # canvas, then try again and increment the counter
            if (test > 1):
                counter += 1
            else:
                break

        # if we reach this point and there is still an overlap, then forget
        # this particle, and move on
        if (test > 1):
            continue

        # if we reach here, then the particle has found a position in the
        # canvas with no overlap, and can then be inserted into the canvas
        montage[r:r + height, c:c + width, :] = particle_image

        immap_test[r:r + height, c:c + width, None] = immap_test[r:r + height, c:c + width, None] + 1

    # now the montage is finished
    # here are some small eye-candy scaling things to tidy up
    montageplot = np.copy(montage)
    montageplot[montage > 1] = 1
    montageplot[montage == 0] = 1
    print('montage complete')

    return montageplot


def gen_roifiles(stats, auto_scaler=500):
    ''' generates a list of filenames suitable for making montages with

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        auto_scaler=500             : approximate number of particle that are attempted to be pack into montage

    Returns:
        roifiles                    : a selection of filenames that can be passed to montage_maker() for making nice montages
    '''

    roifiles = stats['export name'][stats['export name'] != 'not_exported'].values

    # subsample the particles if necessary
    print('rofiles: {0}'.format(len(roifiles)))
    IMSTEP = np.max([int(np.round(len(roifiles) / auto_scaler)), 1])
    print('reducing particles by factor of {0}'.format(IMSTEP))
    roifiles = roifiles[np.arange(0, len(roifiles), IMSTEP)]
    print('rofiles: {0}'.format(len(roifiles)))

    return roifiles


def get_sample_volume(pix_size, path_length, imx=2048, imy=2448):
    ''' calculate the sample volume of one image

    Args:
        pix_size                    : size of pixels in microns (settings.PostProcess.pixel_size)
        path_length                 : path length of sample volume in mm
        imx=2048                    : image x dimention in pixels
        imy=2448                    : image y dimention in pixels

    Returns:
        sample_volume_litres        : the volume of the sample volume in litres

    '''
    sample_volume_litres = imx * pix_size / 1000 * imy * pix_size / 1000 * path_length * 1e-6

    return sample_volume_litres


def get_j(dias, nd):
    ''' calculates the junge slope from a correctly-scale number distribution
    (number per micron per litre must be the units of nd)

    Args:
        dias                        : mid-point of size bins
        nd                          : number distribution in number per micron per litre

    Returns:
        j                           : Junge slope from fitting of psd between 150 and 300um

    '''
    # conduct this calculation only on the part of the size distribution where
    # LISST-100 and SilCam data overlap
    ind = np.isfinite(dias) & np.isfinite(nd) & (dias < 300) & (dias > 150)

    # use polyfit to obtain the slope of the ditriubtion in log-space (which is
    # assumed near-linear in most parts of the ocean)
    p = np.polyfit(np.log(dias[ind]), np.log(nd[ind]), 1)
    j = p[0]
    return j


def count_images_in_stats(stats):
    ''' count the number of raw images used to generate stats

    Args:
        stats                       : pandas DataFrame of particle statistics

    Returns:
        n_images                    : number of raw images

    '''
    u = pd.to_datetime(stats['timestamp']).unique()
    n_images = len(u)

    return n_images


def extract_nth_largest(stats, n=0):
    ''' return statistics of the nth largest particle
    '''
    stats_sorted = stats.sort_values(by=['equivalent_diameter'], ascending=False, inplace=False)
    stats_sorted = stats_sorted.iloc[n]
    return stats_sorted


def extract_nth_longest(stats, n=0):
    ''' return statistics of the nth longest particle
    '''
    stats_sorted = stats.sort_values(by=['major_axis_length'], ascending=False, inplace=False)
    stats_sorted = stats_sorted.iloc[n]
    return stats_sorted


def explode_contrast(im):
    ''' eye-candy function for exploding the contrast of a particle iamge (roi)

    Args:
        im   (float64)       : image (normally a particle ROI)

    Returns:
        im_mod (float64)     : image following exploded contrast

    '''

    # re-scale the instensities in the image to chop off some ends
    p1, p2 = np.percentile(im, (0, 80))
    im_mod = rescale_intensity(im, in_range=(p1, p2))

    # set minimum value to zero
    im_mod -= np.min(im_mod)

    # set maximum value to one
    im_mod /= np.max(im_mod)

    return im_mod


def bright_norm(im, brightness=1):
    ''' eye-candy function for normalising the image brightness

    Args:
        im   (uint8)    : image
        brightness=255  : median of histogram will be shifted to align with this value

    Return:
        im   (uint8)    : image with modified brightness

    '''
    peak = np.median(im.flatten())
    bm = brightness - peak

    im = np.float64(im) + bm
    im[im > 1] = 1

    return im


def nd_rescale(dias, nd, sample_volume):
    ''' rescale a number distribution from
            number per bin per sample volume
        to
            number per micron per litre

    Args:
        dias                : mid-points of size bins
        nd                  : unscaled number distribution
        sample_volume       : sample volume of each image

    Returns:
        nd                  : scaled number distribution (number per micron per litre)
    '''
    nd = np.float64(nd) / sample_volume  # nc per size bin per litre

    # convert nd to units of nc per micron per litre
    dd = np.gradient(dias)
    nd /= dd
    nd[nd < 0] = np.nan  # and nan impossible values!

    return nd


def add_depth_to_stats(stats, time, depth):
    ''' if you have a depth time-series, use this function to find the depth of
    each line in stats

    Args:
        stats               : pandas DataFrame of particle statistics
        time                : time stamps associated with depth argument
        depth               : depths associated with the time argument

    Return:
        stats               : pandas DataFrame of particle statistics, now with a depth column
    '''
    # get times
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Depth'] = np.interp(np.float64(sctime), np.float64(time), depth)
    return stats


def roi_from_export_name(exportname, path):
    ''' returns an image from the export name string in the -STATS.h5 file

    get the exportname like this: exportname = stats['export name'].values[0]

    Args:
        exportname              : string containing the name of the exported particle e.g. stats['export name'].values[0]
        path                    : path to exported h5 files

    Returns:
        im                      : particle ROI image

    '''

    # the particle number is defined after the time info
    pn = exportname.split('-')[1]
    # the name is the first bit
    name = exportname.split('-')[0] + '.h5'

    # combine the name with the location of the exported HDF5 files
    fullname = os.path.join(path, name)

    # open the H5 file
    fh = h5py.File(fullname, 'r')

    if (fh[pn].dtype) == np.uint8:
        im = np.float64(fh[pn]) / 255
    else:
        im = np.float64(fh[pn])

    return im


def extract_latest_stats(stats, window_size):
    ''' extracts the stats data from within the last number of seconds specified
    by window_size.

    Args:
        stats                   : pandas DataFrame of particle statistics
        window_size             : number of seconds to extract from the end of the stats data

    Returns:
        stats dataframe (from the last window_size seconds)
    '''
    end = np.max(pd.to_datetime(stats['timestamp']))
    start = end - pd.to_timedelta('00:00:' + str(window_size))
    stats = stats[pd.to_datetime(stats['timestamp']) > start]
    return stats


def make_timeseries_vd(stats, pixel_size, path_length):
    '''makes a dataframe of time-series volume distribution and d50

    Args:
        stats (silcam stats dataframe): loaded from a *-STATS.h5 file
        pixel_size () : pixel size in microns per pixel
        path_length : path length of the sample volume in mm

    Returns:
        dataframe: of time series volume concentrations are in uL/L columns with number headings are diameter min-points

    Example
    -------
    .. code-block:: python
        time_series_vd = pyopia.statistics.make_timeseries_vd(stats,
                                settings['general']['pixel_size'],
                                path_length=40)

        # particle diameters
        dias = np.array(time_series_vd.columns[0:52], dtype=float)

        # an array of volume concentrations with shape (diameter, time)
        vdarray = time_series_vd.iloc[:, 0:52].to_numpy(dtype=float)

        # time-series of d50 in each image
        d50 = time_series_vd.iloc[:, 52].to_numpy(dtype=float)

        # time variable
        time = pd.to_datetime(time_series_vd['Time'].values)

        # time-series of total volume concentration
        vc = np.sum(vdarray, axis=1)

    '''
    stats['timestamp'] = pd.to_datetime(stats['timestamp'])

    u = stats['timestamp'].unique()

    sample_volume = get_sample_volume(pixel_size, path_length=path_length)

    vdts = []
    d50 = []
    timestamp = []
    dias = []
    for s in tqdm(u):
        dias, vd = vd_from_stats(stats[stats['timestamp'] == s], pixel_size)
        nims = count_images_in_stats(stats[stats['timestamp'] == s])
        sv = sample_volume * nims
        vd /= sv
        d50_ = d50_from_vd(vd, dias)
        d50.append(d50_)
        timestamp.append(pd.to_datetime(s))
        vdts.append(vd)

    if len(vdts) == 0:
        dias, limits = get_size_bins()
        vdts = np.zeros_like(dias) * np.nan

        time_series = pd.DataFrame(data=[np.squeeze(vdts)], columns=dias)

        time_series['D50'] = np.nan
        time_series['Time'] = np.nan

        return time_series

    time_series = pd.DataFrame(data=np.squeeze(vdts), columns=dias)

    time_series['D50'] = d50
    time_series['Time'] = pd.to_datetime(timestamp)

    time_series.sort_values(by='Time', inplace=True, ascending=True)

    return time_series


def fill_zero_concentration_images(time, total_vc, class_time, class_vc):
    '''if zero particles are detected in a sub-class given to
    :func:`pyopia.statistics.make_timeseries_vd`,
    then instead of having times of 0 concentration
    they will just be missing from the output.
    Use this function to re-fill the zeros back to align with the total population.

    Parameters
    ----------
    time : array
        time
    total_vc : array
        total concentration (of length time array)
    class_time : array
        time array associated with class_vc
    class_vc : array
        class concentration (of length class_time array)

    Returns
    -------
    new_class_vc
        zero-filled class concentration (of length time array)

    Example
    -------

    .. code-block:: python

        time_series_vd = pyopia.statistics.make_timeseries_vd(stats,
                                settings['general']['pixel_size'],
                                path_length=40)

        # time variable
        time = pd.to_datetime(time_series_vd['Time'].values)
        # time-series of total volume concentration
        total_vc = np.sum(vdarray, axis=1)

        bubbles = stats[stats['probability_bubble'] > 0.8]

        time_series_vd_bubbles = pyopia.statistics.make_timeseries_vd(bubbles,
                                settings['general']['pixel_size'],
                                path_length=40)

        # time variable
        time_bubbles = pd.to_datetime(time_series_vd['Time'].values)
        # time-series of total volume concentration
        vc_bubbles = np.sum(vdarray, axis=1)

        vc_bubbles = fill_zero_concentration_images(time, total_vc, time_bubbles, vc_bubbles)

        plt.plot(time, total_vc)
        plt.plot(time, vc_bubbles)

    '''
    new_vc = np.zeros_like(total_vc)

    for i in range(len(class_time)):
        idx = np.argwhere(class_time[i] == time).flatten()
        new_vc[idx] = class_vc[i]
    return new_vc


def statscsv_to_statshdf(stats_file):
    '''Convert old STATS.csv file to a STATS.h5 file

    Args:
        stats_file              : filename of stats file
    '''
    stats = pd.read_csv(stats_file, index_col=False)
    assert stats_file[-10:] == '-STATS.csv', f"Stats file {stats_file} should end in '-STATS.csv'."
    write_stats(stats_file[:-10], stats, append=False)


def trim_stats(stats_file, start_time, end_time, write_new=False, stats=[]):
    '''Chops a STATS.h5 file given a start and end time

    Args:
        stats_file              : filename of stats file
        start_time                  : start time of interesting window
        end_time                    : end time of interesting window
        write_new=False             : boolean if True will write a new stats csv file to disc
        stats=[]                    : pass stats DataFrame into here if you don't want to load the data from the stats_file given.
                                      In this case the stats_file string is only used for creating the new output datafilename.

    Returns:
        trimmed_stats       : pandas DataFram of particle statistics
        outname             : name of new stats csv file written to disc
    '''
    if len(stats) == 0:
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    trimmed_stats = stats[
        (pd.to_datetime(stats['timestamp']) > start_time) & (pd.to_datetime(stats['timestamp']) < end_time)]

    if np.isnan(trimmed_stats.equivalent_diameter.max()) or len(trimmed_stats) == 0:
        print('No data in specified time range!')
        outname = ''
        return trimmed_stats, outname

    actual_start = pd.to_datetime(trimmed_stats['timestamp'].min()).strftime('D%Y%m%dT%H%M%S.%f')
    actual_end = pd.to_datetime(trimmed_stats['timestamp'].max()).strftime('D%Y%m%dT%H%M%S.%f')

    path, name = os.path.split(stats_file)

    outname = os.path.join(path, name.replace('-STATS.h5', '')) + '-Start' + str(actual_start) + '-End' + str(
        actual_end) + '-STATS.h5'

    if write_new:
        trimmed_stats.to_csv(outname)

    return trimmed_stats, outname


def add_best_guesses_to_stats(stats):
    '''
    Calculates the most likely tensorflow classification and adds best guesses
    to stats dataframe.

    Args:
        stats (DataFrame)           : particle statistics from silcam process

    Returns:
        stats (DataFrame)           : particle statistics from silcam process
                                      with new columns for best guess and best guess value
    '''

    cols = stats.columns

    p = np.zeros_like(cols) != 0
    for i, c in enumerate(cols):
        p[i] = str(c).startswith('probability')

    pinds = np.squeeze(np.argwhere(p))

    parray = np.array(stats.iloc[:, pinds[:]])

    stats['best guess'] = cols[pinds.min() + np.argmax(parray, axis=1)]
    stats['best guess value'] = np.max(parray, axis=1)

    return stats


def show_h5_meta(h5file):
    '''
    prints metadata from an exported hdf5 file created from silcam process

    Args:
        h5file              : h5 filename from exported data from silcam process
    '''

    with h5py.File(h5file, 'r') as f:
        keys = list(f['Meta'].attrs.keys())

        for k in keys:
            print(k + ':')
            print('    ' + f['Meta'].attrs[k])


def vd_to_nd(vd, dias):
    '''convert volume distribution to number distribution

    Args:
        vd (array)           : particle volume distribution calculated from vd_from_stats()
        dias (array)         : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()

    Returns:
        nd (array)           : number distribution as number per micron per bin (scaling is the same unit as the input vd)
    '''
    DropletVolume = ((4 / 3) * np.pi * ((dias * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
    nd = vd / (DropletVolume * 1e9)  # the number distribution in each bin
    return nd


def vd_to_nc(vd, dias):
    '''calculate number concentration from volume distribution

    Args:
        vd (array)           : particle volume distribution calculated from vd_from_stats()
        dias (array)         : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()

    Returns:
        nn (float)           : number concentration (scaling is the same unit as the input vd).
                               If vd is a 2d array [time, vd_bins], nc will be the concentration for row
    '''
    nd = vd_to_nd(dias, vd)
    if np.ndim(nd) > 1:
        nc = np.sum(nd, axis=1)
    else:
        nc = np.sum(nd)
    return nc
