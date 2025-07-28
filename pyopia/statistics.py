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

import logging
logger = logging.getLogger()


def d50_from_stats(stats, pixel_size):
    '''Calculate the d50 from the stats and settings

    Parameters
    ----------
    stats : DataFrame
        particle statistics from silcam process
    pixel_size : float
        pixel size in microns per pixel

    Returns
    -------
    d50 : float
        the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''
    # the volume distribution needs calculating first
    dias, vd = vd_from_stats(stats, pixel_size)

    # then the d50
    d50 = d50_from_vd(vd, dias)
    return d50


def d50_from_vd(volume_distribution, dias):
    '''
    Calculate d50 from a volume distribution

    Parameters
    ----------
    volume_distribution : array
        Particle volume distribution calculated from vd_from_stats()
    dias : array
        mid-points in the size classes corresponding the the volume distribution,
        returned from get_size_bins()

    Returns
    -------
    d50 : float
        The 50th percentile of the cumulative sum of the volume distributon, in microns
    '''
    # calculate cumulative sum of the volume distribution
    csvd = np.cumsum(volume_distribution / np.sum(volume_distribution))

    # find the 50th percentile and interpolate if necessary
    d50 = np.interp(0.5, csvd, dias)
    return d50


def get_size_bins():
    '''
    Retrieve log-spaced size bins for PSD analysis by doing the same binning as LISST-100x, but with 53 size bins

    Returns
    -------
    dias : array
        Mid-points of size bins in microns
    bin_limits : array
        Limits of size bins in microns
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
    '''Filters stats file based on whether the particles are within a rectangle specified by crop_stats.

    Parameters
    ----------
    stats : DataFrame
        Particle stats dataframe for every particle
    crop_stats : tuple
        4-tuple of lower-left (row, column) then upper-right (row, column) coord of crop

    Returns
    -------
    cropped_stats : DataFrame
        cropped silcam stats file
    '''
    r = np.array(((stats['maxr'] - stats['minr']) / 2) + stats['minr'])  # pixel row of middle of bounding box
    c = np.array(((stats['maxc'] - stats['minc']) / 2) + stats['minc'])  # pixel column of middle of bounding box

    pts = np.array([[(r_, c_)] for r_, c_ in zip(r, c)])
    pts = pts.squeeze()

    ll = np.array(crop_stats[:2])  # lower-left
    ur = np.array(crop_stats[2:])  # upper-right

    ind = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    cropped_stats = stats[ind]
    return cropped_stats


def vd_from_nd(number_distribution, dias, sample_volume=1.):
    '''Calculate volume concentration from particle count

    Parameters
    ----------
    number_distribution : array
        number distribution
    dias : array
        particle diameters in microns associated with number_distribution
    sample_volume : float, optional
        sample volume size (litres), by default 1

    Returns
    -------
    volume_distribution : array
        Particle volume distribution
    '''
    dias = dias * 1e-6  # convert to m
    particle_volume = 4 / 3 * np.pi * (dias / 2)**3  # volume in m^3
    total_particle_volume = particle_volume * number_distribution * 1e9  # volume in micro-litres
    volume_distribution = total_particle_volume / sample_volume  # micro-litres / litre

    return volume_distribution


def nc_from_nd(number_distribution, sample_volume):
    '''
    Calculate the number concentration from the count and sample volume

    Parameters
    ----------
    number_distribution : array
        number distribution
    sample_volume : float, optional
        sample volume size (litres), by default 1

    Returns
    -------
    number_concentration : float
        Particle number concentration in #/L
    '''
    number_concentration = np.sum(number_distribution) / sample_volume
    return number_concentration


def nc_vc_from_stats(stats, pix_size, path_length, imx=2048, imy=2448):
    '''Calculates important summary statistics from a stats DataFrame

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    pix_size : float
        size of pixels in microns
    path_length : float
        path length of sample volume in mm
    imx : int, optional
        number of x-dimention pixels in the image, by default 2048
    imy : int, optional
        number of y-dimention pixels in the image, by default 2448

    Returns
    -------
    number_concentration : float
        Total number concentration in #/L
    volume_concentration : float
        Total volume concentration in uL/L
    sample_volume : float
        Total volume of water sampled in L
    junge_slope : float
        Slope of a fitted juge distribution between 150-300um
    '''
    # calculate the sample volume per image
    sample_volume = get_sample_volume(pix_size, path_length, imx=imx, imy=imy)

    # count the number of images analysed
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images recorded
    sample_volume *= nims

    # calculate the number distribution
    dias, necd = nd_from_stats(stats, pix_size)

    # calculate the volume distribution from the number distribution
    volume_distribution = vd_from_nd(necd, dias, sample_volume)

    # calculate the volume concentration
    volume_concentration = np.sum(volume_distribution)

    # calculate the number concentration
    number_concentration = nc_from_nd(necd, sample_volume)

    # convert nd to units of nc per micron per litre
    number_distribution = nd_rescale(dias, necd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(number_distribution > 0)
    number_distribution[ind[0]] = np.nan

    # calcualte the junge distirbution slope
    junge_slope = get_j(dias, number_distribution)

    return number_concentration, volume_concentration, sample_volume, junge_slope


def nd_from_stats_scaled(stats, pix_size, path_length):
    '''Calcualte a scaled number distribution from stats.
    units of nd are in number per micron per litre

    Parameters
    ----------
    stats : DataFrame
        Particle statistics from silcam process
    pix_size : float
        size of pixels in microns
    path_length : float
        path length of sample volume in mm

    Returns
    -------
    dias : array
        mid-points of size bins
    number_distribution : array
        number distribution in number/micron/litre
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
    number_distribution = nd_rescale(dias, necd, sample_volume)

    # nan the first bin in measurement because it will always be part full
    ind = np.argwhere(number_distribution > 0)
    number_distribution[ind[0]] = np.nan

    return dias, number_distribution


def nd_from_stats(stats, pix_size):
    '''Calculate  number distirbution from stats
    units are number per bin per sample volume

    Parameters
    ----------
    stats : DataFrame
        particle statistics from silcam process
    pix_size : float
        pixel size in microns

    Returns
    -------
    dias : array
        mid-points of size bins
    number_distribution : array
        number distribution in number/size-bin/sample-volume
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
    number_distribution = np.float64(necd)

    return dias, number_distribution


def vd_from_stats(stats, pix_size):
    '''Calculate volume distribution from stats
    units of miro-litres per sample volume

    Parameters
    ----------
    stats : DataFrame
        particle statistics from silcam process
    pix_size : float
        pixel size in microns

    Returns
    -------
    dias : array
        mid-points of size bins
    volume_distribution : array
        volume distribution in micro-litres/sample-volume
    '''

    # obtain the number distribution
    dias, necd = nd_from_stats(stats, pix_size)

    # convert the number distribution to volume in units of micro-litres per
    # sample volume
    volume_distribution = vd_from_nd(necd, dias)

    return dias, volume_distribution


def make_montage(stats_file_or_df, pixel_size, roidir,
                 auto_scaler=500, msize=1024, maxlength=100000, crop_stats=None, brightness=1, eyecandy=True):
    '''Makes nice looking montage from a directory of extracted particle images

    Parameters
    ----------
    stats_file_or_df : DataFrame or str
        either a str specifying the location of the STATS.nc file that comes from processing, or a stats dataframe
    pixel_size : float
        pixel size of system
    roidir : str
        location of roifiles
    auto_scaler : int, optional
        approximate number of particle that are attempted to be packed into montage, by default 500
    msize : int, optional
        size of canvas in pixels, by default 1024
    maxlength : int, optional
        maximum length in microns of particles to be included in montage, by default 100000
    crop_stats : tuple, optional
        None or 4-tuple of lower-left then upper-right coord of crop, by default None
    brightness : int, optional
        brighness of packaged particles used with eyecandy option, by default 1
    eyecandy : bool, optional
        boolean which if True will explode the contrast of packed particles
        (nice for natural particles, but not so good for oil and gas)., by default True

    Returns
    -------
    montage_image : array
        montage image that can be plotted with :func:`pyopia.plotting.montage_plot`
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
    logger.info('making a montage - this might take some time....')

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
    montage_image = np.copy(montage)
    montage_image[montage > 1] = 1
    montage_image[montage == 0] = 1
    logger.info('montage complete')

    return montage_image


def gen_roifiles(stats, auto_scaler=500):
    '''Generates a list of filenames suitable for making montages with

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    auto_scaler : int
        approximate number of particle that are attempted to be pack into montage, by default 500

    Parameters
    ----------
    roifiles : list
        a list of string of filenames that can be passed to montage_maker() for making nice montages
    '''

    roifiles = stats['export_name'][stats['export_name'] != 'not_exported'].values

    # subsample the particles if necessary
    logger.info('rofiles: {0}'.format(len(roifiles)))
    IMSTEP = np.max([int(np.round(len(roifiles) / auto_scaler)), 1])
    logger.info('reducing particles by factor of {0}'.format(IMSTEP))
    roifiles = roifiles[np.arange(0, len(roifiles), IMSTEP)]
    logger.info('rofiles: {0}'.format(len(roifiles)))

    return roifiles


def get_sample_volume(pix_size, path_length, imx=2048, imy=2448):
    ''' calculate the sample volume of one image

    Parameters
    ----------
    pix_size : float
        size of pixels in microns
    path_length : float
        path length of sample volume in mm
    imx : int, optional
        image x dimention in pixels, by default 2048
    imy : int, optional
        image y dimention in pixels, by default 2448

    Returns
    -------
    sample_volume_litres : float
        Volume of the sample volume in litres
    '''
    sample_volume_litres = imx * pix_size / 1000 * imy * pix_size / 1000 * path_length * 1e-6

    return sample_volume_litres


def get_j(dias, number_distribution):
    '''Calculates the junge slope from a correctly-scale number distribution
    (number per micron per litre must be the units of nd)

    Parameters
    ----------
    dias : array
        mid-point of size bins
    number_distribution : array
        number distribution in number per micron per litre

    Returns
    -------
    junge_slope : float
        Junge slope from fitting of psd between 150 and 300um
    '''
    # conduct this calculation only on the part of the size distribution where
    # LISST-100 and SilCam data overlap
    ind = np.isfinite(dias) & np.isfinite(number_distribution) & (dias < 300) & (dias > 150)

    # use polyfit to obtain the slope of the ditriubtion in log-space (which is
    # assumed near-linear in most parts of the ocean)
    p = np.polyfit(np.log(dias[ind]), np.log(number_distribution[ind], where=number_distribution[ind] > 0), 1)
    junge_slope = p[0]
    return junge_slope


def count_images_in_stats(stats):
    '''count the number of raw images used to generate stats

    Parameters
    ----------
    stats : DataFrame
        particle statistics

    Returns
    -------
    n_images : int
        number of images in the stats data
    '''
    u = pd.to_datetime(stats['timestamp']).unique()
    n_images = len(u)

    return n_images


def extract_nth_largest(stats, n=0):
    '''Return statistics of the nth largest particle

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    n : int, optional
        nth largest particle to use, by default 0

    Returns
    -------
    stats_extract
        statistics of the nth largest particle
    '''
    stats_sorted = stats.sort_values(by=['equivalent_diameter'], ascending=False, inplace=False)
    stats_extract = stats_sorted.iloc[n]
    return stats_extract


def extract_oil(stats, proabability_threshold=0.85, solidity_threshold=0.95, feret_threshold=0.3):
    '''Creates a new stats dataframe containing only oil, based on some thresholds on calculated statistic

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    proabability_threshold : float, optional
        Threshold applied to probability_oil (from the classifier), by default 0.85
    solidity_threshold : float, optional
        Threshold applied to the solidity statistic (area of object / convex hull).
        For droplets, this threshold is used as a crude way of removing operlapping droplets
        by ensuring there are no substantial indents in the alpha shape, by default 0.95
    feret_threshold : float, optional
        Threshold of deformation (minor/major axis) beyond which the droplet is considered
        significantly deformed or at risk of breakup., by default 0.3

    Returns
    -------
    oilstats
        particle statistics for just oil (a new stats dataframe containing only oil).
        .. warning: this returned dataframe will likely have a shorter length than the original,
        so be carefull to include all analysed images when calculating volume concentraitons
    '''
    ma = stats['minor_axis_length'] / stats['major_axis_length']
    stats = stats[ma > feret_threshold]  # cannot have a deformation more than 0.3
    stats = stats[stats['solidity'] > solidity_threshold]
    ind = np.logical_or((stats['probability_oil'] > stats['probability_bubble']),
                        (stats['probability_oil'] > stats['probability_oily_gas']))

    ind2 = (stats['probability_oil'] > proabability_threshold)

    ind = np.logical_and(ind, ind2)

    oil_stats = stats[ind]
    return oil_stats


def extract_nth_longest(stats, n=0):
    '''Return statistics of the nth longest particle

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    n : int, optional
        nth largest particle to use, by default 0

    Returns
    -------
    stats_extract
        statistics of the nth largest particle
    '''
    stats_sorted = stats.sort_values(by=['major_axis_length'], ascending=False, inplace=False)
    stats_extract = stats_sorted.iloc[n]
    return stats_extract


def explode_contrast(im):
    '''Eye-candy function for exploding the contrast of a particle iamge (roi)

    Parameters
    ----------
    im : array
        image (normally a particle ROI)

    Returns
    -------
    im_mod : array
        image following exploded contrast
    '''
    # re-scale the instensities in the image to chop off some ends
    p1, p2 = np.percentile(im, (0, 80))
    im_mod = rescale_intensity(im, in_range=(p1, p2))

    # set minimum value to zero
    im_mod -= np.min(im_mod)

    # set maximum value to one
    im_mod /= np.max(im_mod)
    return im_mod


def bright_norm(im, brightness=1.):
    '''Eye-candy function for normalising the image brightness

    Parameters
    ----------
    im : array
        image (normally a particle ROI)
    brightness : float, optional
        median of histogram will be shifted to align with this value. Should be a float between 0-1, by default 1

    Returns
    -------
    im : array
        image with modified brightness
    '''
    peak = np.median(im.flatten())
    bm = brightness - peak

    im = np.float64(im) + bm
    im[im > 1] = 1

    return im


def nd_rescale(dias, number_distribution, sample_volume):
    '''Rescale a number distribution from number per bin per sample volume to number per micron per litre.

    Parameters
    ----------
    dias : array
        mid-points of size bins
    number_distribution : array
        unscaled number distribution
    sample_volume : float
        sample volume of each image

    Returns
    -------
    number_distribution_scaled : array
        scaled number distribution (number per micron per litre)
    '''

    number_distribution_scaled = np.float64(number_distribution) / sample_volume  # nc per size bin per litre

    # convert nd to units of nc per micron per litre
    dd = np.gradient(dias)
    number_distribution_scaled /= dd
    number_distribution_scaled[number_distribution_scaled < 0] = np.nan  # and nan impossible values!
    return number_distribution_scaled


def add_depth_to_stats(stats, time, depth):
    '''If you have a depth time-series, use this function to find the depth of each line in stats

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    time : array
        time stamps associated with depth argument
    depth : array
        depths associated with the time argument

    Returns
    -------
    stats : DataFrame
        particle statistics now with a 'Depth' column for each particle
    '''
    # get times
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Depth'] = np.interp(np.float64(sctime), np.float64(time), depth)
    return stats


def roi_from_export_name(exportname, path):
    '''Returns an image from the export_name string in the -STATS.h5 file

    Get the exportname like this:
    ```python
    exportname = stats['export_name'].values[0]
    ```

    Parameters
    ----------
    exportname : str
        string containing the name of the exported particle e.g. stats['export_name'].values[0]
    path : str
        path to exported h5 files

    Returns
    -------
    im : array
        particle ROI image
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
    '''Extracts the stats data from within the last number of seconds specified by window_size.

    Parameters
    ----------
    stats : DataFrame
        particle statistics
    window_size : float
        number of seconds to extract from the end of the stats data

    Returns
    -------
    stats_selected : DataFrame
        particle statistics after specified time window (given by `window_size`)
    '''
    end = np.max(pd.to_datetime(stats['timestamp']))
    start = end - pd.to_timedelta('00:00:' + str(window_size))
    stats_selected = stats[pd.to_datetime(stats['timestamp']) > start]
    return stats_selected


def make_timeseries_vd(stats, pixel_size, path_length, time_reference):
    '''Makes a dataframe of time-series volume distribution and d50
    similar to Sequoia LISST-100 output,
    and exportable to things like Excel or csv.

    Note
    ----
    If zero particles are detected within the stats daraframe,
    then the volume concentration should be reported as zero for that
    time. For this function to have awareness of these times, it requires
    time_reference variable. If you use `stats['timestamp'].unique()` for this,
    then you are assuming you have at least one particle per image.
    It is better to use `image_stats['timestamp'].values` instead, which can be obtained from
    :func:`pyopia.io.load_image_stats`

    Parameters
    ----------
    stats : DataFrame
        loaded from a *-STATS.nc file (convert from xarray like this: `stats = xstats.to_dataframe()`)
    pixel_size : float
        pixel size in microns per pixel
    path_length : float
        path length of the sample volume in mm
    time_reference : array
        time-series associated with the stats dataset stats['timestamp'].unique()

    Returns
    -------
    time_series : DataFrame
        time series volume concentrations are in uL/L columns with number headings are diameter mid-points

    Example
    -------
    .. code-block:: python
        path_length = 40  # for a 40mm long path length

        time_series_vd = pyopia.statistics.make_timeseries_vd(stats,
                                settings['general']['pixel_size'],
                                path_length,
                                image_stats['timestamp'].values)

        # particle diameters
        dias = np.array(time_series_vd.columns[0:52], dtype=float)

        # an array of volume concentrations with shape (diameter, time)
        vdarray = time_series_vd.iloc[:, 0:52].to_numpy(dtype=float)

        # time-series of d50 in each image
        d50 = time_series_vd.iloc[:, 52].to_numpy(dtype=float)

        # time variable
        time = pd.to_datetime(time_series_vd['Time'].values)

        # time-series of total volume concentration
        volume_concentration = np.sum(vdarray, axis=1)
    '''
    sample_volume = get_sample_volume(pixel_size, path_length=path_length)

    vdts = np.zeros((len(time_reference), len(get_size_bins()[0])), dtype=np.float64)
    d50 = np.zeros((len(time_reference)), dtype=np.float64) * np.nan
    for i, s in enumerate(tqdm(time_reference)):
        nims = count_images_in_stats(stats[stats['timestamp'] == s])
        if nims > 0:
            dias, vd = vd_from_stats(stats[stats['timestamp'] == s], pixel_size)
            sv = sample_volume * nims
            vd /= sv
            d50[i] = d50_from_vd(vd, dias)
            vdts[i, :] = vd

    if len(vdts) == 0:
        dias, limits = get_size_bins()
        vdts = np.zeros_like(dias) * np.nan

        time_series = pd.DataFrame(data=[np.squeeze(vdts)], columns=dias)

        time_series['D50'] = np.nan
        time_series['Time'] = np.nan

        return time_series

    time_series = pd.DataFrame(data=np.squeeze(vdts), columns=dias)

    time_series['D50'] = d50
    time_series['Time'] = pd.to_datetime(time_reference)

    time_series.sort_values(by='Time', inplace=True, ascending=True)

    return time_series


def statscsv_to_statshdf(stats_file):
    '''Convert old STATS.csv file to a STATS.h5 file

    Parameters
    ----------
    stats_file : str
        filename of stats file
    '''
    stats = pd.read_csv(stats_file, index_col=False)
    assert stats_file[-10:] == '-STATS.csv', f"Stats file {stats_file} should end in '-STATS.csv'."
    write_stats(stats_file[:-10], stats, append=False)


def trim_stats(stats_file, start_time, end_time, write_new=False, stats=[]):
    '''Chops a STATS.h5 file given a start and end time

    Parameters
    ----------
    stats_file : str
        filename of stats file
    start_time : timestr
        start time of interesting window
    end_time : timestr
        end time of interesting window
    write_new : bool, optional
        if True will write a new stats csv file to disc, by default False
    stats : DataFrame, optional
        pass stats DataFrame into here if you don't want to load the data from the stats_file given.
        In this case the stats_file string is only used for creating the new output datafilename., by default []

    Returns
    -------
    trimmed_stats : DataFrame
        particle statistics
    outname : str
        name of new stats csv file written to disc
    '''
    if len(stats) == 0:
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    trimmed_stats = stats[
        (pd.to_datetime(stats['timestamp']) > start_time) & (pd.to_datetime(stats['timestamp']) < end_time)]

    if np.isnan(trimmed_stats.equivalent_diameter.max()) or len(trimmed_stats) == 0:
        logger.info('No data in specified time range!')
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

    Parameters
    ----------
    stats : DataFrame)
        particle statistics from silcam process

    Returns
    -------
    stats : DataFrame
        particle statistics from silcam process with new columns for best guess and best guess value
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

    Parameters
    ----------
    h5file : str
        h5 filename from exported data from silcam process
    '''
    with h5py.File(h5file, 'r') as f:
        keys = list(f['Meta'].attrs.keys())

        for k in keys:
            logger.info(k + ':')
            logger.info('    ' + f['Meta'].attrs[k])


def vd_to_nd(volume_distribution, dias):
    '''convert volume distribution to number distribution

    Parameters
    ----------
    volume_distribution : array
        particle volume distribution calculated from vd_from_stats()
    dias : array
        mid-points in the size classes corresponding the the volume distribution, returned from get_size_bins()

    Returns
    -------
    number_distribution : array
        number distribution as number per micron per bin (scaling is the same unit as the input vd)
    '''
    DropletVolume = ((4 / 3) * np.pi * ((dias * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
    number_distribution = volume_distribution / (DropletVolume * 1e9)  # the number distribution in each bin
    return number_distribution


def vd_to_nc(volume_distribution, dias):
    '''calculate number concentration from volume distribution

    Parameters
    ----------
    volume_distribution : array
        particle volume distribution calculated from vd_from_stats()
    dias : array
        mid-points in the size classes corresponding the the volume distribution, returned from get_size_bins()

    Returns
    -------
    number_concentration : float
        number concentration (scaling is the same unit as the input vd).
        If vd is a 2d array [time, vd_bins], nc will be the concentration for row
    '''
    number_distribution = vd_to_nd(dias, volume_distribution)
    if np.ndim(number_distribution) > 1:
        number_concentration = np.sum(number_distribution, axis=1)
    else:
        number_concentration = np.sum(number_distribution)
    return number_concentration
