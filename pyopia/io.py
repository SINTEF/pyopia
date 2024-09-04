'''
Module containing tools for datafile and metadata handling
'''

from datetime import datetime

import h5py
import pandas as pd
import toml
import xarray
import os
from glob import glob

from pyopia import __version__ as pyopia_version

import logging
logger = logging.getLogger()

# The netcdf4 engine seems to produce errors with the stats dataset, so we use h5netcdf instead
NETCDF_ENGINE = 'h5netcdf'


def write_stats(stats,
                datafilename,
                settings=None,
                export_name_len=40,
                dataformat='nc',
                append=True,
                image_stats=None):
    '''
    Writes particle stats into the ouput file.
    Appends if file already exists.

    Args:
        datafilename (str):     filame prefix for -STATS.h5 file that may or may not include a path
        stats_all (DataFrame):  stats dataframe returned from processImage()
        export_name_len (int):  max number of chars allowed for col 'export name'
        append (bool):          Append all processed data into one nc file.
                                Defaults to True.
                                If False, then one nc file will be generated per raw image,
                                which can be loaded using :func:`pyopia.io.combine_stats_netcdf_files`
                                This is useful for larger datasets, where appending causes substantial slowdown
                                as the dataset gets larger.
    '''

    if len(stats) == 0:  # to avoid issue with wrong time datatypes in xarray
        return

    if 'export name' in stats.columns:
        min_itemsize = {'export name': export_name_len}
    else:
        min_itemsize = None

    if dataformat == 'h5':
        with pd.HDFStore(datafilename + '-STATS.h5', 'a') as fh:
            stats.to_hdf(
                fh, 'ParticleStats/stats', append=True, format='t',
                data_columns=True, min_itemsize=min_itemsize)

        # metadata
        with h5py.File(datafilename + '-STATS.h5', "a") as fh:
            meta = fh.require_group('Meta')
            meta.attrs['Modified'] = str(datetime.now())
            meta.attrs['PyOpia version'] = pyopia_version
            meta.attrs['Pipeline steps'] = settings
    elif dataformat == 'nc':
        xstats = make_xstats(stats, settings)
        if append and os.path.isfile(datafilename + '-STATS.nc'):
            existing_stats = load_stats(datafilename + '-STATS.nc')
            xstats = xarray.concat([existing_stats, xstats], 'index')
        elif not append:
            datafilename += ('-Image-D' +
                             str(xstats['timestamp'][0].values).replace('-', '').replace(':', '').replace('.', '-'))
        encoding = {k: {'dtype': 'str'} for k in ['export name', 'holo_filename'] if k in xstats.data_vars}
        xstats.to_netcdf(datafilename + '-STATS.nc', encoding=encoding, engine=NETCDF_ENGINE, format='NETCDF4')

        # If we have image statistics, add this to a group
        if image_stats is not None:
            image_stats.to_xarray().to_netcdf(datafilename + '-STATS.nc', group='image_stats', mode='a', engine=NETCDF_ENGINE)


def make_xstats(stats, toml_steps):
    '''Converts a stats dataframe into xarray DataSet, with metadata

    Parameters
    ----------
    stats : Pandas DataFrame
        particle statistics
    toml_steps : dict
        TOML-based steps dictionary

    Returns
    -------
    xarray.DataSet
        Xarray version of stats dataframe, including metadata
    '''
    xstats = stats.to_xarray()
    xstats.attrs["steps"] = toml.dumps(toml_steps)
    xstats.attrs['Modified'] = str(datetime.now())
    xstats.attrs['PyOpia version'] = pyopia_version
    xstats = xstats.assign_coords(time=xstats.timestamp)
    return xstats


def load_image_stats(datafilename):
    '''Load the summary stats and time information for each image

    Parameters
    ----------
    datafilename : str
        filename of -STATS.nc

    Returns
    -------
    xarray.DataArray
        image_stats
    '''
    with xarray.open_dataset(datafilename, engine=NETCDF_ENGINE, group='image_stats') as image_stats:
        image_stats.load()
    return image_stats


def load_stats(datafilename):
    '''Load STATS file as a DataFrame

    Parameters
    ----------
    datafilename : str
        filename of -STATS.h5 or STATS.nc

    Returns
    -------
    DataFrame
        STATS DataFrame  / xarray dataset
    '''

    if datafilename.endswith('.nc'):
        with xarray.open_dataset(datafilename) as stats:
            stats.load()
    elif datafilename.endswith('.h5'):
        stats = pd.read_hdf(datafilename, 'ParticleStats/stats')
    else:
        logger.warning('WARNING. File extension not specified.' +
                       'Assuming prefix of -STATS.h5 for backwards compatability.' +
                       'In future, this function will only take .nc files')
        stats = pd.read_hdf(datafilename + '-STATS.h5', 'ParticleStats/stats')
    return stats


def combine_stats_netcdf_files(path_to_data):
    '''Combine a multi-file directory of STATS.nc files into a 'stats' xarray dataset created by :func:`pyopia.io.write_stats`
    when using 'append = false'

    Parameters
    ----------
    path_to_data : str
        Folder name containing nc files with pattern '*Image-D*-STATS.nc'

    Returns
    -------
    DataFrame
        STATS xarray dataset
    '''

    sorted_filelist = sorted(glob(os.path.join(path_to_data, '*Image-D*-STATS.nc')))
    with xarray.open_mfdataset(sorted_filelist, combine='nested', concat_dim='index') as ds:
        xstats = ds.load()

    # Check if we have image statistics in the last file, if so, load it.
    # The last file should contain the entire time series of processed images.
    try:
        ds = xarray.open_dataset(sorted_filelist[-1], group='image_stats')
    except OSError:
        image_stats = None
    else:
        image_stats = ds.load()
    finally:
        ds.close()

    return xstats, image_stats


def merge_and_save_mfdataset(path_to_data):
    '''Combine a multi-file directory of STATS.nc files into a single '-STATS.nc' file
    that can then be loaded with {func}`pyopia.io.load_stats`

    Parameters
    ----------
    path_to_data : str
        Folder name containing nc files with pattern '*Image-D*-STATS.nc'
    '''

    logging.info(f'combine stats netcdf files from {path_to_data}')
    xstats, image_stats = combine_stats_netcdf_files(path_to_data)

    settings = steps_from_xstats(xstats)

    prefix = os.path.basename(settings['steps']['output']['output_datafile'])
    output_name = os.path.join(path_to_data, prefix)

    logging.info(f'writing {output_name}')
    write_stats(xstats.to_dataframe(),
                output_name,
                settings,
                image_stats=image_stats.to_dataframe())
    logging.info(f'writing {output_name} done.')


def steps_from_xstats(xstats):
    '''Get the steps attribute from xarray version of the particle stats into a dictionary

    Parameters
    ----------
    xstats : xarray.DataSet
        xarray version of the particle stats dataframe, containing metadata

    Returns
    -------
    dict
        TOML-formatted dictionary of pipeline steps
    '''
    steps = toml.loads(xstats.__getattr__('steps'))
    return steps


def load_stats_as_dataframe(stats_file):
    '''A loading function for stats files that forces stats into a pandas DataFrame

    Parameters
    ----------
    stats_file : str
        filename of NetCDF of H5 -STATS file

    Returns
    -------
    DataFrame
        stats pandas dataframe
    '''
    # obtain particle statistics from the stats file
    stats = load_stats(stats_file)
    try:
        stats = stats.to_dataframe()
    except AttributeError:
        logger.info('STATS was likely loaded from an old h5 format, \
                    which will be deprecated in future. Please use NetCDF in future.')
        pass
    return stats


def show_h5_meta(h5file):
    '''
    prints metadata from an exported hdf5 file created from pyopia.process

    Args:
        h5file              : h5 filename from exported data from pyopia.process
    '''

    with h5py.File(h5file, 'r') as f:
        keys = list(f['Meta'].attrs.keys())

        for k in keys:
            logger.info(k + ':')
            logger.info('    ' + f['Meta'].attrs[k])


class StatsToDisc():
    '''PyOpia pipline-compatible class for calling write_stats() that created NetCDF files.

    Replaces the old StatsH5 class

    Args:
        output_datafile (str): prefix path for output nc file
        dataformat (str): either 'nc' or 'h5
        append (bool): if to allow append to an existing STATS file. Defaults to True
        export_name_len (int): max number of chars allowed for col 'export name'. Defaults to 40
        append (bool):          Append all processed data into one nc file.
                                Defaults to True.
                                If False, then one nc file will be generated per raw image,
                                which can be loaded using :func:`pyopia.io.combine_stats_netcdf_files`
                                This is useful for larger datasets, where appending causes substantial slowdown
                                as the dataset gets larger.

    Returns:
        data (dict): data from pipeline

    Example config for pipeline useage:

    .. code-block:: python

        [steps.output]
        pipeline_class = 'pyopia.io.StatsToDisc'
        output_datafile = './test' # prefix path for output nc file
        append = true
    '''
    def __init__(self,
                 output_datafile='data',
                 dataformat='nc',
                 export_name_len=40,
                 append=True):

        self.output_datafile = output_datafile
        self.dataformat = dataformat
        self.export_name_len = export_name_len
        self.append = append

    def __call__(self, data):
        write_stats(data['stats'], self.output_datafile,
                    settings=data['settings'],
                    dataformat=self.dataformat,
                    export_name_len=self.export_name_len,
                    append=self.append,
                    image_stats=data['image_stats'])

        return data


StatsH5 = StatsToDisc


def load_toml(toml_file):
    with open(toml_file, 'r') as f:
        settings = toml.load(f)
    return settings
