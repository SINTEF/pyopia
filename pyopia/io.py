'''
Module containing tools for datafile and metadata handling
'''

from datetime import datetime

import h5py
import pandas as pd
import toml
import xarray
import os

from pyopia import __version__ as pyopia_version


def write_stats(
        stats,
        datafilename,
        settings=None,
        export_name_len=40,
        dataformat='nc'):
    '''
    Writes particle stats into the ouput file.
    Appends if file already exists.

    Args:
        datafilename (str):     filame prefix for -STATS.h5 file that may or may not include a path
        stats_all (DataFrame):  stats dataframe returned from processImage()
        export_name_len (int):  max number of chars allowed for col 'export name'
    '''
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
        if os.path.isfile(datafilename + '-STATS.nc'):
            existing_stats = load_stats(datafilename + '-STATS.nc')
            xstats = xarray.concat([existing_stats, xstats], 'index')

        xstats.to_netcdf(datafilename + '-STATS.nc')


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


def load_stats(datafilename):
    '''Load STATS file as a DataFrame

    Parameters
    ----------
    datafilename : str
        filename of -STATS.h5 or STATS.nc

    Returns
    -------
    DataFrame
        STATS SataFrame
    '''

    if datafilename.endswith('.nc'):
        with xarray.open_dataset(datafilename) as stats:
            stats.load()
    elif datafilename.endswith('.h5'):
        stats = pd.read_hdf(datafilename, 'ParticleStats/stats')
    else:
        print('WARNING. File extension not specified.' +
              'Assuming prefix of -STATS.h5 for backwards compatability.' +
              'In future, this function will only take .nc files')
        stats = pd.read_hdf(datafilename + '-STATS.h5', 'ParticleStats/stats')
    return stats


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
        print('STATS was likely loaded from an old h5 format, which will be deprecated in future. Please use NetCDF in future.')
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
            print(k + ':')
            print('    ' + f['Meta'].attrs[k])


class StatsToDisc():
    '''PyOpia pipline-compatible class for calling write_stats() that created NetCDF files.

    Replaces the old StatsH5 class

    Args:
        output_datafile (str): prefix path for output nc file
        dataformat (str): either 'nc' or 'h5
        append (bool): if to allow append to an existing STATS file. Defaults to True
        export_name_len (int): max number of chars allowed for col 'export name'. Defaults to 40

    Returns:
        data (dict): data from pipeline

    Example config for pipeline useage:

    .. code-block:: python

        [steps.output]
        pipeline_class = 'pyopia.io.StatsToDisc'
        output_datafile = './test' # prefix path for output nc file
    '''
    def __init__(self,
                 output_datafile='data',
                 dataformat='nc',
                 export_name_len=40):

        self.output_datafile = output_datafile
        self.dataformat = dataformat
        self.export_name_len = export_name_len

    def __call__(self, data):

        write_stats(data['stats'], self.output_datafile,
                    settings=data['settings'],
                    dataformat=self.dataformat,
                    export_name_len=self.export_name_len)

        return data


StatsH5 = StatsToDisc


def load_toml(toml_file):
    with open(toml_file, 'r') as f:
        settings = toml.load(f)
    return settings
