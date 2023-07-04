'''
Module containing tools for datafile and metadata handling
'''

from datetime import datetime

import h5py
import pandas as pd

from pyopia import __version__ as pyopia_version


def write_stats(
        datafilename,
        stats,
        steps_string=None,
        append=True,
        export_name_len=40):
    '''
    Writes particle stats into the ouput file

    Args:
        datafilename (str):     filame prefix for -STATS.h5 file that may or may not include a path
        stats_all (DataFrame):  stats dataframe returned from processImage()
        append (bool):          if to allow append
        export_name_len (int):  max number of chars allowed for col 'export name'
    '''

    # create or append particle statistics to output file
    # if the output file does not already exist, create it
    # otherwise data will be appended
    # @todo accidentally appending to an existing file could be dangerous
    # because data will be duplicated (and concentrations would therefore
    # double) GUI promts user regarding this - directly-run functions are more dangerous.
    if 'export name' in stats.columns:
        min_itemsize = {'export name': export_name_len}
    else:
        min_itemsize = None

    with pd.HDFStore(datafilename + '-STATS.h5', 'a') as fh:
        stats.to_hdf(
            fh, 'ParticleStats/stats', append=append, format='t',
            data_columns=True, min_itemsize=min_itemsize)

    # metadata
    with h5py.File(datafilename + '-STATS.h5', "a") as fh:
        meta = fh.require_group('Meta')
        meta.attrs['Modified'] = str(datetime.now())
        meta.attrs['PyOpia version'] = pyopia_version
        meta.attrs['Pipeline steps'] = steps_string


def load_stats(datafile_hdf):
    stats = pd.read_hdf(datafile_hdf + '-STATS.h5', 'ParticleStats/stats')
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


class StatsH5():
    '''PyOpia pipline-compatible class for calling write_stats()
    '''
    def __init__(self, datafilename):
        self.datafilename = datafilename

    def __call__(self,
                 data,
                 append=True,
                 export_name_len=40):
        write_stats(self.datafilename, data['stats'], steps_string=data['steps_string'],
                    append=append, export_name_len=export_name_len)
        return data
