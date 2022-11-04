import pandas as pd
import h5py
from pyopia import __version__ as pyopia_version
from datetime import datetime


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

    def __init__(self, datafilename):
        self.datafilename = datafilename

    def __call__(self,
                 stats,
                 steps_string=None,
                 append=True,
                 export_name_len=40):
        write_stats(self.datafilename, stats, steps_string=steps_string, append=append, export_name_len=40)
        pass