'''
Module containing tools for datafile and metadata handling
'''

from datetime import datetime

import h5py
import pandas as pd
import toml
import xarray

from pyopia import __version__ as pyopia_version


def write_stats(
        datafilename,
        stats,
        steps_string=None,
        append=True,
        export_name_len=40,
        format='h5'):
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

    if format == 'h5':
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
    elif format == 'nc':
        xstats = stats.to_xarray()
        xstats.attrs["steps"] = steps_string
        xstats.attrs['Modified'] = str(datetime.now())
        xstats.attrs['PyOpia version'] = pyopia_version
        xstats = xstats.assign_coords(time=xstats.timestamp)
        xstats.to_netcdf(datafilename + '-STATS.nc')

        if append:
            raise Exception("append not implemented for nc output yet") 


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


def load_toml(toml_file):
    with open(toml_file, 'r') as f:
        settings = toml.load(f)
    return settings


def build_repr(settings, step_name):
    repr_string = settings['steps'][step_name]['pipeline_class'] + '('
    arg_names = [k for k in list(settings['steps'][step_name].keys()) if k != 'pipeline_class']
    for i, a in enumerate(arg_names):
        v = settings['steps'][step_name][a]
        repr_string += f'{a}='
        if type(v) is str:
            repr_string += f'\'{v}\', '
        else:
            repr_string += f'{v}, '

    if len(arg_names) > 0:
        repr_string = repr_string[:-2]
    repr_string += ')'

    output = f'\'{step_name}\': {repr_string}'
    return output


def build_step_string(settings):
    step_names = list(settings['steps'].keys())
    step_str = '{'
    for step_name in step_names:
        step_str += build_repr(settings, step_name)
        step_str += ', '
    step_str = step_str[:-2]
    step_str += '}'

    return step_str
