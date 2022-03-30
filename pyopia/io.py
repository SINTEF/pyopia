import pandas as pd


def write_stats(
        datafilename,
        stats,
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
