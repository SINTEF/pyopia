"""
Module containing tools for datafile and metadata handling
"""

import logging
import os
from datetime import datetime
from glob import glob

import h5py
import numpy as np
import pandas as pd
import toml
import xarray
import xarray as xr
from tqdm.auto import tqdm

import json
from pathlib import Path

from pyopia import __version__ as pyopia_version

logger = logging.getLogger()

# The netcdf4 engine seems to produce errors with the stats dataset, so we use h5netcdf instead
NETCDF_ENGINE = "h5netcdf"


def write_stats(
    stats,
    datafilename,
    settings=None,
    export_name_len=40,
    dataformat="nc",
    append=True,
    image_stats=None,
    proj_metadata_df=None,
):
    """
    Writes particle stats into the ouput file.
    Appends if file already exists.

    Parameters
    ----------
    datafilename : str
        Filame prefix for -STATS.h5 file that may or may not include a path
    stats : DataFrame or xr.Dataset
        Particle statistics
    export_name_len : int
        Max number of chars allowed for col 'export_name'
    append : bool
        Append all processed data into one nc file.
        Defaults to True.
        If False, then one nc file will be generated per raw image,
        which can be loaded using :func:`pyopia.io.combine_stats_netcdf_files`
        This is useful for larger datasets,
        where appending causes substantial slowdown
        as the dataset gets larger.
    image_stats : xr.Dataset
        Summary statistics of each raw image (including those with no particles)
    proj_metadata_df : pd.DataFrame
        Project metadata, such as license, creator, etc. Added to stats netcdf
    """

    if len(stats) == 0:  # to avoid issue with wrong time datatypes in xarray
        return

    if "export_name" in stats.columns:
        min_itemsize = {"export_name": export_name_len}
    else:
        min_itemsize = None

    if dataformat == "h5":
        with pd.HDFStore(datafilename + "-STATS.h5", "a") as fh:
            stats.to_hdf(
                fh,
                "ParticleStats/stats",
                append=True,
                format="t",
                data_columns=True,
                min_itemsize=min_itemsize,
            )

        # metadata
        with h5py.File(datafilename + "-STATS.h5", "a") as fh:
            meta = fh.require_group("Meta")
            meta.attrs["Modified"] = str(datetime.now())
            meta.attrs["PyOPIA_version"] = pyopia_version
            meta.attrs["steps"] = settings
    elif dataformat == "nc":
        if isinstance(stats, xr.Dataset):
            xstats = stats
        else:
            xstats = make_xstats(stats, settings, proj_metadata_df)

        xstats = add_cf_attributes(xstats)

        if append and os.path.isfile(datafilename + "-STATS.nc"):
            existing_stats = load_stats(datafilename + "-STATS.nc")
            xstats = xarray.concat([existing_stats, xstats], "index")
        elif not append:
            # When appending, only store the last row in the image_stats DataFrame
            datafilename += "-Image-D" + str(xstats["timestamp"][0].values).replace(
                "-", ""
            ).replace(":", "").replace(".", "-")

        logger.info(f"Writing stats to file: {datafilename + '-STATS.nc'}")
        logger.debug(f"Settings provided: {settings}")

        encoding = setup_xstats_encoding(xstats)
        xstats.to_netcdf(
            datafilename + "-STATS.nc",
            encoding=encoding,
            engine=NETCDF_ENGINE,
            format="NETCDF4",
        )

        # If we have image statistics (summary data for each raw image), add the image_stats a group
        if image_stats is not None:
            if append:
                ximage_stats = image_stats.to_xarray()
            else:
                ximage_stats = image_stats.loc[[image_stats.index[-1]], :].to_xarray()

            encoding_imagestats = setup_xstats_encoding(ximage_stats)
            ximage_stats.to_netcdf(
                datafilename + "-STATS.nc",
                group="image_stats",
                mode="a",
                encoding=encoding_imagestats,
                engine=NETCDF_ENGINE,
            )


def setup_xstats_encoding(xstats, string_vars=["export_name", "holo_filename"]):
    """Setup encoding for writing to NetCDF, where string variables are explicitly defined as string types

    Notes
    -----
    Setting up encoding like this for xstats is needed because default behaviour is to set everything as float if there is no
    value, so in a situation where the first image contains no particles we must ensure that string variables are set as string
    types.

    Parameters
    ----------
    xstats : xarray.Dataset
        Xarray version of stats dataframe, including metadata
    string_vars : list, optional
        list of string columns in xstats, by default ['export_name', 'holo_filename']

    Returns
    -------
    encoding : dict
        'encoding' input argument to be given to xstats.to_netcdf()
    """
    encoding = {k: {"dtype": "str"} for k in string_vars if k in xstats.data_vars}

    # Set timestamp encoding to microseconds since smallest datetime in xstats
    # Avoid resolution issues with int64 dates (days since)
    encoding["timestamp"] = {
        "units": f"microseconds since {np.datetime_as_string(xstats.timestamp.min())}"
    }
    return encoding


def make_xstats(stats, toml_steps, proj_metadata_df=None):
    """Converts a stats dataframe into xarray DataSet, with metadata

    Parameters
    ----------
    stats : Pandas DataFrame
        particle statistics
    toml_steps : dict
        TOML-based steps dictionary
    proj_metadata_df : pd.DataFrame
        Project metadata, such as license, creator, etc. Added to stats netcdf

    Returns
    -------
    xstats : xarray.Dataset
        Xarray version of stats dataframe, including metadata
    """
    xstats = stats.to_xarray()
    xstats.attrs["steps"] = toml.dumps(toml_steps)
    xstats.attrs["Modified"] = str(datetime.now())
    xstats.attrs["PyOPIA_version"] = pyopia_version
    if proj_metadata_df is not None:
        for _, row in proj_metadata_df.iterrows():
            xstats.attrs[row.iloc[0]] = row.iloc[1]
    xstats = xstats.assign_coords({"timestamp": xstats.timestamp})
    xstats = add_cf_attributes(xstats)
    return xstats


def load_image_stats(datafilename):
    """Load the summary stats and time information for each image

    Parameters
    ----------
    datafilename : str
        filename of -STATS.nc

    Returns
    -------
    image_stats : xarray.Dataset
        summary statistics of each raw image (including those with no particles)
    """
    with xarray.open_dataset(
        datafilename, engine=NETCDF_ENGINE, group="image_stats"
    ) as image_stats:
        image_stats.load()
    return image_stats


def load_stats(datafilename):
    """Load -STATS.nc file as xarray Dataset

    .. warning:: Support for loading of old -STATS.h5 formats will be removed in version 3.0.0.
        They will need to be converted to .nc prior to loading.
        Data loaded from -STATS.h5 are returned as an xarray Dataset without metadata.

    Parameters
    ----------
    datafilename : str
        filename of -STATS.h5 or STATS.nc

    Returns
    -------
    xstats : xarray.Dataset
        Particle statistics
    """

    if datafilename.endswith(".nc"):
        with xarray.open_dataset(datafilename, engine=NETCDF_ENGINE) as xstats:
            xstats.load()
    elif datafilename.endswith(".h5"):
        logger.warning("In future, load_stats will only take .nc files")
        stats = pd.read_hdf(datafilename, "ParticleStats/stats")
        xstats = stats.to_xarray()
    else:
        logger.warning(
            "WARNING. File extension not specified."
            + "Assuming prefix of -STATS.h5 for backwards compatability."
            + "In future, load_stats will only take .nc files"
        )
        stats = pd.read_hdf(datafilename + "-STATS.h5", "ParticleStats/stats")
        xstats = stats.to_xarray()
    return xstats


def combine_stats_netcdf_files(path_to_data, prefix="*"):
    """.. deprecated:: 2.4.11
        :class:`pyopia.io.combine_stats_netcdf_files` will be removed in version 3.0.0, it is replaced by
        :class:`pyopia.io.concat_stats_netcdf_files`.

    Combine a multi-file directory of STATS.nc files into a 'stats' xarray dataset created by :func:`pyopia.io.write_stats`
    when using 'append = false'

    Parameters
    ----------
    path_to_data : str
        Folder name containing nc files with pattern '*Image-D*-STATS.nc'

    prefix : str
        Prefix to multi-file dataset (for replacing <prefix> in the file name pattern '<prefix>Image-D*-STATS.nc').
        Defaults to '*'

    Returns
    -------
    xstats : xarray.Dataset
        Particle statistics and metatdata from processing steps
    image_stats : xarray.Dataset
        summary statistics of each raw image (including those with no particles)
    """

    sorted_filelist = sorted(
        glob(os.path.join(path_to_data, prefix + "Image-D*-STATS.nc"))
    )
    with xarray.open_mfdataset(
        sorted_filelist,
        combine="nested",
        concat_dim="index",
        decode_cf=True,
        parallel=False,
        coords="minimal",
        compat="override",
    ) as ds:
        xstats = ds.load()

    # Check if we have image statistics, if so, load it.
    try:
        with xarray.open_mfdataset(sorted_filelist, group="image_stats") as ds:
            image_stats = ds.load()
    except OSError:
        logger.info(
            "Could get image_stats from netcdf files for merging, returning None for this."
        )
        image_stats = None

    return xstats, image_stats


def concat_stats_netcdf_files(sorted_filelist):
    """Concatenate specified list of STATS.nc files into one 'xstats' xarray dataset
    created by :func:`pyopia.io.write_stats when using 'append = false'.

    Existing files are first loaded and then combined, so memory usage will go up with longer file lists.

    Parameters
    ----------
    sorted_filelist : str
        List of files to be combined into single dataset

    Returns
    -------
    xstats : xarray.Dataset or None
        Particle statistics and metatdata from processing steps
    image_stats : xarray.Dataset or None
        Summary statistics of each raw image (including those with no particles)
    """
    if len(sorted_filelist) < 1:
        logger.error("No files found to concatenate, doing nothing.")
        return None, None

    # We load one dataset at the time into a list for later merge
    datasets = []
    datasets_image_stats = []

    # Check if we have image statistics in first file, if not, we skip checking the rest
    skip_image_stats = False
    try:
        with xr.open_dataset(
            sorted_filelist[0], group="image_stats", engine=NETCDF_ENGINE
        ) as ds:
            ds.load()
    except OSError:
        logger.info(
            "Could get image_stats from netcdf files for merging, returning None for this."
        )
        skip_image_stats = True

    # Load datasets from each file into the lists
    for f in tqdm(sorted_filelist, desc="Loading datasets"):
        with xr.open_dataset(f, engine=NETCDF_ENGINE) as ds:
            ds.load()
            datasets.append(ds)

        if not skip_image_stats:
            with xr.open_dataset(f, group="image_stats", engine=NETCDF_ENGINE) as ds:
                ds.load()
                datasets_image_stats.append(ds)

    # Combine the individual datasets loaded above
    logging.info("Combining datasets")
    xstats = xr.concat(datasets, dim="index")
    image_stats = None
    if not skip_image_stats:
        image_stats = xr.concat(datasets_image_stats, dim="timestamp")

    return xstats, image_stats


def merge_and_save_mfdataset(
    path_to_data, prefix="*", overwrite_existing_partials=False, chunk_size=None
):
    """Combine a multi-file directory of STATS.nc files into a single '-STATS.nc' file
    that can then be loaded with :func:`pyopia.io.load_stats`

    Parameters
    ----------
    path_to_data : str
        Folder name containing nc files with pattern '*Image-D*-STATS.nc'

    prefix : str
        Prefix to multi-file dataset (for replacing the wildcard in '*Image-D*-STATS.nc').
        Defaults to '*'

    overwrite_existing_partials : bool
        Do not reprocess existing merged netcdf files for each chunk if False.
        Otherwise reprocess (load) and overwrite. This can be used to restart
        or continue a previous merge operation as new files become available.

    chunk_size : int
        Number of files to be loaded and merged in each step. Produces a number
        of intermediate/partially merged netcdf files equal to the total number
        of input files divided by chunk_size. The last chunk may contain less
        files than specified, depending on the total number of files.
        Default: None, which processes all files together.
    """
    logging.info(f"Combine stats netcdf files from {path_to_data}")
    if (chunk_size is not None) and (chunk_size < 1):
        raise ValueError(
            f"Invalid chunk size, must be greater than 0, was {chunk_size}"
        )

    # Get sorted list of per-image stats netcdf files
    sorted_filelist = sorted(
        glob(os.path.join(path_to_data, prefix + "Image-D*-STATS.nc"))
    )

    # fmt: off
    # Chunk the file list into smaller parts if specified
    num_files = len(sorted_filelist)
    chunk_size_used = num_files if chunk_size is None else min(chunk_size, num_files)
    num_chunks = int(np.ceil(num_files / chunk_size_used))
    filelist_chunks = [
        sorted_filelist[i * chunk_size_used:min(num_files, (i + 1) * chunk_size_used)]
        for i in range(num_chunks)
    ]
    infostr = (
        f"Processing {num_chunks} partial file lists of {chunk_size_used} files each"
    )
    infostr += f", based on a total of {num_files} files."
    logging.info(infostr)
    # fmt: on

    # Get config from first file in list
    xstats = load_stats(sorted_filelist[0])
    settings = steps_from_xstats(xstats)
    prefix_out = os.path.basename(settings["steps"]["output"]["output_datafile"])
    encoding = setup_xstats_encoding(xstats)

    def process_store(i, filelist_):
        output_name = os.path.join(path_to_data, f"part-{i:04d}-{prefix_out}-STATS.nc")

        # Skip this chunk if the merged output file exists and overwrite is set to False
        if os.path.exists(output_name) and not overwrite_existing_partials:
            logging.info(f"File exists ({output_name}), skipping")
            return output_name

        # Load the individual datasets
        xstats, image_stats = concat_stats_netcdf_files(filelist_)

        # Save the particle statistics (xstats) to NetCDF
        logging.info(f"Writing {output_name}")
        if xstats is not None:
            xstats.to_netcdf(
                output_name,
                mode="w",
                encoding=encoding,
                engine=NETCDF_ENGINE,
                format="NETCDF4",
            )

        # If summary data for each raw image are available (image_stats), save this into the image_stats group
        if image_stats is not None:
            image_stats.to_netcdf(
                output_name, group="image_stats", mode="a", engine=NETCDF_ENGINE
            )
        logging.info(f"Writing {output_name} done.")

        return output_name

    # Loop over filelist chunkst and created merged netcdf files for each
    merged_files = []
    for i, filelist_ in enumerate(filelist_chunks):
        output_name = process_store(i, filelist_)
        merged_files.append(output_name)

    # Finally, merge the partially merged files
    logging.info("Doing final merge of partially merged files")
    output_name = os.path.join(path_to_data, prefix_out + "-STATS.nc")
    with xr.open_mfdataset(merged_files, concat_dim="index", combine="nested") as ds:
        ds.to_netcdf(
            output_name,
            mode="w",
            encoding=encoding,
            engine=NETCDF_ENGINE,
            format="NETCDF4",
        )

    with xr.open_mfdataset(
        merged_files, group="image_stats", concat_dim="timestamp", combine="nested"
    ) as ds:
        ds.to_netcdf(output_name, mode="a", group="image_stats", engine=NETCDF_ENGINE)

    logging.info(f"Writing {output_name} done.")


def steps_from_xstats(xstats):
    """Get the steps attribute from xarray version of the particle stats into a dictionary

    Parameters
    ----------
    xstats : xarray.DataSet
        xarray version of the particle stats dataframe, containing metadata

    Returns
    -------
    steps : dict
        TOML-formatted dictionary of pipeline steps
    """
    steps = toml.loads(xstats.__getattr__("steps"))
    return steps


def load_stats_as_dataframe(stats_file):
    """A loading function for stats files that forces stats into a pandas DataFrame

    Parameters
    ----------
    stats_file : str
        filename of NetCDF of H5 -STATS file

    Returns
    -------
    stats : DataFrame
        stats pandas dataframe
    """
    # obtain particle statistics from the stats file
    stats = load_stats(stats_file)
    try:
        stats = stats.to_dataframe()
    except AttributeError:
        logger.info(
            "STATS was likely loaded from an old h5 format, \
                    which will be deprecated in future. Please use NetCDF in future."
        )
        pass
    return stats


def show_h5_meta(h5file):
    """
    prints metadata from an exported hdf5 file created from pyopia.process

    Parameters
    ----------
    h5file : str
        h5 filename from exported data from pyopia.process
    """

    with h5py.File(h5file, "r") as f:
        keys = list(f["Meta"].attrs.keys())

        for k in keys:
            logger.info(k + ":")
            logger.info("    " + f["Meta"].attrs[k])


class StatsToDisc:
    """PyOpia pipline-compatible class for calling write_stats() that created NetCDF files.

    Parameters
    ----------
    output_datafile : str
        prefix path for output nc file
    dataformat : str
        either 'nc' or 'h5
    export_name_len : int
        max number of chars allowed for col 'export_name'. Defaults to 40
    append : bool
        Append all processed data into one nc file.
        Defaults to True.
        If False, then one nc file will be generated per raw image,
        which can be loaded using :func:`pyopia.io.combine_stats_netcdf_files`
        This is useful for larger datasets, where appending causes substantial slowdown
        as the dataset gets larger.

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        data from the pipeline

    Example
    -------
    Example config for pipeline useage:

    .. code-block:: toml

        [steps.output]
        pipeline_class = 'pyopia.io.StatsToDisc'
        output_datafile = './test' # prefix path for output nc file
        append = true
        project_metadata_file = 'metadata.txt'
    """

    def __init__(
        self,
        output_datafile="data",
        dataformat="nc",
        export_name_len=40,
        append=True,
        project_metadata_file=None,
    ):
        self.output_datafile = output_datafile
        self.dataformat = dataformat
        self.export_name_len = export_name_len
        self.append = append
        self.project_metadata_file = project_metadata_file
        if self.project_metadata_file is not None:
            self.proj_metadata_df = pd.read_csv(self.project_metadata_file)

    def __call__(self, data):
        # Add raw image shape to metadata
        self.proj_metadata_df.loc[len(self.proj_metadata_df)] = [
            "raw_image_shape",
            data["imraw"].shape,
        ]

        write_stats(
            data["stats"],
            self.output_datafile,
            settings=data["settings"],
            dataformat=self.dataformat,
            export_name_len=self.export_name_len,
            append=self.append,
            image_stats=data["image_stats"],
            proj_metadata_df=self.proj_metadata_df,
        )

        return data


def load_toml(toml_file):
    """Load a TOML settings file from file

    Parameters
    ----------
    toml_file : str
        TOML filename

    Returns
    -------
    settings : dict
        TOML settings
    """
    with open(toml_file, "r") as f:
        settings = toml.load(f)
    return settings


def StatsH5(**kwargs):
    """.. deprecated:: 2.4.8
        :class:`pyopia.io.StatsH5` will be removed in version 3.0.0, it is replaced by
        :class:`pyopia.io.StatsToDisc`.

    PyOpia pipline-compatible class for calling write_stats() that creates h5 files.

    Parameters
    ----------
    output_datafile : str
        prefix path for output nc file
    dataformat : str
        either 'nc' or 'h5
    export_name_len : int
        max number of chars allowed for col 'export_name'. Defaults to 40
    append : bool
        Append all processed data into one nc file.
        Defaults to True.
        If False, then one nc file will be generated per raw image,
        which can be loaded using :func:`pyopia.io.combine_stats_netcdf_files`
        This is useful for larger datasets, where appending causes substantial slowdown
        as the dataset gets larger.

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        data from the pipeline

    Example
    -------
    Example config for pipeline useage:

    .. code-block:: toml

        [steps.output]
        pipeline_class = 'pyopia.io.StatsH5'
        output_datafile = './test' # prefix path for output nc file
        append = true
    """
    logger.warning(
        "StatsH5 will be removed in version 3.0.0, it is replaced by pyopia.io.StatsToDisc"
    )
    return StatsToDisc(**kwargs)


def get_cf_metadata_spec():
    path_to_cf_json = Path(__file__).parent / Path("cf_metadata.json")
    with open(path_to_cf_json, "r") as file:
        return json.load(file)


def add_cf_attributes(xstats):
    """
    Adds CF-compliant attributes and units to the xarray Dataset.

    Parameters
    ----------
    xstats : xarray.Dataset
        The dataset to which CF-compliant attributes will be added.
    """

    # Read in CF Metadata from .json specification file
    cf_metadata = get_cf_metadata_spec()

    # Apply metadata from CF_METADATA if available
    for var in xstats.data_vars:
        if var in cf_metadata:
            metadata = cf_metadata[var]
            xstats[var].attrs.update(metadata)

        # Variables starting with probability_ are handled separately
        if var.startswith("probability_"):
            class_name = "_".join(var.split("_")[1:])
            xstats[var].attrs["long_name"] = (
                f"Probability of particle belonging to class {class_name}"
            )
            xstats[var].attrs["units"] = "dimensionless fraction"
            xstats[var].attrs["calculation_method"] = (
                "Determined by neural network classifier"
            )
            xstats[var].attrs["standard_name"] = f"probability_of_{class_name}"
            xstats[var].attrs["pyopia_process_level"] = 1

    return xstats
