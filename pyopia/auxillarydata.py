import pathlib
import pandas as pd

# The netcdf4 engine seems to produce errors with the stats dataset, so we use
#  h5netcdf instead
NETCDF_ENGINE = "h5netcdf"

AUXILLARY_DATA_FILE_TEMPLATE = """time,depth,temperature
,metres,degC
2025-03-19T16:59:29.950729,0.0,5.0
2025-03-19T16:59:57.120311,5.5,6.3

ADD YOUR METADATA
One line per image in the dataset
Only the time column is required
Add as many auxillary columns as you need
Each column will become a netCDF variable
Note that the second row contains units for each variable
"""


class AuxillaryData:
    """
    Handle and add auxillary data to PyOPIA particle statistics file.

    Auxillary data variables may include (image) depth, longitude, latitude, etc.
    This class parses and adds such data to the PyOPIA statistics file, based on a
    defined input format.
    """

    def __init__(self, metadata_dir):
        # Load metadata
        self.metadata = pd.read_csv(
            pathlib.Path(metadata_dir, "metadata.txt"),
            index_col=0,
        )

    def add_metadata(self, xstats):
        for ncvar, ncmeta in NCVARS_META.items():
            if ncvar in xstats:
                for attrname, attrval in ncmeta.items():
                    xstats[ncvar].attrs[attrname] = attrval

        # Add global metadata
        for attrname, attrval in self.metadata.iterrows():
            xstats.attrs[attrname] = attrval.values[0]

    def store_augmented_file(self, xstats, output_filename):
        xstats.to_netcdf(output_filename, engine=NETCDF_ENGINE)
