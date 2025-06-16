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

    Parameters
    ----------
    auxillary_data_path : str
        path to auxillary data files .csv creates by the enduser

    Example
    -------
    time,depth,temperature,salinity
    ,m,degC,psu
    2025-03-19T16:59:29.950729,0.0,5.0,34

    Note
    ----
    ADD YOUR METADATA
    One line per image in the dataset
    Only the time column is required
    Add as many auxillary columns as you need
    Each column will become a netCDF variable
    Note that the second row contains units for each variable
    """

    def __init__(
        self,
        auxillary_data_path=None,
    ):
        self.auxillary_data_path = auxillary_data_path
        # Load in the auxillary data file
        self.auxillary_data = pd.read_csv(auxillary_data_path)
        # Drop the first row of metadata file which contains the units and reset the index
        self.auxillary_data = self.auxillary_data.drop([0])
        self.auxillary_data = self.auxillary_data.drop([1]).reset_index(drop=True)
        # Set time as the index and make sure its type is datetime64[ns]
        self.auxillary_data["time"] = self.auxillary_data["time"].astype(
            "datetime64[ns]"
        )
        # Transform into xarray
        self.auxillary_data = self.auxillary_data.to_xarray()

    def __call__(self, data):
        # loop over columns in the auxillary data.csv and interpolate tbased on their timestep
        if (
            self.auxillary_data is not None
        ):  # Check that the data is there and not empty
            for col in self.auxillary_data.data_vars:  # Iterate over each column
                data["stats"][col] = (
                    self.auxillary_data[col]
                    .astype(float)
                    .interp(time=data["stats"]["timestamp"])
                    .to_pandas()
                )
        return data

    def store_augmented_file(self, xstats, output_filename):
        xstats.to_netcdf(output_filename, engine=NETCDF_ENGINE)
