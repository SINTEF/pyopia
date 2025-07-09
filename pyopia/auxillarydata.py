import pandas as pd
import logging
import xarray as xr

logger = logging.getLogger()

AUXILLARY_DATA_FILE_TEMPLATE = """% COMMENT LINE: PLEASE UPDATE THIS FILE WITH PROJECT RELEVANT DATA. EACH COLUMN WILL BECOME A NETCDF VARIABLE.
% COMMENT LINE: ONE LINE PER MEASUREMENT, TIME IS INTERPOLATED TO IMAGE DATA TIMES IN PYOPIA. FOLLOWING LINES ARE UNITS, DESCRIPTION AND VARIABLE NAME.
,metres,degC
Time of measurement,Depth at measurement location,Temperature at measurement location
time,depth,temperature
2022-06-08T18:40:00.00000,0.0,5.0
2022-06-08T18:41:00.00000,5.0,6.0
2022-06-08T18:42:00.00000,10.0,7.0
2022-06-08T18:43:00.00000,20.0,8.0
"""


class AuxillaryData:
    """
    Handle auxillary data for PyOPIA particle statistics file.

    Auxillary data variables may include (image) depth, longitude, latitude, etc.
    This class parses a well-defined defined .csv input format, see example below.

    Parameters
    ----------
    auxillary_data_path : str
        Path to auxillary data files .csv creates by the enduser


    Example of auxillary data file
    ------------------------------
    % COMMENT LINE:
    % COMMENT LINE:
    ,m,degC,psu
    Time of measurement,Depth at measurement location,Temperature at measurement location
    time,depth,temperature,salinity
    2025-03-19T16:59:29.950729,0.0,5.0,34
    2025-03-19T17:59:29.950729,0.0,5.0,34


    Note
    ----
    Each column will become a netCDF variable
    The two first lines are comments, and are ignored
    Note that the third and fourth rows contain units and description for each variable
    """

    def __init__(self, auxillary_data_path=None):
        self.auxillary_data_path = auxillary_data_path
        if auxillary_data_path is not None:
            self.auxillary_data = self.load_auxillary_data(auxillary_data_path)
        else:
            self.auxillary_data = pd.DataFrame(
                index=pd.Index([], name="time")
            ).to_xarray()

    def load_auxillary_data(self, auxillary_data_path):
        """Load and format uxillary data from .csv file"""

        # Load in the auxillary data file
        auxillary_data = pd.read_csv(auxillary_data_path, skiprows=4)

        # Load units and description rows
        units = pd.read_csv(auxillary_data_path, skiprows=2, nrows=0).columns
        long_names = pd.read_csv(auxillary_data_path, skiprows=3, nrows=0).columns

        # Set time as the index and make sure its type is datetime64[ns]
        auxillary_data["time"] = auxillary_data["time"].astype("datetime64[ns]")
        auxillary_data = auxillary_data.set_index("time")

        # Transform into xarray, add units
        auxillary_data = auxillary_data.to_xarray()
        for i, col in enumerate(auxillary_data.data_vars):  # Iterate over each column
            auxillary_data[col].attrs["units"] = units[i + 1]
            auxillary_data[col].attrs["long_name"] = long_names[i + 1]

        logging.info(auxillary_data)

        return auxillary_data

    def add_auxillary_data_to_xstats(self, xstats):
        """Add auxillary data to a PyOPIA xstats object"""
        logging.info("Adding auxillary data to xstats and storing to new file")

        # Add each auxillary data variable to xstats, interpolated to xstats times
        for (
            data_var
        ) in self.auxillary_data.data_vars:  # Iterate over each data variable
            xstats[data_var] = xr.DataArray(
                data=self.auxillary_data[data_var]
                .astype(float)
                .interp(time=xstats["timestamp"]),
                dims=("index",),
                coords=xstats.coords,
                attrs=self.auxillary_data[data_var].attrs,
            )

        return xstats
