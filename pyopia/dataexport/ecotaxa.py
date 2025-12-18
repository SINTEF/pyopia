"""
Create data bundle from PyOPIA stats that can be imported into EcoTaxa.

EcoTaxa: https://ecotaxa.obs-vlfr.fr/
"""

import numpy as np
import pyopia
import pyopia.io
import pyopia.statistics
import zipfile
import io
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
from tqdm.rich import tqdm

# Define the mapping from EcoTaxa variable names to PyOPIA stats column names.
# The type expected by EcoTaxa is also defined.
# Direct mapping from PyOPIA stats row values (statsrows) is possible in many cases.
# Where no direct mapping from a single PyOPIA stats variable exists, a combination
# or retrieval of other relevant metadata is used to construct the EcoTaxa value.
_ecotaxa_dict = {
    "img_file_name": ("export_name", "str"),  # source: statsrow
    "img_rank": ("img_rank", "float"),  # source: we will make it exist
    "object_id": ("export_name", "str"),  # source: statsrow
    "object_lat": (
        "latitude",
        "float",
    ),  # needs to be in decimal degrees, source: auxillary data
    "object_lon": (
        "longitude",
        "float",
    ),  # needs to be in decimal degrees, source: auxillary data
    "object_date": (
        "timestamp",
        "str",
    ),  # UTC, format: YYYYMMDD, source: statsrow timestamp.dt.strftime(%Y%m%d)
    "object_time": (
        "timestamp",
        "str",
    ),  # UTC, format: HHMMSS, source: statsrow timestamp.dt.strftime(%H%M%S)
    "object_depth_min": ("depth", "float"),  # source: statsrow
    "object_depth_max": ("depth", "float"),  # source: statsrow
    "object_major": ("major_axis_length", "float"),  # source: statsrow
    "object_minor": ("minor_axis_length", "float"),  # source: statsrow
    "object_circ.": ("equivalent_diameter", "float"),  # source: statsrow
    "process_id": (
        "process_id",
        "float",
    ),  # If missing will be added by EcoTaxa
    "process_img_software_version": ("pyopia.__version__", "str"),
    "process_img_resolution": ("process_img_resolution", "float"),
    "process_particle_pixel_size_?m": (
        "pixel_size",
        "float",
    ),  # Needs to be converted from pixels to micrometers, the ? is a choice of EcoTaxa
    "process_date": (
        "process_date",
        "str",
    ),  # Added by custom step in create_bundle, from Modified time in PyOPIA netcdf (UTC)
    "process_time": (
        "process_time",
        "str",
    ),  # Same as process_date. datetime.now(timezone.utc).strftime("%H%M%S").
    "acq_id": ("acq_id", "float"),  # If missing will be added by EcoTaxa
    "sample_id": ("sample_id", "float"),  # If missing will be added by EcoTaxa
    "sample_stationid": (
        "sample_stationid",
        "str",
    ),  # Will be constructed from project name and station (PyOPIA netcdf)
}

# Some columns requires formatting or transformations
_ecotaxa_formatters = {
    "object_date": lambda x: x.strftime("%Y%m%d"),
    "object_time": lambda x: x.strftime("%H%M%S"),
    "img_file_name": lambda x: x + ".png",
}

_ecotaxa_types = {
    "float": "[f]",
    "str": "[t]",
}


class EcotaxaExporter:
    """Export particle statistics (xstats) and images (ROIs) to a zip file for EcoTaxa import"""

    def statsrow_to_ecotaxarow(self, statsrow, xstats_attrs):
        """Create a row in the EcoTaxa import file from PyOPIA stats row"""
        ecotaxarow = dict()
        for k, (statsname, statstype) in _ecotaxa_dict.items():
            formatter = _ecotaxa_formatters.get(k, lambda x: x)
            if statsname in statsrow.index:
                ecotaxarow[k] = formatter(statsrow.get(statsname))
            elif statsname in xstats_attrs:
                ecotaxarow[k] = formatter(xstats_attrs.get(statsname))
            else:
                ecotaxarow[k] = np.nan
            # type casting
        return ecotaxarow

    def create_bundle(
        self,
        xstats: xr.Dataset,
        export_filename: Path,
        roi_dir: Optional[Path] = None,
        make_label_folders=False,
    ):
        """
        Create an EcoTaxa import bundle file containing particle images and stats csv.

        Parameters
        ----------
        xstats : xr.Dataset
            A dataset containing particle statistics and metadata.
        export_filename : pathlib.Path
            Path to the output `.zip` file that will contain the bundle.
        roi_dir : pathlib.Path, optional
            Directory containing particle image files to include in the bundle.
            If None, the roi_dir in the xstats steps metadata will be used.
        make_label_folders : bool, optional, default False
            If True, store particle images in sub-folders with label names.
            NB: This must be False to create an EcoTaxa compatible zip file.
        Returns
        -------
        None
            This method does not return a value. It writes a zip file.
        """

        # Reconstruct PyOPIA config from xstats
        self.config = pyopia.io.steps_from_xstats(xstats)

        self.export_filename = export_filename

        # Name of particle roi directory from config, unless overridden by input argument
        self.roi_dir = (
            Path(self.config["steps"]["statextract"]["export_outputpath"])
            if roi_dir is None
            else roi_dir
        )

        # Create statistics DataFrame, add classification best guess info for all particles
        self.df_stats_export = pyopia.statistics.add_best_guesses_to_stats(
            xstats.to_pandas()
        )

        buffer = io.BytesIO()
        with zipfile.ZipFile(self.export_filename, mode="w") as zip:
            # Export particle statistics dataframe to buffer as csv
            buffer.seek(0)
            buffer.truncate(0)
            self.df_stats_export.to_csv(buffer)

            ecotaxarows = []

            # Write particle statistics csv to zip file, only if labelled folders are enabled
            # This file is not used by EcoTaxa import
            if make_label_folders:
                buffer.seek(0)
                zip.writestr("particle_statistics.csv", buffer.read())

            # create it outside here
            for idx, (_, row) in enumerate(
                tqdm(
                    self.df_stats_export.iterrows(), total=self.df_stats_export.shape[0]
                )
            ):
                # Get particle image
                export_name = row["export_name"]
                if export_name == "not_exported":
                    continue
                particle_image = pyopia.statistics.roi_from_export_name(
                    export_name, self.roi_dir
                )

                # Convert PyOPIA stats row to EcoTaxa import row
                ecotaxarows.append(self.statsrow_to_ecotaxarow(row, xstats.attrs))

                # Save the particle image to the buffer
                buffer.seek(0)
                buffer.truncate(0)
                plt.imsave(buffer, particle_image, format="png")

                # Write the buffer to the zip file with a filename
                label_folder = ""
                if make_label_folders:
                    label_folder = row["best guess"].replace("probability_", "") + "/"
                buffer.seek(0)
                zip.writestr(f"{label_folder}{export_name}.png", buffer.read())

            # Convert export table to Pandas DataFrame
            df_ecotaxa_export_table = pd.DataFrame(ecotaxarows)
            df_ecotaxa_types = pd.DataFrame(
                [
                    {
                        k: _ecotaxa_types[ettype]
                        for k, (_, ettype) in _ecotaxa_dict.items()
                    }
                ]
            )

            # Add some EcoTaxa required information manually here.
            # Construct from multiple PyOPIA data values or metadata as needed.
            pixel_size = self.config["general"]["pixel_size"]
            df_ecotaxa_export_table["process_img_software_version"] = (
                f"PyOPIA {xstats.attrs['PyOPIA_version']}"
            )
            df_ecotaxa_export_table["img_rank"] = 0
            df_ecotaxa_export_table["process_particle_pixel_size_?m"] = pixel_size
            df_ecotaxa_export_table["sample_stationid"] = "-".join(
                [xstats.attrs.get(el, "") for el in ["project_name", "station"]]
            )
            processing_datetime = pd.to_datetime(xstats.attrs["Modified"])
            df_ecotaxa_export_table["process_time"] = processing_datetime.strftime(
                "%H%M%S"
            )
            df_ecotaxa_export_table["process_date"] = processing_datetime.strftime(
                "%Y%m%d"
            )

            for col in ["object_major", "object_minor", "object_circ."]:
                df_ecotaxa_export_table[col] *= pixel_size

            # Insert type defintion as first row in exported table
            df_ecotaxa_export_table = pd.concat(
                [df_ecotaxa_types, df_ecotaxa_export_table],
                axis=0,
            )

            # Write to zip archive
            buffer.seek(0)
            buffer.truncate(0)
            df_ecotaxa_export_table.to_csv(buffer, index=False, sep="\t")
            buffer.seek(0)
            zip.writestr("ecotaxa_particle_statistics.tsv", buffer.read())
