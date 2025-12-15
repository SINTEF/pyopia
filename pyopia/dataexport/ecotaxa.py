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

_ecotaxa_dict = {
    "img_file_name": ("export_name", "str"),  # source: statsrow
    "img_rank": ("img_rank", "float"),  # source: we will make it exist
    "object_id": ("export_name", "str"),  # source: statsrow
    "object_lat": (
        "latitude",
        "str",
    ),  # needs to be in decimal degrees, source: auxillary data
    "object_lon": (
        "longitude",
        "str",
    ),  # needs to be in decimal degrees, source: auxillary data
    "object_date": (
        "timestamp",
        "float",
    ),  # UTC, format: YYYYMMDD, source: statsrow timestamp.dt.strftime(%Y%m%d)
    "object_time": (
        "timestamp",
        "float",
    ),  # UTC, format: HHMMSS, source: statsrow timestamp.dt.strftime(%H%M%S)
    "object_depth_min": ("depth", "float"),  # source: statsrow
    "object_depth_max": ("depth", "float"),  # source: statsrow
    "object_major": ("major_axis_length", "float"),  # source: statsrow
    "object_minor": ("minor_axis_length", "float"),  # source: statsrow
    "object_circ.": ("equivalent_diameter", "float"),  # source: statsrow
    "process_id": (
        "process_id",
        "float",
    ),  # if missing will be added by EcoTaxa
    "process_img_software_version": ("pyopia.__version__", "str"),
    "process_img_resolution": ("process_img_resolution", "float"),
    "process_particle_pixel_size_?m": (
        "pixel_size",
        "float",
    ),  # needs to be converted from pixels to micrometers, the ? is a choice of EcoTaxa
    "process_date": (
        "process_date",
        "float",
    ),  # we will make it exists, datetime.today().strftime('%Y%m%d'),
    "process_time": (
        "process_time",
        "float",
    ),  # we will make it exist, datetime.now(timezone.utc).strftime("%H%M%S"),
    "acq_id": ("acq_id", "float"),  # if missing will be added by EcoTaxa
    "sample_id": ("sample_id", "float"),  # if missing will be added by EcoTaxa
    "sample_stationid": (
        "sample_stationid",
        "str",
    ),  # we will make it exist, sintef specific marker
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

    def statsrow_to_ecoataxarow(self, statsrow, xstats_attrs):
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
        self, xstats: xr.Dataset, export_filename: Path, roi_dir: Optional[Path] = None
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

            # Write particle statistics csv to zip file
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
                particle_image = pyopia.statistics.roi_from_export_name(
                    export_name, self.roi_dir
                )

                # call it here
                ecotaxarows.append(self.statsrow_to_ecoataxarow(row, xstats.attrs))

                # Save the particle image to the buffer
                buffer.seek(0)
                buffer.truncate(0)
                plt.imsave(buffer, particle_image, format="png")

                # Write the buffer to the zip file with a filename
                # label_folder = row["best guess"].replace("probability_", "")
                buffer.seek(0)
                zip.writestr(f"{export_name}.png", buffer.read())

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

            # Add some information manually here
            pixel_size = self.config["general"]["pixel_size"]
            df_ecotaxa_export_table["process_img_software_version"] = (
                f"PyOPIA {xstats.attrs['PyOPIA_version']}"
            )
            df_ecotaxa_export_table["img_rank"] = 0
            df_ecotaxa_export_table["process_particle_pixel_size_?m"] = pixel_size
            df_ecotaxa_export_table["sample_stationid"] = "-".join(
                ["SINTEF", xstats.attrs["project_name"]]
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
