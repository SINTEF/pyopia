"""
Create data bundle from PyOPIA stats that can be imported into EcoTaxa.

EcoTaxa: https://ecotaxa.obs-vlfr.fr/
"""

import pyopia.io
import pyopia.statistics
import zipfile
import io
import xarray as xr
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
from tqdm.rich import tqdm


class EcotaxaExporter:
    """Export particle statistics (xstats) and images (ROIs) to a zip file for EcoTaxa import"""

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

        # Create ecotaxa dataframe, which specified (mostly) pyopia name and types
        # define this as a dict
        ecotaxa_dict = {
            "img_statsrow_name": ("export_name", "str"),  # source: statsrow
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
            "object_depth_min": ("Depth", "float"),  # source: statsrow
            "object_depth_max": ("Depth", "float"),  # source: statsrow
            "object_major": ("major_axis_length", "float"),  # source: statsrow
            "object_minor": ("minor_axis_length", "float"),  # source: statsrow
            "object_circ.": ("equivalent_diameter", "float"),  # source: statsrow
            "process_id": (
                "process_id",
                "float",
            ),  # if missing will be added by EcoTaxa
            "process_img_software_version": ("pyopia.__version__", "str"),
            "process_img_resolution": ("process_img_resolution", "float"),
            "particle_pixel_size_μm": (
                "pixel_size",
                "float",
            ),  # needs to be converted from pixels to micrometers
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
            ), # we will make it exist, sintef specific marker
        }

        buffer = io.BytesIO()
        with zipfile.ZipFile(self.export_filename, "w") as zip:
            # Export particle statistics dataframe to buffer as csv
            buffer.seek(0)
            buffer.truncate(0)
            self.df_stats_export.to_csv(buffer)

            # go after the loop, with a ecoataxa df instead of df statsexport
            def statsrow_to_ecoataxarow(statsrow):
                ecotaxarow = dict()
                for k, (statsname, statstype) in ecotaxa_dict.items():
                    ecotaxarow[k] = statsrow.get(statsname)
                    # type casting
                return ecotaxarow

            ecotaxarows = []

            # Write particle statistics csv to zip file
            buffer.seek(0)
            zip.writestr("particle_statistics.csv", buffer.read())
            # create it outside here
            for _, row in tqdm(
                self.df_stats_export.iterrows(), total=self.df_stats_export.shape[0]
            ):
                # Get particle image
                export_name = row["export_name"]
                particle_image = pyopia.statistics.roi_from_export_name(
                    export_name, self.roi_dir
                )

                # call it here
                ecotaxarows.append(statsrow_to_ecoataxarow(row))

                # Save the particle image to the buffer
                buffer.seek(0)
                buffer.truncate(0)
                plt.imsave(buffer, particle_image, format="png")

                # Write the buffer to the zip file with a filename
                label_folder = row["best guess"].replace("probability_", "")
                buffer.seek(0)
                zip.writestr(f"{label_folder}/{export_name}.png", buffer.read())
