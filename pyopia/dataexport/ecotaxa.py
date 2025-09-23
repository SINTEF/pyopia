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

        buffer = io.BytesIO()
        with zipfile.ZipFile(self.export_filename, "w") as zip:
            # Export particle statistics dataframe to buffer as csv
            buffer.seek(0)
            buffer.truncate(0)
            self.df_stats_export.to_csv(buffer)

            # Write particle statistics csv to zip file
            buffer.seek(0)
            zip.writestr("particle_statistics.csv", buffer.read())

            for _, row in tqdm(
                self.df_stats_export.iterrows(), total=self.df_stats_export.shape[0]
            ):
                # Get particle image
                export_name = row["export_name"]
                particle_image = pyopia.statistics.roi_from_export_name(
                    export_name, self.roi_dir
                )

                # Save the particle image to the buffer
                buffer.seek(0)
                buffer.truncate(0)
                plt.imsave(buffer, particle_image, format="png")

                # Write the buffer to the zip file with a filename
                label_folder = row["best guess"].replace("probability_", "")
                buffer.seek(0)
                zip.writestr(f"{label_folder}/{export_name}.png", buffer.read())
