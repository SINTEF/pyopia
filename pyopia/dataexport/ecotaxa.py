"""
Create data bundle from PyOPIA stats that can be imported into EcoTaxa.

EcoTaxa: https://ecotaxa.obs-vlfr.fr/
"""

import pandas as pd


class EcotaxaExporter:
    def __init__(self):
        """"""
        pass

    def setup(self, xstats):
        """Create data structures compatible with EcoTaxa and copy over PyOPIA stats"""
        self.df_stats_export = pd.DataFrame()

        # List of particle images (ROIs) to bundle
        self.particle_images = []

    def create_bundle(self, filename):
        """Create and EcoTaxa import bundle file."""
