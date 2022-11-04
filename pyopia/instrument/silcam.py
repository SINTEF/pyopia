import pandas as pd
import os
import numpy as np
from pyopia.process import statextract


def timestamp_from_filename(filename):

    # get the timestamp of the image (in this case from the filename)
    timestamp = pd.to_datetime(os.path.splitext(os.path.basename(filename))[0][1:])
    return timestamp


class SilCamLoad():

    def __init__(self, filename):
        self.filename = filename
        pass

    def __call__(self):

        timestamp = timestamp_from_filename(self.filename)

        img = np.load(self.filename, allow_pickle=False)

        return timestamp, img


class SilCamStatExtract():

    def __init__(self,
                 minimum_area=12,
                 threshold=0.98,
                 real_time_stats=False,
                 max_coverage=30,
                 max_particles=5000):

        self.minimum_area = minimum_area
        self.threshold = threshold
        self.real_time_stats = real_time_stats
        self.max_coverage = max_coverage
        self.max_particles = max_particles
        pass

    def __call__(self, timestamp, imc, Classification):
        stats = statextract(timestamp, imc, Classification,
                            minimum_area=self.minimum_area, threshold=self.threshold,
                            real_time_stats=self.real_time_stats,
                            max_coverage=self.max_coverage, max_particles=self.max_particles)
        return stats
