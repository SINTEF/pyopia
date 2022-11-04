import os

import numpy as np
import pandas as pd

from pyopia.process import statextract


def timestamp_from_filename(filename):
    '''get a pandas timestamp from a silcam filename

    Args:
        filename (string): silcam filename (.silc)

    Returns:
        timestamp: from pandas.to_datetime()
    '''

    # get the timestamp of the image (in this case from the filename)
    timestamp = pd.to_datetime(os.path.splitext(os.path.basename(filename))[0][1:])
    return timestamp


class SilCamLoad():
    '''PyOpia pipline-compatible class for loading a single silcam image

    Args:
        filename (string): silcam filename (.silc)

    Returns:
        timestamp: from timestamp_from_filename()
        img (np.array): raw silcam image
    '''

    def __init__(self, filename):
        self.filename = filename
        pass

    def __call__(self):
        timestamp = timestamp_from_filename(self.filename)
        img = np.load(self.filename, allow_pickle=False)
        return timestamp, img


class SilCamStatExtract():
    '''PyOpia pipline-compatible class for calling statextract

    Args:
        minimum_area (int, optional): minimum number of pixels for particle detection. Defaults to 12.
        threshold (float, optional): threshold for segmentation. Defaults to 0.98.
        real_time_stats (bool, optional): changed segmentation method
          (@todo this option for historical reasons and should be changed). Defaults to False.
        max_coverage (int, optional): percentage of the image that is allowed to be filled by particles. Defaults to 30.
        max_particles (int, optional): maximum allowed number of particles in an image.
          exceeding this will discard the image from analysis. Defaults to 5000.
    '''
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
        stats, imbw, saturation = statextract(timestamp, imc, Classification,
                                              minimum_area=self.minimum_area,
                                              threshold=self.threshold,
                                              real_time_stats=self.real_time_stats,
                                              max_coverage=self.max_coverage,
                                              max_particles=self.max_particles)
        stats['timestamp'] = timestamp
        stats['saturation'] = saturation
        return stats
