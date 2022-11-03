import pandas as pd
import os
import numpy as np


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
    
        # setup the 'data' tuple with an image number, timestamp and the image
        data = (1, timestamp, img)
        return data