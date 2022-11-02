import pyopia.process
import numpy as np


def statextract_holo(data, Classification, extractparticles_function=None, max_particles=np.inf):

    timestamp = data[1]
    im_flat = data[2]

    imbw = data[3]
    imbw = np.max(imbw, axis=2)  # quick conversion from montage to binary image

    # a work-around to make image dimensions fit - we should look into making an option for working on grayscale images
    # for now, we can just copy the grayscale into three 'RGB' channels
    r, c = np.shape(im_flat)
    img = np.zeros((r, c, 3), dtype=np.uint8)
    img[:, :, 0] = im_flat
    img[:, :, 1] = im_flat
    img[:, :, 2] = im_flat

    region_properties = pyopia.process.measure_particles(imbw, max_particles=max_particles)

    # build the stats and export to HDF5
    stats = extractparticles_function(img, timestamp, Classification, region_properties)

    saturation = None

    return stats, imbw, saturation
