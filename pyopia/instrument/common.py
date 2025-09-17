"""
Non-instrument-specific functions that operates on the image loading or initial processing level.
"""

import numpy as np
from skimage.draw import disk


def apply_circular_mask(image, radius):
    """
    Apply a circular mask to an RGB image, zeroing out pixels outside the disc.

    Parameters
    ----------
    image: (np.array)
        numpy array of shape (h, w, 3)
    radius: int
        radius of the circular mask

    Returns
    -------
    masked_image:
        numpy array of shape (h, w, 3) with mask applied
    """
    h, w, _ = image.shape
    center = (h // 2, w // 2)

    # Create mask
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = disk(center, radius, shape=mask.shape)
    mask[rr, cc] = True

    # Expand mask to 3 channels
    mask_rgb = np.stack([mask] * 3, axis=-1)

    # Apply mask
    masked_image = image * mask_rgb

    return masked_image


class RectangularImageMask:
    """PyOpia pipline-compatible class for masking out part of the raw image.

        Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.imraw`

    Parameters
    ----------
    mask_bbox : (list, optional)
        Pixel corner coordinates of rectangle to mask (image outside the rectangle is set to 0)

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        containing the new key:

        :attr:`pyopia.pipeline.Data.im_masked`


    Example pipeline use:
    ----------------------
    Put this in your pipeline right after load step to mask out border outside specified pixel coordinates:

    .. code-block:: toml

        [steps.mask]
        pipeline_class = 'pyopia.instrument.common.RectangularImageMask'
        mask_bbox = [[200, 1850], [400, 2048], [0, 3]]

    The mask_bbox is [[start_row, end_row], [start_col, end_col], [start_colorchan, end_colorchan]]
    """

    def __init__(self, mask_bbox=None):
        if mask_bbox is None:
            self.mask_bbox = (slice(None), slice(None), slice(None))
        else:
            self.mask_bbox = (
                slice(mask_bbox[0][0], mask_bbox[0][1]),
                slice(mask_bbox[1][0], mask_bbox[1][1]),
                slice(mask_bbox[2][0], mask_bbox[2][1]),
            )

    def __call__(self, data):
        # Create a masked version of imraw, where space between defined mask rectangle and border is set to 0,
        # while inside is kept.
        imraw_masked = np.ones_like(data["imraw"])
        imraw_masked[self.mask_bbox] = data["imraw"][self.mask_bbox]
        data["im_masked"] = imraw_masked

        return data


class CircularImageMask:
    """PyOPIA pipline-compatible class for masking out part of the raw image with a circular centered disc

        Required keys in :class:`pyopia.pipeline.Data`:
        - :attr:`pyopia.pipeline.Data.imraw`

    Parameters
    ----------
    radius : (int)
        Radius in pixel of the circular disc mask (image outside disc is set to 0)

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        containing the new key:

        :attr:`pyopia.pipeline.Data.im_masked`


    Example pipeline use:
    ----------------------
    Put this in your pipeline right after load step to mask out border outside specified pixel coordinates:

    .. code-block:: toml

        [steps.mask]
        pipeline_class = 'pyopia.instrument.common.CircularImageMask'
        radius = 500

    The mask_bbox is [[start_row, end_row], [start_col, end_col], [start_colorchan, end_colorchan]]
    """

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, data):
        """
        Create a masked version of imraw, where the area outside a disc
        with given radius centered in the image is set to 0, while the inside is kept.
        """
        data["im_masked"] = apply_circular_mask(data["imraw"], self.radius)

        return data
