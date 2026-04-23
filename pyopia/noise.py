'''
Noise reduction module for optional preprocessing steps in the pipeline.
'''
from skimage import exposure
from skimage import filters


class ReduceNoise():
    '''
    :class:`pyopia.pipeline` compatible class for optional noise reduction.

    Required keys in :class:`pyopia.pipeline.Data`:
        - key given by ``image_source``

    Parameters
    ----------
    method : str, optional
        Noise reduction method. Supported values are ``'gaussian'`` and ``'clahe'``.
        Defaults to ``'gaussian'``.
    image_source : str, optional
        Key in Pipeline.data containing the input image. Defaults to ``'im_corrected'``.
    output_key : str | None, optional
        Key where the filtered image will be stored. If ``None``, the image is updated in place
        in ``'im_denoised'``. Defaults to ``None``.
    gaussian_sigma : float, optional
        Gaussian sigma when ``method='gaussian'``. Defaults to ``1.0``.
    clahe_clip_limit : float, optional
        CLAHE clip limit when ``method='clahe'``. Defaults to ``0.01``.
    clahe_nbins : int, optional
        Number of bins for CLAHE when ``method='clahe'``. Defaults to ``256``.

    Returns
    -------
    data : :class:`pyopia.pipeline.Data`
        containing filtered image in ``output_key``.

    Example pipeline use
    --------------------

    .. code-block:: toml

        [steps.noisereduction]
        pipeline_class = 'pyopia.noise.ReduceNoise'
        method = 'gaussian'
        image_source = 'imraw'
        output_key = 'im_denoised'
        gaussian_sigma = 1.0
    '''

    def __init__(self,
                 method='gaussian',
                 image_source='im_corrected',
                 output_key=None,
                 gaussian_sigma=1.0,
                 clahe_clip_limit=0.01,
                 clahe_nbins=256):

        self.method = method
        self.image_source = image_source
        self.output_key = 'im_denoised' if output_key is None else output_key
        self.gaussian_sigma = gaussian_sigma
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_nbins = clahe_nbins

    def __call__(self, data):
        im = data[self.image_source]

        if self.method == 'gaussian':
            data[self.output_key] = filters.gaussian(im,
                                                     sigma=self.gaussian_sigma,
                                                     preserve_range=True,
                                                     channel_axis=-1 if im.ndim == 3 else None)
        elif self.method == 'clahe':
            data[self.output_key] = exposure.equalize_adapthist(im,
                                                                clip_limit=self.clahe_clip_limit,
                                                                nbins=self.clahe_nbins)
        else:
            raise ValueError(f"Unknown noise reduction method '{self.method}'. "
                             "Expected one of: 'gaussian', 'clahe'.")

        return data
