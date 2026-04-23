import numpy as np
import pytest

from pyopia.noise import ReduceNoise
from pyopia.pipeline import Pipeline


class _Data(dict):
    pass


def test_reduce_noise_gaussian_smooths_image():
    data = _Data()
    data['imraw'] = np.zeros((5, 5), dtype=float)
    data['imraw'][2, 2] = 1.0

    reducer = ReduceNoise(method='gaussian', image_source='imraw', output_key='im_denoised', gaussian_sigma=1.0)
    out = reducer(data)

    assert out['im_denoised'].shape == data['imraw'].shape
    assert out['im_denoised'][2, 2] < 1.0
    assert out['im_denoised'][2, 2] > 0.0


def test_reduce_noise_clahe_returns_contrast_enhanced_float_image():
    data = _Data()
    data['im_corrected'] = np.full((8, 8), 0.4, dtype=float)
    data['im_corrected'][2:6, 2:6] = 0.6

    reducer = ReduceNoise(method='clahe', image_source='im_corrected')
    out = reducer(data)

    assert out['im_denoised'].shape == data['im_corrected'].shape
    assert np.issubdtype(out['im_denoised'].dtype, np.floating)
    assert out['im_denoised'].min() >= 0.0
    assert out['im_denoised'].max() <= 1.0


def test_reduce_noise_invalid_method_raises_value_error():
    data = _Data(im_corrected=np.ones((4, 4), dtype=float))

    reducer = ReduceNoise(method='invalid', image_source='im_corrected')

    with pytest.raises(ValueError, match='Unknown noise reduction method'):
        reducer(data)


def test_reduce_noise_can_run_as_pipeline_step():
    settings = {
        'general': {'raw_files': None},
        'steps': {
            'noisereduction': {
                'pipeline_class': 'pyopia.noise.ReduceNoise',
                'method': 'gaussian',
                'image_source': 'im_corrected',
                'gaussian_sigma': 0.5,
            }
        }
    }
    pipeline = Pipeline(settings=settings, initial_steps='')
    pipeline.data['im_corrected'] = np.zeros((6, 6), dtype=float)
    pipeline.data['im_corrected'][3, 3] = 1.0

    pipeline.run_step('noisereduction')

    assert pipeline.data['im_denoised'][3, 3] < 1.0
    assert pipeline.data['im_corrected'][3, 3] == 1.0
