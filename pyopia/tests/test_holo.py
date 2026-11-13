'''
Tests for pyopia.instrument.holo focus-depth bookkeeping, using small synthetic stacks.
'''

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from skimage.filters import sobel

import pyopia.instrument.holo as holo


def _stack_with_particle_at_plane(plane, n_planes=11):
    im_stack = np.zeros((n_planes, 40, 40))
    im_stack[:, :, :] = 0.1
    im_stack[plane, 16:24, 16:24] = 1.0
    return im_stack


def _focus(im_stack):
    focus = holo.Focus(stacksummary_function='max_map', threshold=0.97, focus_function='find_focus_imax')
    return focus({'im_stack': im_stack})


def test_focus_discards_particles_at_either_end_of_the_stack():
    assert _focus(_stack_with_particle_at_plane(5))['stack_ifocus'] == [5]
    assert _focus(_stack_with_particle_at_plane(0))['stack_ifocus'] == []
    assert _focus(_stack_with_particle_at_plane(10))['stack_ifocus'] == []


@pytest.mark.parametrize('focus_function', [holo.find_focus_imax, holo.find_focus_sobel])
def test_increased_depth_of_field_includes_both_planes_either_side_of_focus(focus_function):
    im_stack = _stack_with_particle_at_plane(5)
    im_stack[6, 13:15, 13:15] = 0.5  # only visible on the plane after the focus plane
    im_stack[4, 25:27, 25:27] = 0.5  # only visible on the plane before the focus plane
    bbox = (12, 12, 28, 28)

    im_single, ifocus = focus_function(im_stack, bbox, False)
    im_deep, _ = focus_function(im_stack, bbox, True)

    assert ifocus == 5
    assert (im_single[1:3, 1:3] == im_single[0, 0]).all()
    assert (im_deep[1:3, 1:3] > im_single[1:3, 1:3]).all()
    assert (im_deep[13:15, 13:15] > im_single[13:15, 13:15]).all()


RECON_PARAMS = dict(wavelength=658, n=1.33, offset=27, minZ=0, maxZ=1, stepZ=0.5)
PIXEL_SIZE = 4.4


def _reconstruct(im, **reconstruct_args):
    data = {
        'im_corrected': im,
        'settings': {'general': {'pixel_size': PIXEL_SIZE}},
        'holo_recon_params': RECON_PARAMS,
    }
    return holo.Reconstruct(**reconstruct_args)(data)['im_stack']


def _expected_stack(im, forward_filter_option, inverse_output_option):
    '''Real part of the propagated field, plane by plane, on the unshifted frequency grid'''
    spectrum = np.fft.fft2(im)
    if forward_filter_option == 1:
        spectrum[0, 0] = 0
    elif forward_filter_option == 2:
        spectrum[:, 0] = 0
        spectrum[0, :] = 0

    fy = np.fft.fftfreq(im.shape[0], 1 / im.shape[0])[:, None] / (im.shape[0] / 2)
    fx = np.fft.fftfreq(im.shape[1], 1 / im.shape[1])[None, :] / (im.shape[1] / 2)
    f_squared = (np.pi / (PIXEL_SIZE / 1e6))**2 * (fx**2 + fy**2)
    k = 2 * np.pi / (RECON_PARAMS['wavelength'] * 1e-9)
    z = np.arange(RECON_PARAMS['minZ'], RECON_PARAMS['maxZ'] + RECON_PARAMS['stepZ'], RECON_PARAMS['stepZ'])
    z = z * 1e-3 / RECON_PARAMS['n'] + RECON_PARAMS['offset'] * 1e-3

    planes = [np.fft.ifft2(spectrum * np.exp(-1j * f_squared * z_ / (2 * k))).real for z_ in z]
    stack = np.stack(planes, axis=0)
    return stack**2 if inverse_output_option == 0 else stack


@pytest.mark.parametrize('dtype, atol', [(np.float64, 1e-10), (np.float32, 5e-6)])
@pytest.mark.parametrize('shape', [(64, 96), (65, 97)])
def test_kernel_recurrence_matches_exact_cosines_over_several_blocks(shape, dtype, atol):
    kern = holo.create_kernel(shape, PIXEL_SIZE, **{**RECON_PARAMS, 'maxZ': 20}, dtype=dtype)

    fy = (np.arange(shape[0] // 2 + 1) / (shape[0] / 2))[:, None]
    fx = (np.arange(shape[1] // 2 + 1) / (shape[1] / 2))[None, :]
    f_squared = (np.pi / (PIXEL_SIZE / 1e6))**2 * (fx**2 + fy**2)
    k = 2 * np.pi / (RECON_PARAMS['wavelength'] * 1e-9)
    z = np.arange(0, 20 + RECON_PARAMS['stepZ'], RECON_PARAMS['stepZ']) * 1e-3 / RECON_PARAMS['n'] + RECON_PARAMS['offset'] * 1e-3

    n_planes = 0
    for plane, z_ in zip(kern, z):  # a plane is only valid until two more have been requested
        assert plane.dtype == dtype
        np.testing.assert_allclose(plane, np.cos(f_squared * z_ / (2 * k)), atol=atol)
        n_planes += 1
    assert n_planes == len(kern) == len(z) == 41


@pytest.mark.parametrize('dtype, atol', [('float64', 1e-12), ('float32', 5e-6)])
@pytest.mark.parametrize('shape', [(32, 48), (33, 47)])
@pytest.mark.parametrize('inverse_output_option', [0, 1])
@pytest.mark.parametrize('forward_filter_option', [0, 1, 2])
def test_reconstruct_matches_an_independent_calculation(shape, forward_filter_option, inverse_output_option, dtype, atol):
    im = np.random.default_rng(0).normal(1, 0.05, shape)

    stack = _reconstruct(im, forward_filter_option=forward_filter_option, inverse_output_option=inverse_output_option,
                         dtype=dtype)

    expected = _expected_stack(im, forward_filter_option, inverse_output_option)
    assert stack.dtype == dtype
    np.testing.assert_allclose(stack, expected, atol=atol)


def test_reconstruct_with_threads_matches_the_serial_stack():
    im = np.random.default_rng(0).normal(1, 0.05, (48, 64))
    params = {**RECON_PARAMS, 'maxZ': 20}

    def stack(workers):
        data = {'im_corrected': im, 'settings': {'general': {'pixel_size': PIXEL_SIZE}}, 'holo_recon_params': params}
        return holo.Reconstruct(forward_filter_option=2, workers=workers)(data)['im_stack']

    assert stack(1).shape[0] == 41
    np.testing.assert_allclose(stack(4), stack(1), atol=1e-10)


def test_reconstruct_rejects_fewer_than_one_worker():
    with pytest.raises(ValueError, match='workers'):
        holo.Reconstruct(workers=0)


def test_reconstruct_rejects_other_dtypes():
    with pytest.raises(ValueError, match='float16'):
        holo.Reconstruct(dtype='float16')


def test_reconstruct_stack_clean_zeroes_pixels_below_a_fraction_of_the_maximum():
    im = np.random.default_rng(0).normal(1, 0.05, (32, 48))

    stack = _reconstruct(im, forward_filter_option=2, stack_clean=0.02)

    expected = _expected_stack(im, 2, 0)
    expected[expected < expected.max() * 0.02] = 0
    np.testing.assert_allclose(stack, expected, atol=1e-12)
    assert (stack == 0).any()


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (7, 1), (3, 3), (40, 50)])
def test_sobel_stack_equals_skimage_sobel_on_each_plane(shape, dtype):
    im_stack = np.random.default_rng(0).random((5, *shape)).astype(dtype)

    result = holo.sobel_stack(im_stack)

    assert result.dtype == dtype
    assert np.array_equal(result, np.stack([sobel(plane) for plane in im_stack]))


@pytest.mark.parametrize('increase_depth_of_field', [False, True])
def test_find_focus_sobel_does_not_depend_on_the_chunk_size(monkeypatch, increase_depth_of_field):
    im_stack = np.random.default_rng(0).random((9, 40, 50))
    im_stack[4, 10:30, 10:30] += 3

    im_single, ifocus_single = holo.find_focus_sobel(im_stack, (5, 5, 35, 45), increase_depth_of_field)
    monkeypatch.setattr(holo, '_SOBEL_CHUNK_PIXELS', 1)
    im_chunked, ifocus_chunked = holo.find_focus_sobel(im_stack, (5, 5, 35, 45), increase_depth_of_field)

    assert ifocus_single == ifocus_chunked == 4
    assert np.array_equal(im_single, im_chunked)


def test_mergestats_reports_the_depth_of_the_focus_plane():
    stats = pd.DataFrame({'minr': [16], 'minc': [16], 'maxr': [24], 'maxc': [24]})
    data = {
        'stats': stats,
        'stack_rp': [SimpleNamespace(bbox=(16, 16, 24, 24))],
        'stack_ifocus': [4],
        'filename': 'holo.pgm',
        'settings': {'steps': {'initial': {'minZ': 0, 'maxZ': 5, 'stepZ': 0.5}}},
    }

    z = holo.MergeStats()(data)['stats']['z'].iloc[0]

    assert z == 2.0  # plane index 4 of 0, 0.5, 1.0, ...
