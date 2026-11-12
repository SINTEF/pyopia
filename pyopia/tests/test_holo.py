'''
Tests for pyopia.instrument.holo focus-depth bookkeeping, using small synthetic stacks.
'''

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pyopia.instrument.holo as holo


def _stack_with_particle_at_plane(plane, n_planes=11):
    im_stack = np.zeros((40, 40, n_planes))
    im_stack[:, :, :] = 0.1
    im_stack[16:24, 16:24, plane] = 1.0
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
    im_stack[13:15, 13:15, 6] = 0.5  # only visible on the plane after the focus plane
    im_stack[25:27, 25:27, 4] = 0.5  # only visible on the plane before the focus plane
    bbox = (12, 12, 28, 28)

    im_single, ifocus = focus_function(im_stack, bbox, False)
    im_deep, _ = focus_function(im_stack, bbox, True)

    assert ifocus == 5
    assert (im_single[1:3, 1:3] == im_single[0, 0]).all()
    assert (im_deep[1:3, 1:3] > im_single[1:3, 1:3]).all()
    assert (im_deep[13:15, 13:15] > im_single[13:15, 13:15]).all()


def test_output_option_1_is_the_real_part_of_the_reconstructed_field():
    im = np.random.default_rng(0).normal(1, 0.05, (32, 48))
    kern = holo.create_kernel(im, 4.4, 658, 1.33, 27, 0, 1, 0.5)
    im_fft = holo.forward_transform(im, 2)

    stack_real = holo.inverse_transform(im_fft, kern, np.zeros(kern.shape), 1)
    stack_squared = holo.inverse_transform(im_fft, kern, np.zeros(kern.shape), 0)

    # the same field without the fftshift of the spectrum
    im_fft_unshifted = np.fft.fft2(im)
    im_fft_unshifted[:, 0] = 0
    im_fft_unshifted[0, :] = 0
    for i in range(kern.shape[2]):
        expected = np.fft.ifft2(im_fft_unshifted * np.fft.ifftshift(kern[:, :, i])).real
        np.testing.assert_allclose(stack_real[:, :, i], expected, atol=1e-12)
    np.testing.assert_allclose(stack_real**2, stack_squared, atol=1e-12)


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
