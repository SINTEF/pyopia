'''
Tests for pyopia.instrument.holo focus-depth bookkeeping, using small synthetic stacks.
'''

from types import SimpleNamespace

import numpy as np
import pandas as pd

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
