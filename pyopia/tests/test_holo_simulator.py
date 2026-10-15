'''
Tests for pyopia.simulator.holo, using a small simulated hologram.
'''

import numpy as np
import pandas as pd

import pyopia.instrument.holo as holo
from pyopia.simulator.holo import match_to_truth, random_particles, synthetic_hologram, write_synthetic_holograms


def test_synthetic_hologram_is_reproducible_and_keeps_mean_level():
    particles = random_particles((200, 200), 3, np.random.default_rng(1), margin=40, min_separation=60)

    first = synthetic_hologram((200, 200), particles, pad=64, rng=np.random.default_rng(2))
    second = synthetic_hologram((200, 200), particles, pad=64, rng=np.random.default_rng(2))

    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert abs(first.mean() - 160) < 2


def test_reconstruction_focuses_a_synthetic_particle_at_its_true_depth():
    true_depth = 20.0
    particles = np.array([[150.0, 150.0, true_depth, 100.0]])
    im = synthetic_hologram((300, 300), particles, pad=128, noise_counts=0) / 160.0

    kern = holo.create_kernel(im, 4.4, 658, 1.33, 27, 0, 50, 0.5)
    stack = holo.inverse_transform(holo.forward_transform(im, 2), kern, np.zeros(kern.shape))

    focus = stack[140:160, 140:160, :].sum(axis=(0, 1))
    assert abs(np.argmax(focus) * 0.5 - true_depth) <= 1.0


def test_write_synthetic_holograms_saves_images_and_matching_truth(tmp_path):
    kwargs = dict(shape=(200, 200), pad=64, margin=40, min_separation=60)

    truth = write_synthetic_holograms(tmp_path, 2, 3, **kwargs)

    assert sorted(f.name for f in tmp_path.glob('*.pgm')) == ['001-2001.pgm', '001-2002.pgm']
    assert list(truth.columns) == ['image', 'x_px', 'y_px', 'z_mm', 'diameter_um']
    assert truth['image'].value_counts().to_dict() == {'001-2001': 3, '001-2002': 3}
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'truth.csv'), truth)
    assert write_synthetic_holograms(tmp_path / 'again', 2, 3, **kwargs).equals(truth)


def test_match_to_truth_reports_errors_missed_particles_and_extra_detections():
    truth = pd.DataFrame({'image': ['a', 'a'], 'x_px': [100.0, 300.0], 'y_px': [100.0, 300.0],
                          'z_mm': [10.0, 20.0], 'diameter_um': [44.0, 88.0]})
    stats = pd.DataFrame({'holo_filename': ['/data/a.pgm', '/data/a.pgm'],
                          'minr': [95, 500], 'minc': [95, 500], 'maxr': [105, 510], 'maxc': [105, 510],
                          'equivalent_diameter': [11.0, 5.0], 'z': [10.5, 3.0]})

    matches, n_unmatched = match_to_truth(stats, truth, pixel_size=4.4)

    assert matches['found'].tolist() == [True, False]
    assert np.isclose(matches.loc[0, 'size_error'], (11.0 * 4.4 - 44.0) / 44.0)
    assert matches.loc[0, 'depth_error'] == 0.5
    assert np.isnan(matches.loc[1, 'size_error'])
    assert n_unmatched == 1


def test_match_to_truth_handles_stats_that_restart_their_index_for_each_image():
    truth = pd.DataFrame({'image': ['a', 'b'], 'x_px': [100.0, 100.0], 'y_px': [100.0, 100.0],
                          'z_mm': [10.0, 10.0], 'diameter_um': [44.0, 44.0]})
    stats = pd.DataFrame({'holo_filename': ['a.pgm', 'b.pgm'], 'minr': [95, 95], 'minc': [95, 95],
                          'maxr': [105, 105], 'maxc': [105, 105], 'equivalent_diameter': [10.0, 10.0], 'z': [10.0, 10.0]},
                         index=[0, 0])

    matches, n_unmatched = match_to_truth(stats, truth)

    assert matches['found'].tolist() == [True, True]
    assert n_unmatched == 0
