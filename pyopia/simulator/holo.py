# -*- coding: utf-8 -*-
'''
Module containing tools for simulating in-line holograms of particles with known size and position
'''
import os

import numpy as np
import pandas as pd
import scipy.fft as fft
import skimage.io


def _transfer_function(shape, pixel_size, wavelength, distance):
    '''Band-limited angular spectrum transfer function for propagating a distance in a medium

    Parameters
    ----------
    shape : tuple
        (rows, columns) of the field
    pixel_size : float
        pixel size in metres
    wavelength : float
        wavelength in the propagation medium in metres
    distance : float
        propagation distance in metres

    Returns
    -------
    np.array
        complex transfer function, zero for evanescent and band-limited frequencies
    '''
    ny, nx = shape
    fx = fft.fftfreq(nx, pixel_size)[None, :]
    fy = fft.fftfreq(ny, pixel_size)[:, None]
    root = 1 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2
    propagating = root > 0
    h = np.exp(2j * np.pi * distance / wavelength * np.sqrt(np.where(propagating, root, 0)))

    # band limit avoids aliasing of the transfer function at long distances
    limit_x = 1 / (wavelength * np.sqrt((2 * distance / (nx * pixel_size)) ** 2 + 1))
    limit_y = 1 / (wavelength * np.sqrt((2 * distance / (ny * pixel_size)) ** 2 + 1))
    return h * propagating * (np.abs(fx) < limit_x) * (np.abs(fy) < limit_y)


def synthetic_hologram(shape, particles, pixel_size=4.4, wavelength=658, n=1.33, offset=27.0,
                       pad=512, mean_counts=160, noise_counts=1.0, rng=None):
    '''Simulate an in-line hologram of opaque circular particles

    Each particle is a disk illuminated by a plane wave. Its scattered field (Babinet's principle)
    is propagated to the sensor with the band-limited angular spectrum method [1]_, through the
    medium (wavelength / n over the particle depth) and then through the offset (as air). The
    fields of all particles are summed (single scattering) and the sensor records the intensity.

    Parameters
    ----------
    shape : tuple
        (rows, columns) of the hologram in pixels
    particles : array
        one row per particle: (x pixel, y pixel, depth in mm within the sample volume, diameter in microns)
    pixel_size : float
        pixel size in microns
    wavelength : float
        laser wavelength in nm
    n : float
        refractive index of the sample medium
    offset : float
        air-equivalent distance in mm from the sensor to the front of the sample volume
    pad : int
        pixels of zero padding on each side while propagating, to avoid wrap-around
    mean_counts : float
        mean grey level of the 8-bit hologram
    noise_counts : float
        standard deviation of added Gaussian noise in grey levels
    rng : np.random.Generator, optional
        random number generator for the noise

    Returns
    -------
    np.array
        8-bit hologram

    References
    ----------
    .. [1] Matsushima and Shimobaba, Optics Express 17, 19662 (2009)
    '''
    rng = rng if rng is not None else np.random.default_rng()
    ny, nx = shape
    pitch = pixel_size * 1e-6
    wavelength_m = wavelength * 1e-9
    padded = (ny + 2 * pad, nx + 2 * pad)

    h_offset = _transfer_function(padded, pitch, wavelength_m, offset * 1e-3)
    rows, cols = np.mgrid[0:padded[0], 0:padded[1]]

    scattered = np.zeros(padded, dtype=complex)
    for x, y, depth, diameter in particles:
        radius = diameter / pixel_size / 2
        x0, y0 = x + pad, y + pad
        half = int(radius + 3)
        window = (slice(int(y0) - half, int(y0) + half + 1), slice(int(x0) - half, int(x0) + half + 1))
        distance = np.hypot(cols[window] - x0, rows[window] - y0)
        disk = np.zeros(padded)
        disk[window] = np.clip(radius - distance + 0.5, 0, 1)  # anti-aliased edge

        h_medium = _transfer_function(padded, pitch, wavelength_m / n, depth * 1e-3)
        scattered += fft.ifft2(fft.fft2(disk) * h_offset * h_medium, workers=os.cpu_count())

    field = (1 - scattered)[pad:pad + ny, pad:pad + nx]
    image = np.abs(field) ** 2 * mean_counts + rng.normal(0, noise_counts, shape)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def random_particles(shape, count, rng, diameter_range=(20, 150), depth_range=(3, 47),
                     margin=80, min_separation=160):
    '''Random particles for :func:`synthetic_hologram`, log-uniform in size and uniform in position and depth

    Parameters
    ----------
    shape : tuple
        (rows, columns) of the hologram in pixels
    count : int
        number of particles
    rng : np.random.Generator
        random number generator
    diameter_range : tuple
        minimum and maximum diameter in microns
    depth_range : tuple
        minimum and maximum depth in mm within the sample volume
    margin : float
        minimum distance of particle centres from the image edge in pixels
    min_separation : float
        minimum distance between particle centres in pixels

    Returns
    -------
    np.array
        one row per particle: (x pixel, y pixel, depth in mm, diameter in microns)
    '''
    ny, nx = shape
    particles = []
    while len(particles) < count:
        x, y = rng.uniform(margin, nx - margin), rng.uniform(margin, ny - margin)
        if all(np.hypot(x - px, y - py) > min_separation for px, py, *_ in particles):
            diameter = np.exp(rng.uniform(np.log(diameter_range[0]), np.log(diameter_range[1])))
            particles.append((x, y, rng.uniform(*depth_range), diameter))
    return np.array(particles)


def write_synthetic_holograms(folder, n_images, particles_per_image, shape=(1200, 1600), seed=0, **kwargs):
    '''Write simulated holograms to a folder, with the true particles in truth.csv

    Images are named like the example holograms (``001-2001.pgm``, ``001-2002.pgm``, ...).

    Parameters
    ----------
    folder : str
        output folder (created if needed)
    n_images : int
        number of holograms
    particles_per_image : int
        number of particles in each hologram
    shape : tuple
        (rows, columns) of each hologram in pixels
    seed : int
        seed for the particles and noise
    **kwargs
        passed to :func:`synthetic_hologram`, and to :func:`random_particles` for
        diameter_range, depth_range, margin and min_separation

    Returns
    -------
    truth : DataFrame
        one row per particle with columns image, x_px (column), y_px (row),
        z_mm (depth within the sample volume) and diameter_um; also saved as truth.csv
    '''
    particle_args = {k: kwargs.pop(k) for k in ('diameter_range', 'depth_range', 'margin', 'min_separation') if k in kwargs}
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)

    rows = []
    for i in range(n_images):
        name = f'001-{2001 + i}'
        particles = random_particles(shape, particles_per_image, rng, **particle_args)
        image = synthetic_hologram(shape, particles, rng=rng, **kwargs)
        skimage.io.imsave(os.path.join(folder, name + '.pgm'), image, check_contrast=False)
        rows += [(name, *particle) for particle in particles]

    truth = pd.DataFrame(rows, columns=['image', 'x_px', 'y_px', 'z_mm', 'diameter_um'])
    truth.to_csv(os.path.join(folder, 'truth.csv'), index=False)
    return truth


def match_to_truth(stats, truth, pixel_size=4.4):
    '''Match processed particle statistics to the true particles of simulated holograms

    Each true particle is matched to the nearest unclaimed detection in the same image whose
    centre lies within the larger of 8 pixels or the particle radius.

    Parameters
    ----------
    stats : DataFrame
        particle statistics from processing, with columns holo_filename, minr, minc, maxr, maxc,
        equivalent_diameter (pixels) and z (mm)
    truth : DataFrame
        true particles, as returned by :func:`write_synthetic_holograms`
    pixel_size : float
        pixel size in microns

    Returns
    -------
    matches : DataFrame
        truth with added columns found, ecd_um (equivalent diameter), z_detected,
        size_error (relative to the true diameter) and depth_error (mm); nan where not found
    n_unmatched : int
        number of detections not matched to a true particle
    '''
    stats = stats.dropna(subset=['minr']).copy()
    stats['image'] = [os.path.splitext(os.path.basename(f))[0] for f in stats['holo_filename']]
    stats['x_px'] = (stats['minc'] + stats['maxc']) / 2
    stats['y_px'] = (stats['minr'] + stats['maxr']) / 2

    matches = truth.copy()
    for column in ('ecd_um', 'z_detected', 'size_error', 'depth_error'):
        matches[column] = np.nan
    matches['found'] = False

    claimed = set()
    for i, particle in truth.iterrows():
        candidates = stats[(stats['image'] == particle['image']) & ~stats.index.isin(claimed)]
        distance = np.hypot(candidates['x_px'] - particle['x_px'], candidates['y_px'] - particle['y_px'])
        close = distance[distance < max(8, particle['diameter_um'] / pixel_size / 2)]
        if close.empty:
            continue
        detection = candidates.loc[close.idxmin()]
        claimed.add(close.idxmin())
        ecd = detection['equivalent_diameter'] * pixel_size
        matches.loc[i, ['found', 'ecd_um', 'z_detected']] = [True, ecd, detection['z']]
        matches.loc[i, 'size_error'] = (ecd - particle['diameter_um']) / particle['diameter_um']
        matches.loc[i, 'depth_error'] = detection['z'] - particle['z_mm']

    return matches, len(stats) - len(claimed)
