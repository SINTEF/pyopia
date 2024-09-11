'''
Module containing tools for assessing statistical reliability of silcam size distributions
'''
import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
import skimage.util
import pandas as pd

import pyopia.statistics
import pyopia.plotting
import pyopia.process
import pyopia.instrument.silcam
from pyopia.pipeline import Pipeline


class SilcamSimulator():
    def __init__(self, total_volume_concentration=1000,
                 d50=1000,
                 MinD=10,
                 PIX_SIZE=28,
                 PATH_LENGTH=40,
                 imx=2048,
                 imy=2448,
                 nims=50):
        '''SilCam simulator

        Parameters
        ----------
        total_volume_concentration : int, optional
            total volume concentration, by default 1000
        d50 : int, optional
            median particle size, by default 1000
        MinD : int, optional
            minimum diameter to simulate, by default 10
        PIX_SIZE : int, optional
            pixel size (um), by default 28
        PATH_LENGTH : int, optional
            path length (mm), by default 40
        imx : int, optional
            image x dimension, by default 2048
        imy : int, optional
            image y dimension, by default 2448
        nims : int, optional
            number of images to simulate, by default 50

        Example:
        --------

        ```python
        from pyopia.simulator.silcam import SilcamSimulator

        sim = SilcamSimulator()
        sim.check_convergence()
        sim.synthesize()
        sim.process_synthetic_image()
        sim.plot()
        ```

        '''
        self.total_volume_concentration = total_volume_concentration
        self.d50 = d50
        self.MinD = MinD
        self.PIX_SIZE = PIX_SIZE
        self.PATH_LENGTH = PATH_LENGTH
        self.imx = imx
        self.imy = imy
        self.nims = nims

        self.dias, self.bin_limits = pyopia.statistics.get_size_bins()

        # calculate the sample volume of the SilCam specified
        self.sample_volume = pyopia.statistics.get_sample_volume(self.PIX_SIZE,
                                                                 path_length=self.PATH_LENGTH,
                                                                 imx=self.imx, imy=self.imy)

        self.data = dict()

    def weibull_distribution(self, x):
        '''calculate weibull distribution

        Parameters
        ----------
        x : array
            size bins of input

        Returns
        -------
        array
            weibull distribution
        '''
        a = 2.8
        n = self.d50 * 1.5723270440251573  # scaling required for the log-spaced size bins to match the input d50
        return (a / n) * (x / n) ** (a - 1) * np.exp(-(x / n) ** a)

    def check_convergence(self):
        '''Check statistical convergence of randomly selected size distributions
        over the `nims`number of images

        Attributes
        ----------
        data['volume_distribution'] : array
            volume distribution of shape (nims, dias)
        data['cumulative_volume_concentration'] : float
            cumulative mean volume concentration of length `nims`
        data['cumulative_d50'] : float
            cumulative average d50 of length `nims`
        '''
        self.data['weibull_x'] = np.linspace(np.min(self.dias), np.max(self.dias), 10000)
        self.data['weibull_y'] = self.weibull_distribution(self.data['weibull_x'])

        self.data['volume_distribution_input'] = self.weibull_distribution(self.dias)
        self.data['volume_distribution_input'] = self.data['volume_distribution_input'] / \
            np.sum(self.data['volume_distribution_input']) * \
            self.total_volume_concentration  # scale the distribution according to concentration

        DropletVolume = ((4 / 3) * np.pi * ((self.dias * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
        # the number distribution in each bin
        self.data['number_distribution'] = self.data['volume_distribution_input'] / (DropletVolume * 1e9)
        self.data['number_distribution'][self.dias < self.MinD] = 0  # remove small particles for speed purposes

        # scale the number distribution by the sample volume so resulting units are #/L/bin
        self.data['number_distribution'] = self.data['number_distribution'] * self.sample_volume
        nc = int(sum(self.data['number_distribution']))  # calculate the total number concentration. must be integer number

        # convert the number distribution to volume distribution in uL/L/bin
        vd2 = pyopia.statistics.vd_from_nd(self.data['number_distribution'], self.dias, self.sample_volume)

        # obtain the resulting concentration, now having remove small particles
        self.data['initial_volume_concentration'] = sum(vd2)

        # calculate the d50 in um
        self.data['d50_theoretical_best'] = pyopia.statistics.d50_from_vd(vd2, self.dias)

        # preallocate variables
        self.data['volume_distribution'] = np.zeros((self.nims, len(self.dias)))
        self.data['cumulative_volume_concentration'] = np.zeros(self.nims)
        self.data['cumulative_d50'] = np.zeros(self.nims)

        for i in range(self.nims):
            # randomly select a droplet radius from the input distribution
            # radius is in pixels
            rad = np.random.choice(self.dias / 2,
                                   size=nc,
                                   p=self.data['number_distribution'] / sum(self.data['number_distribution'])) / self.PIX_SIZE
            log_ecd = rad * 2 * self.PIX_SIZE  # log this size as a diameter in um

            necd, edges = np.histogram(log_ecd, self.bin_limits)  # count particles into number distribution

            # convert to volume distribution
            self.data['volume_distribution'][i, :] = pyopia.statistics.vd_from_nd(necd,
                                                                                  self.dias,
                                                                                  sample_volume=self.sample_volume)

            # calculated the cumulate volume distribution over image number
            self.data['cumulative_volume_concentration'][i] = np.sum(np.mean(self.data['volume_distribution'][0:i, :],
                                                                             axis=0))

            # calcualte the cumulate d50 over image number
            self.data['cumulative_d50'][i] = pyopia.statistics.d50_from_vd(np.mean(self.data['volume_distribution'],
                                                                                   axis=0),
                                                                           self.dias)

    def synthesize(self):
        '''Synthesize an image and measure droplets

        Attributes
        ----------
        data['synthetic_image_data']['image'] : array
            synthetic image
        data['synthetic_image_data']['input_volume_distribution'] : array
            Volume distribution used to create the synthetic image
        '''
        nc = int(sum(self.data['number_distribution']))  # number concentration

        # preallocate the image and logged volume distribution variables
        img = np.zeros((self.imx, self.imy, 3), dtype=np.uint8()) + 230  # scale the initial brightness down a bit
        log_ecd = np.zeros(nc)
        # randomly select a droplet radii from the input distribution
        # radius is in pixels
        rad = np.random.choice(self.dias / 2,
                               size=nc,
                               p=self.data['number_distribution'] / sum(self.data['number_distribution'])) / self.PIX_SIZE
        log_ecd = rad * 2 * self.PIX_SIZE  # log these sizes as a diameter in um
        for rad_ in rad:
            # randomly decide where to put particles within the image
            col = np.random.randint(1, high=self.imx - rad_)
            row = np.random.randint(1, high=self.imy - rad_)

            rr, cc = disk((col, row), rad_)  # make a cirle of the radius selected from the distribution
            img[rr, cc, :] = 0

        necd, edges = np.histogram(log_ecd, self.bin_limits)  # count the input diameters into a number distribution
        # convert to a volume distribution
        log_vd = pyopia.statistics.vd_from_nd(necd, self.dias, sample_volume=self.sample_volume)

        # add some noise to the synthesized image
        img = np.uint8(255 * skimage.util.random_noise(np.float64(img) / 255))

        img = np.uint8(img)  # convert to uint8
        self.data['synthetic_image_data'] = dict()
        self.data['synthetic_image_data']['image'] = img
        self.data['synthetic_image_data']['input_volume_distribution'] = log_vd

    def process_synthetic_image(self):
        '''Put the synthetic image `data['synthetic_image_data']['image']` through a basic pyopia processing pipeline

        Attributes
        ----------
        data['synthetic_image_data']['pyopia_processed_volume_distribution'] : array
            pyopia processed volume distribution associated with `dias`size classes
        '''
        pipeline_config = {
            'general': {
                'raw_files': '',
                'pixel_size': 28  # pixel size in um
            },
            'steps': {
                'imageprep': {
                    'pipeline_class': 'pyopia.instrument.silcam.ImagePrep',
                    'image_level': 'im_synthetic'
                },
                'segmentation': {
                    'pipeline_class': 'pyopia.process.Segment',
                    'threshold': 0.85,
                    'segment_source': 'im_minimum'
                },
                'statextract': {
                    'pipeline_class': 'pyopia.process.CalculateStats',
                    'roi_source': 'im_synthetic'
                }
            }
        }
        pipeline = Pipeline(pipeline_config)
        pipeline.data['im_synthetic'] = self.data['synthetic_image_data']['image']
        pipeline.data['timestamp'] = pd.Timestamp.now()
        pipeline.run('')
        dias, vd = pyopia.statistics.vd_from_stats(pipeline.data['stats'], pipeline_config['general']['pixel_size'])
        vd /= self.sample_volume
        self.data['synthetic_image_data']['pyopia_processed_volume_distribution'] = vd

    def plot(self):
        f, a = plt.subplots(2, 2, figsize=(15, 10))

        plt.sca(a[0, 0])
        pyopia.plotting.show_image(self.data['synthetic_image_data']['image'], self.PIX_SIZE)
        plt.title(f'Synthetic image. Path lengh: {self.PATH_LENGTH}')

        plt.sca(a[0, 1])
        plt.plot(self.dias, self.data['volume_distribution'].T, '0.8', alpha=0.2)
        plt.plot(-10, 0, '0.8', alpha=0.2, label='Simulated')
        plt.plot(self.dias, np.mean(self.data['volume_distribution'].T, axis=1), 'k', label=f'{self.nims} statistical average')
        plt.plot(self.dias, self.data['synthetic_image_data']['input_volume_distribution'], 'b',
                 label='Best possible from synthetic image\n(without occlusion)')
        plt.plot(self.dias, self.data['synthetic_image_data']['pyopia_processed_volume_distribution'], 'g',
                 label='PyOPIA processed from synthetic image\n')
        plt.plot(self.dias, self.data['volume_distribution_input'], 'r', label='target')
        plt.xscale('log')
        plt.xlabel('Diameter [um]')
        plt.ylabel('Volume concentration [uL/L]')
        plt.legend()
        plt.xlim(np.min(self.dias), np.max(self.dias))

        plt.sca(a[1, 0])
        plt.plot(self.data['cumulative_volume_concentration'], '0.8', label='simulated')
        plt.hlines(self.total_volume_concentration, xmin=0, xmax=self.nims, colors='r', label='target')
        plt.xlabel('n-images')
        plt.ylabel('Volume concentration of n-images [uL/L]')
        plt.xlim(0, self.nims)
        plt.legend()

        plt.sca(a[1, 1])
        plt.plot(self.data['cumulative_d50'], '0.8', label='simulated')
        plt.hlines(self.d50, xmin=0, xmax=self.nims, colors='r', label='target')
        plt.xlabel('n-images')
        plt.ylabel('D50 over n-images [um]')
        plt.xlim(0, self.nims)
        plt.legend()

        plt.tight_layout()
