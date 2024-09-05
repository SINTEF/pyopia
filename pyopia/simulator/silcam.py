import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
import skimage.util

import pyopia.statistics
import pyopia.plotting


class SilcamSimulator():
    def __init__(self) -> None:
        self.total_volume_concentration = 1000
        self.d50 = 1000
        self.MinD = 10
        self.PIX_SIZE = 28
        self.PATH_LENGTH = 40
        self.imx = 2048
        self.imy = 2448
        self.nims = 50  # the number of images to simulate

        self.dias, self.bin_limits = pyopia.statistics.get_size_bins()

        # calculate the sample volume of the SilCam specified
        self.sample_volume = pyopia.statistics.get_sample_volume(self.PIX_SIZE,
                                                                 path_length=self.PATH_LENGTH,
                                                                 imx=self.imx, imy=self.imy)

        self.data = dict()
        self.data['weibull_x'] = np.linspace(np.min(self.dias), np.max(self.dias), 10000)
        self.data['weibull_y'] = self.weibull_distribution(self.data['weibull_x'])

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
        self.data['volume_distribution_input'] = self.weibull_distribution(self.dias)
        self.data['volume_distribution_input'] = self.data['volume_distribution_input'] / \
            np.sum(self.data['volume_distribution_input']) * \
            self.total_volume_concentration  # scale the distribution according to concentration

        DropletVolume = ((4 / 3) * np.pi * ((self.dias * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
        self.data['number_distribution'] = self.data['volume_distribution_input'] / (DropletVolume * 1e9)  # the number distribution in each bin
        self.data['number_distribution'][self.dias < self.MinD] = 0  # remove small particles for speed purposes

        self.data['number_distribution'] = self.data['number_distribution'] * self.sample_volume  # scale the number distribution by the sample volume so resulting units are #/L/bin
        nc = int(sum(self.data['number_distribution']))  # calculate the total number concentration. must be integer number
        print(nc)

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
            rad = np.random.choice(self.dias / 2, size=nc, p=self.data['number_distribution'] / sum(self.data['number_distribution'])) / self.PIX_SIZE  # radius is in pixels
            log_ecd = rad * 2 * self.PIX_SIZE  # log this size as a diameter in um

            necd, edges = np.histogram(log_ecd, self.bin_limits)  # count particles into number distribution

            # convert to volume distribution
            self.data['volume_distribution'][i, :] = pyopia.statistics.vd_from_nd(necd,
                                                                                  self.dias,
                                                                                  sv=self.sample_volume)

            # calculated the cumulate volume distribution over image number
            self.data['cumulative_volume_concentration'][i] = np.sum(np.mean(self.data['volume_distribution'][0:i, :],
                                                                             axis=0))

            # calcualte the cumulate d50 over image number
            self.data['cumulative_d50'][i] = pyopia.statistics.d50_from_vd(np.mean(self.data['volume_distribution'],
                                                                                   axis=0),
                                                                           self.dias)

    def synthesize(self):
        '''synthesize an image and measure droplets
        '''
        nc = int(sum(self.data['number_distribution']))  # number concentration

        # preallocate the image and logged volume distribution variables
        img = np.zeros((self.imx, self.imy, 3), dtype=np.uint8()) + 230  # scale the initial brightness down a bit
        log_ecd = np.zeros(nc)
        # randomly select a droplet radii from the input distribution
        rad = np.random.choice(self.dias / 2, size=nc, p=self.data['number_distribution'] / sum(self.data['number_distribution'])) / self.PIX_SIZE  # radius is in pixels
        log_ecd = rad * 2 * self.PIX_SIZE  # log these sizes as a diameter in um
        for rad_ in rad:
            # randomly decide where to put particles within the image
            col = np.random.randint(1, high=self.imx - rad_)
            row = np.random.randint(1, high=self.imy - rad_)

            rr, cc = disk((col, row), rad_)  # make a cirle of the radius selected from the distribution
            img[rr, cc, :] = 0

        necd, edges = np.histogram(log_ecd, self.bin_limits)  # count the input diameters into a number distribution
        log_vd = pyopia.statistics.vd_from_nd(necd, self.dias, sv=self.sample_volume)  # convert to a volume distribution

        # add some noise to the synthesized image
        img = np.uint8(255 * skimage.util.random_noise(np.float64(img) / 255))

        img = np.uint8(img)  # convert to uint8
        self.data['synthetic_image_data'] = dict()
        self.data['synthetic_image_data']['image'] = img
        self.data['synthetic_image_data']['volume_distribution'] = log_vd

    def plot(self):
        f, a = plt.subplots(2, 2, figsize=(10, 10))

        plt.sca(a[0, 0])
        pyopia.plotting.show_image(self.data['synthetic_image_data']['image'], self.PIX_SIZE)
        plt.title(f'synthetic image. Path lengh: {self.PATH_LENGTH}')

        plt.sca(a[0, 1])
        plt.plot(self.dias, self.data['volume_distribution'].T, '0.8', alpha=0.2)
        plt.plot(-10, 0, '0.8', alpha=0.2, label='simulated')
        plt.plot(self.dias, np.mean(self.data['volume_distribution'].T, axis=1), 'k', label=f'{self.nims} image average')
        plt.plot(self.dias, self.data['synthetic_image_data']['volume_distribution'], 'b',
                 label='best possible from synthetic image\n(without occlusion)')
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
        plt.ylabel('volume concentration of n-images [uL/L]')
        plt.xlim(0, self.nims)
        plt.legend()

        plt.sca(a[1, 1])
        plt.plot(self.data['cumulative_d50'], '0.8', label='simulated')
        plt.hlines(self.d50, xmin=0, xmax=self.nims, colors='r', label='target')
        plt.xlabel('n-images')
        plt.ylabel('d50 over n-images [um]')
        plt.xlim(0, self.nims)
        plt.legend()

        plt.tight_layout()
