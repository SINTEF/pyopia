import numpy as np

from scipy.stats import lognorm
import matplotlib.pyplot as plt
from skimage.draw import disk, circle_perimeter
from scipy import stats
import scipy.ndimage

import pyopia.statistics
import pyopia.plotting

TotalVolumeConcentration = 1000
d50 = 1000
MinD = 10
PIX_SIZE = 28
PATH_LENGTH = 40
imx = 2048
imy = 2448

def wb(x, d50):
    a=2.8
    n = d50 * 1.5723270440251573
    return (a / n) * (x / n) ** (a - 1) * np.exp(-(x / n) ** a)

x = np.linspace(0, 12000, 10000)
plt.plot(x, wb(x, d50), 'k-')

diams, bin_limits_um = pyopia.statistics.get_size_bins()
vd = wb(diams, d50)
plt.plot(diams, vd, 'r--')

plt.xscale('log')
plt.xlim(0, 12000);

vd = vd / np.sum(vd) * TotalVolumeConcentration  # scale the distribution according to concentration

DropletVolume = ((4 / 3) * np.pi * ((diams * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
nd = vd / (DropletVolume * 1e9)  # the number distribution in each bin
nd[diams < MinD] = 0  # remove small particles for speed purposes

# calculate the sample volume of the SilCam specified
sv = pyopia.statistics.get_sample_volume(PIX_SIZE, path_length=PATH_LENGTH, imx=imx, imy=imy)

nd = nd * sv  # scale the number distribution by the sample volume so resulting units are #/L/bin
nc = int(sum(nd))  # calculate the total number concentration

vd2 = pyopia.statistics.vd_from_nd(nd, diams, sv)  # convert the number distribution to volume distribution in uL/L/bin
vc_initial = sum(vd2)  # obtain the resulting concentration, now having remove small particles

d50_theory = pyopia.statistics.d50_from_vd(vd2, diams)  # calculate the d50 in um

print(d50_theory, nc, vc_initial)

nims = 50  # the number of images to simulate
# preallocate variables
log_vd = np.zeros((nims, len(diams)))
cvd = np.zeros(nims)
cd50 = np.zeros(nims)

for I in range(nims):
    # randomly select a droplet radius from the input distribution
    rad = np.random.choice(diams / 2, size=nc, p=nd / sum(nd)) / PIX_SIZE  # radius is in pixels
    log_ecd = rad * 2 * PIX_SIZE  # log this size as a diameter in um
    
    necd, edges = np.histogram(log_ecd, bin_limits_um)  # count particles into number distribution
    log_vd[I, :] = pyopia.statistics.vd_from_nd(necd, diams, sv=sv)  # convert to volume distribution
    cvd[I] = np.sum(
        np.mean(log_vd[0:I, :], axis=0))  # calculated the cumulate volume distribution over image number
    cd50[I] = pyopia.statistics.d50_from_vd(np.mean(log_vd, axis=0), diams)  # calcualte the cumulate d50 over image number
    
    
def synthesize(diams, bin_limits_um, nd, imx, imy, PIX_SIZE, PATH_LENGTH):
    '''synthesize an image and measure droplets

    Args:
      diams   (array)                       :  size bins of the number distribution
      bin_limits_um (array)                 :  limits of the size bins where dias are the mid-points
      nd      (array)                       :  number of particles per size bin
      imx     (float)                       :  image width in pixels
      imy     (float)                       :  image height in pixels
      PIX_SIZE          (float)             :  pixel size of the setup [um]

    Returns:
      img (unit8)                         : segmented image from pysilcam
      log_vd (array)                      : a volume distribution of the randomly selected particles put into the
                                            synthetic image

    '''
    nc = int(sum(nd))  # number concentration

    # preallocate the image and logged volume distribution variables
    img = np.zeros((imx, imy, 3), dtype=np.uint8()) + 230  # scale the initial brightness down a bit
    log_ecd = np.zeros(nc)
    # randomly select a droplet radii from the input distribution
    rad = np.random.choice(diams / 2, size=nc, p=nd / sum(nd)) / PIX_SIZE  # radius is in pixels
    log_ecd = rad * 2 * PIX_SIZE  # log these sizes as a diameter in um
    for rad_ in rad:
        # randomly decide where to put particles within the image
        col = np.random.randint(1, high=imx - rad_)
        row = np.random.randint(1, high=imy - rad_)
        
        rr, cc = disk((col, row), rad_)  # make a cirle of the radius selected from the distribution
        img[rr, cc, :] = 0
        

    necd, edges = np.histogram(log_ecd, bin_limits_um)  # count the input diameters into a number distribution
    sv = pyopia.statistics.get_sample_volume(PIX_SIZE, path_length=PATH_LENGTH, imx=imx, imy=imy)
    log_vd = pyopia.statistics.vd_from_nd(necd, diams, sv=sv)  # convert to a volume distribution

    # add some noise to the synthesized image
    #img = np.uint8(255 * util.random_noise(np.float64(img) / 255), var=0.01 ** 2)

    img = np.uint8(img)  # convert to uint8
    return img, log_vd

img, log_vd_synth = synthesize(diams, bin_limits_um, nd, imx, imy, PIX_SIZE, PATH_LENGTH)
pyopia.plotting.show_image(img, PIX_SIZE)

plt.plot(diams, log_vd.T, '0.9', alpha=0.25)
plt.plot(diams, np.mean(log_vd.T, axis=1), 'r')
plt.plot(diams, log_vd_synth, 'b')
plt.plot(diams, vd2, 'k')
plt.xscale('log')