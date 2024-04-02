import os

import numpy as np
import matplotlib.pyplot as plt
from pixell import curvedsky
from optweight import map_utils

from sbi_bmode import spectra_utils, sim_utils

opj = os.path.join

datadir = '/mnt/home/aduivenvoorden/local/cca_project/data'
imgdir = '/mnt/home/aduivenvoorden/project/so/20240402_sbi'

os.makedirs(imgdir, exist_ok=True)

lmax = 200
lmin = 30
delta_ell = 10

bins = np.arange(lmin, lmax, delta_ell)
print(bins)
nside = 128
cov_scalar_ell = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_nobb.dat'), lmax)
cov_tensor_ell = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_r1.dat'), lmax)
minfo = map_utils.MapInfo.map_info_healpix(nside)
ainfo = curvedsky.alm_info(lmax)

temp_dust = 19.6
freqs = [27, 39, 93, 145, 225, 280]
freq_pivot_dust = 353
seed = 0
nsplit = 1
noise_cov_ell = np.ones((2, 2, lmax + 1)) * np.eye(2)[:,:,np.newaxis] * 0

A_d_BB = 5
alpha_d_BB = 1
beta_dust = 1
r_tensor = 0
A_lens = 1

seed = np.random.default_rng(seed=0)

omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                          r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                          cov_scalar_ell, cov_tensor_ell, minfo, ainfo)
spectra = sim_utils.estimate_spectra(omap, minfo, ainfo)

print(spectra.shape)


data = sim_utils.get_final_data_vector(spectra, bins, lmin, lmax)

print(data.shape)

fig, ax = plt.subplots(dpi=300)
ax.plot(data)
fig.savefig(opj(imgdir, 'data'))
plt.close(fig)
