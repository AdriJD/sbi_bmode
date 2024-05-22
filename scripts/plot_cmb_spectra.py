import os

import numpy as np
import matplotlib.pyplot as plt

from sbi_bmode import spectra_utils

opj = os.path.join

datadir = '/mnt/home/aduivenvoorden/local/cca_project/data'
imgdir = '/mnt/home/aduivenvoorden/project/so/20240402_sbi'

os.makedirs(imgdir, exist_ok=True)

lmax = 200
ells = np.arange(lmax + 1)
dells = ells * (ells + 1) / 2 / np.pi
cov_ell_nobb = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_nobb.dat'), lmax)
cov_ell_r1 = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_r1.dat'), lmax)

fig, axs = plt.subplots(dpi=300, nrows=2, ncols=2)
for idxs, ax in np.ndenumerate(axs):
    ax.plot(ells, dells * cov_ell_nobb[idxs])
    ax.plot(ells, dells * cov_ell_r1[idxs] * 0.1)

fig.savefig(opj(imgdir, 'cmb_spectra'))
plt.close(fig)

# Dust.
amp = 1
alpha = -2.5
lmax = 200
beta = 1.5
temp = 19.6
freq_pivot = 353 * 1e9

fig, ax = plt.subplots(dpi=300)

for freq_ghz in [20, 40, 90, 150, 220]:
    cl_dust = spectra_utils.get_dust_spectra(amp, alpha, lmax, freq_ghz * 1e9, beta, temp, freq_pivot)

    ax.plot(ells, cl_dust, label=f'{freq_ghz} GHz')

    
ax.legend(frameon=False)    
ax.set_yscale('log')
fig.savefig(opj(imgdir, 'dust_spectra'))
plt.close(fig)

# Sync.
amp = 1
alpha = -2.8
lmax = 200
#freq = 150 * 1e9
beta = -2.7
freq_pivot = 22 * 1e9

fig, ax = plt.subplots(dpi=300)

for freq_ghz in [20, 40, 90, 150, 220]:

    cl_sync = spectra_utils.get_sync_spectra(amp, alpha, lmax, freq_ghz * 1e9, beta, freq_pivot)

    ax.plot(ells, cl_sync, label=f'{freq_ghz} GHz')

ax.legend(frameon=False)
ax.set_yscale('log')
fig.savefig(opj(imgdir, 'sync_spectra'))
plt.close(fig)
