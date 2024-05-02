import os

import numpy as np
import matplotlib.pyplot as plt

from sbi_bmode import spectra_utils, so_utils

opj = os.path.join

datadir = '/mnt/home/aduivenvoorden/local/cca_project/data'
imgdir = '/mnt/home/aduivenvoorden/project/so/20240402_sbi'

os.makedirs(imgdir, exist_ok=True)

lmax = 1000
ells = np.arange(lmax + 1)
dells = ells * (ells + 1) / 2 / np.pi

cov_ell_nobb = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_nobb.dat'), lmax)

freq_strings = ['f030', 'f040', 'f090', 'f150', 'f230', 'f290']

sensitivity_mode = 'goal'
lknee_mode = 'optimistic'
fsky = 0.1

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(ells, dells * cov_ell_nobb[1,1], color='black', label='lensing')

for fidx, fstr in enumerate(freq_strings):

    ax.plot(ells, dells * so_utils.get_sat_noise(
        fstr, sensitivity_mode, lknee_mode, fsky, lmax), label=fstr)

ax.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(10, 1000)
ax.set_ylim(1e-5, 2*1e-1)
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'noise_spectra'))
plt.close(fig)
