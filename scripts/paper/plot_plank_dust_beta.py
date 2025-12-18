import os

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import map_utils

from sbi_bmode import sim_utils, spectra_utils

opj = os.path.join

idir = '/u/adriaand/project/planck/20250612_commander'
maskdir = '/u/adriaand/project/so/20250612_sat_mask'
maskdir_planck = '/u/adriaand/project/actpol/20250406_planck_fnl/masks'
imgdir = '/u/adriaand/project/so/20240521_sbi_bmode/planck_dust_beta'

os.makedirs(imgdir, exist_ok=True)

imap = hp.read_map(opj(idir, f'COM_CompMap_dust-commander_0256_R2.00.fits'), field=7)

fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(imap, hold=True)
fig.savefig(opj(imgdir, f'beta_mean'))
plt.close(fig)

imap_rms = hp.read_map(opj(idir, f'COM_CompMap_dust-commander_0256_R2.00.fits'), field=8)

fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(imap_rms, hold=True)
fig.savefig(opj(imgdir, f'beta_rms'))
plt.close(fig)

mask = hp.read_map(opj(maskdir, 'mask_apo10.0_MSS2_SAT1_f090_coadd_gal.fits'))
mask[mask>1e-4] = 1
print(np.sum(mask) / mask.size)
print(mask.size)

fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(mask, hold=True)
fig.savefig(opj(imgdir, f'mask'))
plt.close(fig)

mask_P = hp.read_map(opj(maskdir_planck, 'COM_Mask_CMB-common-Mask-Pol_2048_R3.00.fits'), field=None)

mask_comb = mask * mask_P

fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(mask_comb, hold=True)
fig.savefig(opj(imgdir, f'mask_combined'))
plt.close(fig)

mask_dg = hp.ud_grade(mask_comb, 256)
mask_dg = hp.sphtfunc.smoothing(mask_dg, fwhm=np.radians(1))

fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(mask_dg, hold=True)
fig.savefig(opj(imgdir, f'mask_dg'))
plt.close(fig)

cl_beta = hp.anafast(imap * mask_dg) / (np.sum(mask_dg) / mask_dg.size)

amp = 1.3
gamma = -2.3
beta0 = 1.6
lmax = 200
cl_beta_in = sim_utils.get_delta_beta_cl(amp, gamma, lmax, 1, 1)
cl_beta_in[0] = beta0 ** 2 * 4 * np.pi

fig, ax = plt.subplots(dpi=300)
ax.plot(cl_beta)
ax.plot(cl_beta_in, ls='dashed')
ax.set_ylim(0.0001, 100)
ax.set_yscale('log')
ax.set_xlim(-1, 100)
fig.savefig(opj(imgdir, 'cl_beta'))
plt.close(fig)
