import os

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import map_utils

from sbi_bmode import sim_utils, spectra_utils

opj = os.path.join

imgdir = '/u/adriaand/project/so/20240521_sbi_bmode/debug_beta'

os.makedirs(imgdir, exist_ok=True)

seed = 0

#amp = 1.5
amp = 0.5
gamma = -2
#gamma = -3
#beta0 = -2
beta0 = -3
#beta0 = 0
lmax = 200
ainfo = curvedsky.alm_info(lmax)
minfo = map_utils.MapInfo.map_info_healpix(128)
freq_pivot_sync = 22e9

cl_beta_in = sim_utils.get_delta_beta_cl(amp, gamma, ainfo.lmax, 1, 1)
cl_beta_in[0] = beta0 ** 2 * 4 * np.pi

beta_sync = sim_utils.get_beta_map(minfo, ainfo, beta0, amp, gamma, seed, ell_0=1, ell_cutoff=1)


fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(beta_sync, hold=True, min=beta_sync.min(), max=beta_sync.max())
fig.savefig(opj(imgdir, f'beta_map'))
plt.close(fig)

print(beta_sync.mean())

cl_beta = hp.anafast(beta_sync)

print(cl_beta[0])

fig, ax = plt.subplots(dpi=300)
ax.plot(cl_beta_in)
ax.plot(cl_beta, ls='dashed')
ax.set_ylim(0.0001, 100)
ax.set_yscale('log')
ax.set_xlim(0, 100)
fig.savefig(opj(imgdir, 'cl_beta'))
plt.close(fig)

for freq in range(20, 350, 50):
    sed_map = spectra_utils.get_sed_sync(freq * 1e9, beta_sync, freq_pivot_sync)
    
    fig, ax = plt.subplots(dpi=300)
    plt.axes(ax)
    hp.mollview(sed_map, hold=True)
    fig.savefig(opj(imgdir, f'sed_map_{freq}'))
    plt.close(fig)
    

# Same for dust beta.

#amp = 1.5
amp = 1.3
gamma = -2.3
#gamma = -3
#beta0 = -2
beta0 = 1.59
#beta0 = 0
lmax = 200
ainfo = curvedsky.alm_info(lmax)
minfo = map_utils.MapInfo.map_info_healpix(128)
temp = 19.6
freq_pivot_dust = 353e9

cl_beta_in = sim_utils.get_delta_beta_cl(amp, gamma, ainfo.lmax, 1, 1)
cl_beta_in[0] = beta0 ** 2 * 4 * np.pi

beta_dust = sim_utils.get_beta_map(minfo, ainfo, beta0, amp, gamma, seed, ell_0=1, ell_cutoff=1)

fig, ax = plt.subplots(dpi=300)
plt.axes(ax)
hp.mollview(beta_dust, hold=True, min=beta_dust.min(), max=beta_dust.max())
fig.savefig(opj(imgdir, f'beta_dust_map'))
plt.close(fig)

cl_beta = hp.anafast(beta_dust)

fig, ax = plt.subplots(dpi=300)
ax.plot(cl_beta_in)
ax.plot(cl_beta, ls='dashed')
ax.set_ylim(0.0001, 100)
ax.set_yscale('log')
ax.set_xlim(0, 100)
fig.savefig(opj(imgdir, 'cl_beta_dust'))
plt.close(fig)

for freq in range(20, 350, 50):
    sed_map = spectra_utils.get_sed_dust(freq * 1e9, beta_dust, temp, freq_pivot_dust)
    
    fig, ax = plt.subplots(dpi=300)
    plt.axes(ax)
    hp.mollview(sed_map, hold=True)
    fig.savefig(opj(imgdir, f'dust_sed_map_{freq}'))
    plt.close(fig)
    
    
