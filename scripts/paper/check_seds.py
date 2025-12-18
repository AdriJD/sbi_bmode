import os

import matplotlib.pyplot as plt
import numpy as np
from pixell import curvedsky
from optweight import map_utils, sht

from sbi_bmode import sim_utils

opj = os.path.join

imgdir = '/u/adriaand/project/so/20240521_sbi_bmode/debug_sync'
os.makedirs(imgdir, exist_ok=True)

A_d_BB = 0
alpha_d_BB = -0.3
beta_dust = 1.6
amp_beta_dust = 0
gamma_beta_dust = -4

A_s_BB = 1
alpha_s_BB = -0.7
beta_sync = -3
amp_beta_sync = 0
gamma_beta_sync = -4
rho_ds = 0

freq_pivot_dust = 353e9
temp_dust = 19.6
freq_pivot_sync = 23e9
r_tensor = 0
A_lens = 0

lmax = 200
cov_noise_ell = np.zeros((8, 2, 2, lmax + 1))
cov_scalar_ell = np.zeros((2, 2, lmax + 1))
cov_tensor_ell = np.zeros((2, 2, lmax + 1))
b_ells = np.ones((8, lmax + 1))
minfo = map_utils.MapInfo.map_info_healpix(128)
ainfo = curvedsky.alm_info(lmax)
nsplit = 1
freqs = np.asarray([25, 27, 39, 93, 145, 225, 280, 340]) * 1e9
seed = np.random.default_rng(0)

odict = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                           r_tensor, A_lens, freqs, seed, nsplit, cov_noise_ell,
                           cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
                           amp_beta_dust=amp_beta_dust, gamma_beta_dust=gamma_beta_dust, A_s_BB=A_s_BB,
                           alpha_s_BB=alpha_s_BB, beta_sync=beta_sync, freq_pivot_sync=freq_pivot_sync,
                           amp_beta_sync=amp_beta_sync, gamma_beta_sync=gamma_beta_sync, rho_ds=rho_ds)

data = odict['data']
print(data.shape)
cls = np.zeros((len(freqs), lmax+1))
for fidx in range(freqs.size):
    alm = np.zeros((2, ainfo.nelem), dtype=np.complex128)
    sht.map2alm(data[0,fidx,...], alm, minfo, ainfo, 2)

    cls[fidx] = ainfo.alm2cl(alm[1])

print(cls)
    
fig, ax = plt.subplots(dpi=300)
for fidx, freq in enumerate(freqs):
    ax.plot(cls[fidx], label=int(freq * 1e-9))
ax.legend(frameon=False)
ax.set_yscale('log')
fig.savefig(opj(imgdir, 'cls_sync'))
plt.close(fig)

exp_sed = np.zeros(len(freqs))
for fidx, freq in enumerate(freqs):
    xx = freq * 6.62607015e-34 / (2.725 * 1.380649e-23)
    sed = (freq / freq_pivot_sync) ** (beta_sync)
    sed *= (np.exp(xx) - 1) ** 2
    sed /= (xx ** 2 * np.exp(xx))
    exp_sed[fidx] = sed

pyilc_sed = np.zeros(len(freqs))
for fidx, freq in enumerate(freqs):
    xx = freq * 6.62607015e-34 / (2.725 * 1.380649e-23)
    sed = (freq / freq_pivot_sync) ** (beta_sync)
    sed *= (np.exp(xx) - 1) ** 2
    sed /= (xx ** 4 * np.exp(xx))
    pyilc_sed[fidx] = sed

pyilc_sed_plus2 = np.zeros(len(freqs))
for fidx, freq in enumerate(freqs):
    xx = freq * 6.62607015e-34 / (2.725 * 1.380649e-23)
    sed = (freq / freq_pivot_sync) ** (beta_sync + 2)
    sed *= (np.exp(xx) - 1) ** 2
    sed /= (xx ** 4 * np.exp(xx))
    pyilc_sed_plus2[fidx] = sed
    
fig, ax = plt.subplots(dpi=300)
# Random pixel in Q map.
ax.plot(freqs * 1e-9, data[0,:,0,80] / data[0,0,0,80], label='sim')
ax.plot(freqs * 1e-9, (exp_sed / exp_sed[0]), label='expected', ls='dashed')
ax.plot(freqs * 1e-9, (pyilc_sed / pyilc_sed[0]), label='pyilc beta', ls='dotted')
ax.plot(freqs * 1e-9, (pyilc_sed_plus2 / pyilc_sed_plus2[0]), label='pyilc beta + 2', ls='dotted')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'map_sync'))
plt.close(fig)

    
fig, ax = plt.subplots(dpi=300)
ax.plot(freqs * 1e-9, cls[:,80] / cls[0,80], label='sim')
ax.plot(freqs * 1e-9, (exp_sed / exp_sed[0]) ** 2, label='expected', ls='dashed')
ax.plot(freqs * 1e-9, (freqs / freqs[0]) ** (-3), label='beta = -3')
ax.plot(freqs * 1e-9, (freqs / freqs[0]) ** (-5), label='beta = -5')
ax.plot(freqs * 1e-9, (freqs / freqs[0]) ** (-6), label='beta = -6')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'cls_sync_ell80'))
plt.close(fig)

# Now for dust.

A_d_BB = 1
alpha_d_BB = -0.3
beta_dust = 1.6
amp_beta_dust = 0
gamma_beta_dust = -4

A_s_BB = 0
alpha_s_BB = -0.7
beta_sync = -3
amp_beta_sync = 0
gamma_beta_sync = -4
rho_ds = 0

freq_pivot_dust = 353e9
temp_dust = 19.6
freq_pivot_sync = 23e9
r_tensor = 0
A_lens = 0

lmax = 200
cov_noise_ell = np.zeros((8, 2, 2, lmax + 1))
cov_scalar_ell = np.zeros((2, 2, lmax + 1))
cov_tensor_ell = np.zeros((2, 2, lmax + 1))
b_ells = np.ones((8, lmax + 1))
minfo = map_utils.MapInfo.map_info_healpix(128)
ainfo = curvedsky.alm_info(lmax)
nsplit = 1
freqs = np.asarray([25, 27, 39, 93, 145, 225, 280, 340]) * 1e9
seed = np.random.default_rng(0)

odict = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                           r_tensor, A_lens, freqs, seed, nsplit, cov_noise_ell,
                           cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
                           amp_beta_dust=amp_beta_dust, gamma_beta_dust=gamma_beta_dust, A_s_BB=A_s_BB,
                           alpha_s_BB=alpha_s_BB, beta_sync=beta_sync, freq_pivot_sync=freq_pivot_sync,
                           amp_beta_sync=amp_beta_sync, gamma_beta_sync=gamma_beta_sync, rho_ds=rho_ds)

data = odict['data']
print(data.shape)
cls = np.zeros((len(freqs), lmax+1))
for fidx in range(freqs.size):
    alm = np.zeros((2, ainfo.nelem), dtype=np.complex128)
    sht.map2alm(data[0,fidx,...], alm, minfo, ainfo, 2)

    cls[fidx] = ainfo.alm2cl(alm[1])

print(cls)
    
fig, ax = plt.subplots(dpi=300)
for fidx, freq in enumerate(freqs):
    ax.plot(cls[fidx], label=int(freq * 1e-9))
ax.legend(frameon=False)
ax.set_yscale('log')
fig.savefig(opj(imgdir, 'cls_dust'))
plt.close(fig)

exp_sed = np.zeros(len(freqs))
for fidx, freq in enumerate(freqs):
    xx = freq * 6.62607015e-34 / (2.725 * 1.380649e-23)
    xx_dust = freq * 6.62607015e-34 / (temp_dust * 1.380649e-23)    
    sed = (freq / freq_pivot_dust) ** (beta_dust - 2)
    sed *= freq ** 3 / (np.exp(xx_dust) - 1)
    sed *= (np.exp(xx) - 1) ** 2
    sed /= (xx ** 2 * np.exp(xx))
    exp_sed[fidx] = sed

pyilc_sed = np.zeros(len(freqs))
for fidx, freq in enumerate(freqs):
    xx = freq * 6.62607015e-34 / (2.725 * 1.380649e-23)
    xx_dust = freq * 6.62607015e-34 / (temp_dust * 1.380649e-23)    
    
    sed = (freq / freq_pivot_dust) ** (beta_dust + 3)
    sed /= (np.exp(xx_dust) - 1)
    sed *= (np.exp(xx) - 1) ** 2
    sed /= (xx ** 4 * np.exp(xx))
    pyilc_sed[fidx] = sed
    
fig, ax = plt.subplots(dpi=300)
# Random pixel in Q map.
ax.plot(freqs * 1e-9, data[0,:,0,80] / data[0,0,0,80], label='sim')
ax.plot(freqs * 1e-9, (exp_sed / exp_sed[0]), label='expected', ls='dashed')
ax.plot(freqs * 1e-9, (pyilc_sed / pyilc_sed[0]), label='pyilc beta + 2', ls='dotted')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'map_dust'))
plt.close(fig)

    
fig, ax = plt.subplots(dpi=300)
ax.plot(freqs * 1e-9, cls[:,80] / cls[0,80], label='sim')
ax.plot(freqs * 1e-9, (exp_sed / exp_sed[0]) ** 2, label='expected', ls='dashed')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'cls_dust_ell80'))
plt.close(fig)
