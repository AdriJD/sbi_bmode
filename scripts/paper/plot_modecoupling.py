import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pixell import curvedsky
from optweight import alm_utils, sht, alm_c_utils, map_utils

from sbi_bmode import spectra_utils, sim_utils

opj = os.path.join

pysmdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(pysmdir, 'masks')
imgdir = opj(pysmdir, 'img')

rng = np.random.default_rng(0)

freqs = [340e9]
fstr = 'p353'
#amp = 28
amp = 45
#alpha = -0.3
#alpha = -0.7
alpha = 2
beta = 1.55
temp = 19.6
freq_pivot = 353e9

lmax = 200
lmin = 30
delta_ell = 15
delta_ell_highpass = 5
nside = 128
ainfo = curvedsky.alm_info(lmax)
minfo = map_utils.MapInfo.map_info_healpix(nside)
nsplit = 2
nfreq = len(freqs)
bins = np.arange(lmin, lmax, delta_ell)

sels_to_coadd = sim_utils.get_coadd_sels(nsplit, nfreq)

mask = hp.read_map(opj(maskdir, 'mask.fits'))
npix = mask.size
pmap = 4 * np.pi / npix
w2 = np.sum((mask ** 2) * pmap) / np.pi / 4.

c_ell = spectra_utils.get_dust_spectra(amp, alpha, lmax, freqs, beta, temp, freq_pivot)
alm_dust = alm_utils.rand_alm(c_ell, ainfo, rng)

fwhm = sim_utils.CMBSimulator.get_beam_fwhms(fstr)
b_ell = hp.gauss_beam(np.radians(fwhm / 60), lmax=lmax)
f_ell = sim_utils.get_highpass_filter(lmin, lmax, delta_ell_highpass)

# Convolve with filter
alm = alm_c_utils.lmul(alm_dust, b_ell * f_ell, ainfo)

alm_ext = np.zeros((2, ainfo.nelem), dtype=np.complex128)
alm_ext[1] = alm

# nsplit, nfreq, npol, npix.
omaps = np.zeros((2, 1, 2, 12 * nside ** 2))
omaps_eb = np.zeros((2, 1, 2, 12 * nside ** 2))

sht.alm2map(alm_ext, omaps[0,0], ainfo, minfo, 2)
sht.alm2map(alm_ext, omaps_eb[0,0], ainfo, minfo, 0)

# Hack to create 2 splits.
omaps[1] = omaps[0]
omaps_eb[1] = omaps_eb[0]

omaps_masked = omaps.copy()
omaps_masked[:] *= mask

omaps_masked_eb = omaps_eb.copy()
omaps_masked_eb[:] *= mask

spectra = sim_utils.estimate_spectra(omaps, minfo, ainfo)
spectra = sim_utils.coadd(spectra, sels_to_coadd)
data_mf = sim_utils.get_final_data_vector(spectra, bins)

spectra_eb = sim_utils.estimate_spectra(omaps_eb, minfo, ainfo)
spectra_eb = sim_utils.coadd(spectra_eb, sels_to_coadd)
data_mf_eb = sim_utils.get_final_data_vector(spectra_eb, bins)

spectra_masked = sim_utils.estimate_spectra(omaps_masked, minfo, ainfo)
spectra_masked = sim_utils.coadd(spectra_masked, sels_to_coadd)
data_mf_masked = sim_utils.get_final_data_vector(spectra_masked, bins)
data_mf_masked /= w2

spectra_masked_eb = sim_utils.estimate_spectra(omaps_masked_eb, minfo, ainfo)
spectra_masked_eb = sim_utils.coadd(spectra_masked_eb, sels_to_coadd)
data_mf_masked_eb = sim_utils.get_final_data_vector(spectra_masked_eb, bins)
data_mf_masked_eb /= w2

fig, ax = plt.subplots(dpi=300)
ax.plot(data_mf, label='fullsky')
ax.plot(data_mf_masked, label='masked')
ax.plot(data_mf_eb, label='fullsky_eb')
ax.plot(data_mf_masked_eb, label='masked_eb')

ax.legend(frameon=False)
#ax.set_yscale('symlog', linthresh=0.0005)
fig.savefig(opj(imgdir, 'mode_coupling'))
plt.close(fig)
