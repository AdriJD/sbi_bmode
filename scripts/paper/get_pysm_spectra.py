'''
Compute pseudo-spectra in the masked region for all pysm maps.
'''

import os

import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import map_utils, alm_utils, alm_c_utils, sht

from sbi_bmode import sim_utils, spectra_utils

opj = os.path.join

pysmdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(pysmdir, 'masks')
odir = opj(pysmdir, 'spectra')
specdir = '/u/adriaand/local/cca_project/data'

os.makedirs(odir, exist_ok=True)

rng = np.random.default_rng(0)

freqs = ['wK', 'f030', 'f040', 'f090', 'f150', 'f230', 'f290', 'p353']

beam_fwhms = [sim_utils.CMBSimulator.get_beam_fwhms(fstr) for fstr in freqs]
# We re-order the freq_strings based on FWHM because pyilc requires that.
freqs = np.asarray(freqs)[np.argsort(np.asarray(beam_fwhms))][::-1]

dust_models = ['d1', 'd4', 'd10', 'd12']
sync_models = ['s5', 's7', ]

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
r_tensor = 0.01
A_lens = 0.45

sels_to_coadd = sim_utils.get_coadd_sels(nsplit, nfreq)

cov_scalar_ell = spectra_utils.get_cmb_spectra(
    opj(specdir, 'camb_lens_nobb.dat'), lmax)
cov_tensor_ell = spectra_utils.get_cmb_spectra(
    opj(specdir, 'camb_lens_r1.dat'), lmax)
cov_ell = spectra_utils.get_combined_cmb_spectrum(
    r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell)
alm_cmb = alm_utils.rand_alm(cov_ell, ainfo, rng, dtype=np.complex128)[1]

mask = hp.read_map(opj(maskdir, 'mask.fits'))

for dust_model in dust_models:
    for sync_model in sync_models:
        print(dust_model, sync_model)

        # nsplit, nfreq, npol, npix.
        omaps = np.zeros((2, len(freqs), 2, 12 * nside ** 2))

        for fidx, freq in enumerate(freqs):

            alm_dust = hp.read_alm(opj(pysmdir, f'pysm_{dust_model}_{freq}.fits'))
            alm_sync = hp.read_alm(opj(pysmdir, f'pysm_{sync_model}_{freq}.fits'))
            alm = alm_dust + alm_sync + alm_cmb

            # Beam convolve
            fwhm = sim_utils.CMBSimulator.get_beam_fwhms(freq)
            b_ell = hp.gauss_beam(np.radians(fwhm / 60), lmax=lmax)
            f_ell = sim_utils.get_highpass_filter(lmin, lmax, delta_ell_highpass)

            # Convolve with filter
            alm = alm_c_utils.lmul(alm, b_ell * f_ell, ainfo)

            alm_ext = np.zeros((2, ainfo.nelem), dtype=np.complex128)
            alm_ext[1] = alm
            sht.alm2map(alm_ext, omaps[0,fidx], ainfo, minfo, 2)

        # Hack to create 2 splits.
        omaps[1] = omaps[0]
        omaps[:] *= mask

        spectra = sim_utils.estimate_spectra(omaps, minfo, ainfo)
        spectra = sim_utils.coadd(spectra, sels_to_coadd)
        data_mf = sim_utils.get_final_data_vector(spectra, bins)

        np.save(opj(odir, f'spectra_{dust_model}_{sync_model}'), data_mf)
