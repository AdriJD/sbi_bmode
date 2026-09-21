import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from pathlib import Path
from pixell import curvedsky
from optweight import map_utils, sht

from sbi_bmode import spectra_utils, sim_utils, so_utils

opj = os.path.join

datadir = '/u/bing/sbi_bmode/data'
imgdir = '/u/bing/sbi_bmode/binh/figures'

os.makedirs(datadir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)

lmax = 200
lmin = 30
delta_ell = 10

bins = np.arange(lmin, lmax, delta_ell)
nside = 128
cov_scalar_ell = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_nobb.dat'), lmax)
cov_tensor_ell = spectra_utils.get_cmb_spectra(opj(datadir, 'camb_lens_r1.dat'), lmax)
minfo = map_utils.MapInfo.map_info_healpix(nside)
ainfo = curvedsky.alm_info(lmax)

nsplit = 2
freq_strings = ['f030', 'f040', 'f090', 'f150', 'f230', 'f290']
freqs = [so_utils.sat_central_freqs[fstr] for fstr in freq_strings]
nfreq = len(freqs)

sensitivity_mode = 'goal'
lknee_mode = 'optimistic'
noise_cov_ell = np.ones((nfreq, 2, 2, lmax + 1))
fsky = 0.1
for fidx, fstr in enumerate(freq_strings):
    noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
        fstr, sensitivity_mode, lknee_mode, lmax)

# Fixed parameters.
freq_pivot_dust = 353
temp_dust = 19.6

# The parameters that we vary.
A_d_BB = 5
alpha_d_BB = -0.2
beta_dust = 1.59
r_tensor = 0.1
A_lens = 1

dust_pars = {'freq_pivot_dust': freq_pivot_dust, 'temp_dust': temp_dust,
             'A_d_BB': A_d_BB, 'alpha_d_BB': alpha_d_BB, 'beta_dust': beta_dust,
             'r_tensor': r_tensor, 'A_lens': A_lens}

seed = np.random.default_rng(seed=0)

# omap shape is (nsplit, nfreq, npol=2, npix)
omap = sim_utils.gen_data(A_d_BB, 
                          alpha_d_BB, 
                          beta_dust, 
                          freq_pivot_dust, 
                          temp_dust,
                          r_tensor, 
                          A_lens, 
                          freqs, 
                          seed, 
                          nsplit,
                          noise_cov_ell,
                          cov_scalar_ell, 
                          cov_tensor_ell,
                          b_ells,
                          minfo, 
                          ainfo)

print(omap.shape)
# obs_dir = Path("/u/bing/so-data/mss2")

# obsmats = so_utils.load_obs_matrix(
#     freqs=freqs,
#     obsmat_dir=obs_dir,
# )

# obs_map = sim_utils.apply_obsmatrix(
#     sky_iqu,
#     obsmats
# )

# hp.mollview(obs_map)