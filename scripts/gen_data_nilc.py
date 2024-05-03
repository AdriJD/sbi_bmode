import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pixell import curvedsky
from optweight import map_utils

import sys
sys.path.append('..')
from sbi_bmode import spectra_utils, sim_utils, so_utils, nilc_utils

opj = os.path.join

datadir = '../data'
imgdir = '../20240402_sbi'

#path to pyilc repository
pyilc_path = '/work2/09334/ksurrao/stampede3/Github/pyilc'

#directory in which to write temporary files that will be removed automatically
#set to None to use the default $TMPDIR
output_dir = '/work2/09334/ksurrao/stampede3/Github/cca_project/scripts/nilc_files' 

os.makedirs(imgdir, exist_ok=True)
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)

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
        fstr, sensitivity_mode, lknee_mode, fsky, lmax)

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
omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                          r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                          cov_scalar_ell, cov_tensor_ell, minfo, ainfo)



# build NILC B-mode maps with shape (nsplit, ncomp=2, npix)
B_maps = np.zeros((nsplit, nfreq, minfo.npix))
for split in range(nsplit):
    for f, freq_str in enumerate(freq_strings):
        Q, U = omap[split, f]
        alm_T, alm_E, alm_B = hp.map2alm([np.zeros_like(Q), Q, U], pol=True) #alm_T is just a placeholder
        B_maps[split, f] = 10**(-6)*hp.alm2map(alm_B, nside)
map_tmpdir = nilc_utils.write_maps(B_maps, output_dir=output_dir)
nilc_maps = nilc_utils.get_nilc_maps(pyilc_path, map_tmpdir, nsplit, nside, dust_pars, 
                                     so_utils.sat_central_freqs, so_utils.sat_beam_fwhms,
                                     output_dir=output_dir, remove_files=True, debug=False)


spectra = sim_utils.estimate_spectra_nilc(nilc_maps, minfo, ainfo)

data = sim_utils.get_final_data_vector(spectra, bins, lmin, lmax)

fig, ax = plt.subplots(dpi=300, constrained_layout=True)
ax.plot(data)
fig.savefig(opj(imgdir, 'data'))
plt.close(fig)
