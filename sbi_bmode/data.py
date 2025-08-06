import os

import numpy as np
import matplotlib.pyplot as plt
from pixell import curvedsky
from optweight import map_utils

from sbi_bmode import spectra_utils, sim_utils, so_utils

opj = os.path.join

def simulator(r_tensor=0.1, A_lens=1, A_d_BB=5, alpha_d_BB=-0.2, beta_dust=1.59, seed=0, savedata=False, plotdata=False):
    """
    Create the data as needed for SBI.
    seed=-1 means random phases for each training data
    savedata doesn't do anything
    """
    datadir = '/mnt/home/abayer/cca_project/data'
    imgdir = '/mnt/home/abayer/cca_project/output/sbi'

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
            fstr, sensitivity_mode, lknee_mode, fsky, lmax)

    # Fixed parameters.
    freq_pivot_dust = 353
    temp_dust = 19.6
    
    if seed == -1:
        seed = None # unpredicable
    seed = np.random.default_rng(seed=seed)

    omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                            r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                            cov_scalar_ell, cov_tensor_ell, minfo, ainfo)
    spectra = sim_utils.estimate_spectra(omap, minfo, ainfo)

    data = sim_utils.get_final_data_vector(spectra, bins)

    if plotdata:
        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        ax.plot(data)
        fig.savefig(opj(imgdir, 'data'))
        plt.close(fig)

    return data
