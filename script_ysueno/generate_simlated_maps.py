import sys, os
import numpy as np
import pandas as pd

#sys.path.append('/home/ys5857/workspace/script/sbi_bmode/')
sys.path.append('/home/ys5857/workspace/script/optweight/')
from sbi_bmode import spectra_utils, so_utils, sim_utils
from optweight.map_utils import MapInfo
from pixell import curvedsky
from optweight import map_utils

def gen_simulated_maps():
    # This is based on https://github.com/AdriJD/sbi_bmode/blob/cb47eeadd122cb508e6caa32e4e53d32e922a62d/scripts/gen_data.py
    # syncrotoron are not included here.

    opj = os.path.join
    datadir = '/home/ys5857/workspace/script/sbi_bmode/data'
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
    b_ells = np.ones((len(freq_strings), lmax+1))
    freqs = np.array([so_utils.sat_central_freqs[fstr] for fstr in freq_strings])
    nfreq = len(freqs)

    sensitivity_mode = 'goal'
    lknee_mode = 'optimistic'
    noise_cov_ell = np.ones((nfreq, 2, 2, lmax + 1))
    cov_noise_ell = noise_cov_ell
    fsky = 0.1
    for fidx, fstr in enumerate(freq_strings):
        noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
            fstr, sensitivity_mode, lknee_mode, lmax)

    # Fixed parameters.
    freq_pivot_dust = 353e9
    temp_dust = 19.6

    # The parameters that we vary.
    r_tensor = 0.1
    A_lens = 1.
    no_cmb_ee=False
    signal_filter=None
    # dust
    A_d_BB = 5.
    alpha_d_BB = -0.2
    beta_dust = 1.59
    gamma_beta_dust = None
    amp_beta_dust=None
    # sycnhrotron
    A_s_BB=None # only synchrotron
    alpha_s_BB=None
    beta_sync = None
    gamma_beta_sync=None
    amp_beta_sync=None
    freq_pivot_sync=None

    rho_ds=None

    seed = np.random.default_rng(seed=None)    
    omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                            r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                            cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
                            amp_beta_dust, gamma_beta_dust, A_s_BB,
                            alpha_s_BB, beta_sync, freq_pivot_sync,
                            amp_beta_sync, gamma_beta_sync, rho_ds,
                            signal_filter, no_cmb_ee)    # Create real-space dust map.

    return omap


def gen_simulated_maps_gamma():
    seed=0
    opj = os.path.join
    datadir = '/home/ys5857/workspace/script/sbi_bmode/data'
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
    b_ells = np.ones((len(freq_strings), lmax+1))
    freqs = [so_utils.sat_central_freqs[fstr] for fstr in freq_strings]
    nfreq = len(freqs)

    sensitivity_mode = 'goal'
    lknee_mode = 'optimistic'
    noise_cov_ell = np.ones((nfreq, 2, 2, lmax + 1))
    cov_noise_ell = noise_cov_ell
    fsky = 0.1
    for fidx, fstr in enumerate(freq_strings):
        noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
            fstr, sensitivity_mode, lknee_mode, lmax)

    # Fixed parameters.
    freq_pivot_dust = 353e9 # 353 GHz
    temp_dust = 19.6
    # The parameters that we vary.
    r_tensor = 0.01
    A_lens = 0.45
    no_cmb_ee=False
    signal_filter=None
    # dust
    A_d_BB = 29.
    alpha_d_BB = -0.3
    beta_dust = 1.55
    gamma_beta_dust = -1
    #amp_beta_dust=0.4
    amp_beta_dust=0.01
    # sycnhrotron
    A_s_BB=1.5 # only synchrotron
    alpha_s_BB=-0.6
    beta_sync = -2.8
    gamma_beta_sync=-1
    #amp_beta_sync=0.4
    amp_beta_sync=0.01
    freq_pivot_sync=23e9 # 23GHz
    # correlation coeff between dust and synchrotron
    rho_ds=0

    seed = np.random.default_rng(seed=seed)    
    omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                            r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                            cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
                            amp_beta_dust, gamma_beta_dust, A_s_BB,
                            alpha_s_BB, beta_sync, freq_pivot_sync,
                            amp_beta_sync, gamma_beta_sync, rho_ds,
                            signal_filter, no_cmb_ee)    # Create real-space dust map.

    return omap


def gen_simulated_maps_gamma_passband():

    seed = 0
    opj = os.path.join
    datadir = '/home/ys5857/workspace/script/sbi_bmode/data'
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
    b_ells = np.ones((len(freq_strings), lmax+1))
    #freqs = [so_utils.sat_central_freqs[fstr] for fstr in freq_strings]
    #nfreq = len(freqs)

    ### passband from Bolocalc
    # /home/ys5857/workspace/script/bolocalc-so-model: https://github.com/simonsobs/bolocalc-so-model
    lf1 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_LF_1.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    lf2 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_LF_2.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    mf1 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_MF_1.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    mf2 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_MF_2.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    hf1 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_UHF_1.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    hf2 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_UHF_2.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    if False:
        plt.plot(lf1['freq'],lf1['passband'], label = 'LF_1')
        plt.plot(lf2['freq'],lf2['passband'], label = 'LF_2')
        plt.plot(mf1['freq'],mf1['passband'], label = 'MF_1')
        plt.plot(mf2['freq'],mf2['passband'], label = 'MF_2')
        plt.plot(hf1['freq'],hf1['passband'], label = 'UHF_1')
        plt.plot(hf2['freq'],hf2['passband'], label = 'UHF_2')
        plt.legend()
        plt.ylabel('Passband')
        plt.xlabel('Frequency [GHz]')
        plt.title('V3r7_Jbolo Model')

    freqs = [lf1['freq'].values*1e9,
            lf2['freq'].values*1e9,
            mf1['freq'].values*1e9,
            mf2['freq'].values*1e9,
            hf1['freq'].values*1e9,
            hf2['freq'].values*1e9] # Hz
    passbands = [lf1['passband'].values,
                lf2['passband'].values,
                mf1['passband'].values,
                mf2['passband'].values,
                hf1['passband'].values,
                hf2['passband'].values]
    nfreq = len(freqs)
    ### passband from Bolocalc


    sensitivity_mode = 'goal'
    lknee_mode = 'optimistic'
    noise_cov_ell = np.ones((nfreq, 2, 2, lmax + 1))
    cov_noise_ell = noise_cov_ell
    fsky = 0.1
    for fidx, fstr in enumerate(freq_strings):
        noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
            fstr, sensitivity_mode, lknee_mode, lmax)

    # Fixed parameters.
    freq_pivot_dust = 353e9 # 353 GHz
    temp_dust = 19.6
    # The parameters that we vary.
    r_tensor = 0.01
    A_lens = 0.45
    no_cmb_ee=False
    signal_filter=None
    # dust
    A_d_BB = 29.
    alpha_d_BB = -0.3
    beta_dust = 1.55
    gamma_beta_dust = -1
    #amp_beta_dust=0.4
    amp_beta_dust=0.01
    # sycnhrotron
    A_s_BB=1.5 # only synchrotron
    alpha_s_BB=-0.6
    beta_sync = -2.8
    gamma_beta_sync=-1
    #amp_beta_sync=0.4
    amp_beta_sync=0.01
    freq_pivot_sync=23e9 # 23GHz
    # correlation coeff between dust and synchrotron
    rho_ds=0

    seed = np.random.default_rng(seed=seed)    
    omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                            r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                            cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
                            amp_beta_dust, gamma_beta_dust, A_s_BB,
                            alpha_s_BB, beta_sync, freq_pivot_sync,
                            amp_beta_sync, gamma_beta_sync, rho_ds,
                            signal_filter, no_cmb_ee, passbands)    # Create real-space dust map.

    return omap, freqs, passbands


def gen_simulated_maps_passband():
    # This is based on https://github.com/AdriJD/sbi_bmode/blob/cb47eeadd122cb508e6caa32e4e53d32e922a62d/scripts/gen_data.py
    # syncrotoron are not included here.

    opj = os.path.join
    datadir = '/home/ys5857/workspace/script/sbi_bmode/data'
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
    b_ells = np.ones((len(freq_strings), lmax+1))
    #freqs = [so_utils.sat_central_freqs[fstr] for fstr in freq_strings]
    #nfreq = len(freqs)

    ### passband from Bolocalc
    # /home/ys5857/workspace/script/bolocalc-so-model
    lf1 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_LF_1.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    lf2 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_LF_2.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    mf1 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_MF_1.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    mf2 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_MF_2.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    hf1 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_UHF_1.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    hf2 = pd.read_csv(f'/home/ys5857/workspace/script/bolocalc-so-model/V3r7_JBolo/V3r7_Baseline/SAT/bands/detectors/V3r7_Baseline_SAT_UHF_2.txt', sep=r'\s+', header=None, names = ['freq', 'passband'])
    if False:
        plt.plot(lf1['freq'],lf1['passband'], label = 'LF_1')
        plt.plot(lf2['freq'],lf2['passband'], label = 'LF_2')
        plt.plot(mf1['freq'],mf1['passband'], label = 'MF_1')
        plt.plot(mf2['freq'],mf2['passband'], label = 'MF_2')
        plt.plot(hf1['freq'],hf1['passband'], label = 'UHF_1')
        plt.plot(hf2['freq'],hf2['passband'], label = 'UHF_2')
        plt.legend()
        plt.ylabel('Passband')
        plt.xlabel('Frequency [GHz]')
        plt.title('V3r7_Jbolo Model')

    freqs = [lf1['freq'].values*1e9,
            lf2['freq'].values*1e9,
            mf1['freq'].values*1e9,
            mf2['freq'].values*1e9,
            hf1['freq'].values*1e9,
            hf2['freq'].values*1e9] # Hz
    passbands = [lf1['passband'].values,
                lf2['passband'].values,
                mf1['passband'].values,
                mf2['passband'].values,
                hf1['passband'].values,
                hf2['passband'].values]
    nfreq = len(freqs)
    ### passband from Bolocalc

    sensitivity_mode = 'goal'
    lknee_mode = 'optimistic'
    noise_cov_ell = np.ones((nfreq, 2, 2, lmax + 1))
    cov_noise_ell = noise_cov_ell
    fsky = 0.1
    for fidx, fstr in enumerate(freq_strings):
        noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
            fstr, sensitivity_mode, lknee_mode, lmax)

    # Fixed parameters.
    freq_pivot_dust = 353e9
    temp_dust = 19.6

    # The parameters that we vary.
    r_tensor = 0.1
    A_lens = 1.
    no_cmb_ee=False
    signal_filter=None
    # dust
    A_d_BB = 5.
    alpha_d_BB = -0.2
    beta_dust = 1.59
    gamma_beta_dust = None
    amp_beta_dust=None
    # sycnhrotron
    A_s_BB=None # only synchrotron
    alpha_s_BB=None
    beta_sync = None
    gamma_beta_sync=None
    amp_beta_sync=None
    freq_pivot_sync=None

    rho_ds=None

    seed = np.random.default_rng(seed=None)    
    omap = sim_utils.gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
                            r_tensor, A_lens, freqs, seed, nsplit, noise_cov_ell,
                            cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
                            amp_beta_dust, gamma_beta_dust, A_s_BB,
                            alpha_s_BB, beta_sync, freq_pivot_sync,
                            amp_beta_sync, gamma_beta_sync, rho_ds,
                            signal_filter, no_cmb_ee, passbands)    # Create real-space dust map.

    return omap, freqs, passbands