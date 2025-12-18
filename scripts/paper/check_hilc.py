import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import yaml
import torch

from scipy.ndimage import uniform_filter1d
from scipy.linalg import eigh
from pixell import enmap, enplot, curvedsky
from optweight import alm_utils, sht, alm_c_utils, mat_utils
from sbi_bmode import sim_utils, script_utils, spectra_utils, nilc_utils

opj = os.path.join
basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
specdir = '/u/adriaand/local/cca_project/data'
pyilcdir = '/u/adriaand/local/pyilc'

odir = opj(basedir, 'check_sync_fiducial_hilc')
imgdir = opj(odir, 'img')
mapdir = opj(imgdir, 'maps')
os.makedirs(odir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)
os.makedirs(mapdir, exist_ok=True)

def draw_from_prior(prior_list, nsamp):
    '''
    Extract prior limits.

    Parameters
    ----------
    prior_list : list of torch.distribition instances
        Priors.
    nsamp : int
        Number of prior draws.

    Returns
    -------
    draw : (nsamp, n_prior) array
        Prior draws.
    '''

    out = np.zeros((nsamp, len(prior_list)))

    for pidx, prior in enumerate(prior_list):
        out[:,pidx] = np.asarray(prior.sample((nsamp,)))[:,0]

    return out

def gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
             r_tensor, A_lens, freqs, seed, nsplit, cov_noise_ell,
             cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
             amp_beta_dust=None, gamma_beta_dust=None, A_s_BB=None,
             alpha_s_BB=None, beta_sync=None, freq_pivot_sync=None,
             amp_beta_sync=None, gamma_beta_sync=None, rho_ds=None,
             signal_filter=None, no_cmb_ee=False):
    '''
    Generate simulated maps.

    Parameters
    ----------
    A_d_BB : float
        Dust amplitude.
    alpha_d_BB : float
        Dust spatial power law index.
    beta_dust : float
        Dust frequency power law index.
    freq_pivot_dust : float
        Pivot frequency for the frequency power law.
    temp_dust : float
        Dust temperature for the blackbody part of the model.
    r_tensor : float
        Tensor-to-scalar ratio.
    A_lens : float
        Amplitude of lensing contribution to BB.
    freqs : array-like
        Passband centers for the frquency channels of the instrument.
    seed : numpy.random._generator.Generator object or int
        Random number generator or seed for new random number generator.
    nsplit : int
        Number of splits of the data that have independent noise.
    cov_noise_ell : (nfreq, npol, npol, nell) array
        Noise covariance matrix.
    cov_scalar_ell : (npol, nell) array
        Signal covariance matrix with the EE and BB spectra from scalar perturbations.
    cov_tensor_ell : (npol, nell) array
        Signal covariance matrix with the EE and BB spectra from tensor perturbations.
    b_ells : (nfreq, nell) array
        Beam for each frequency.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    amp_beta_dust : float, optional
        Amplitude of dust beta power spectrum at pivot multipole.
    gamma_beta_dust : float, optional
        Tilt of dust beta power spectrum.
    A_s_BB : float
        Synchrotron amplitude.
    alpha_s_BB : float
        Synchrotron spatial power law index.
    beta_sync : float
        Synchrotron frequency power law index.
    freq_pivot_sync: float
        Pivot frequency for the synchrotron frequency power law.
    amp_beta_sync : float, optional
        Amplitude of synchrotron beta power spectrum at pivot multipole.
    gamma_beta_sync : float, optional
        Tilt of synchrotron beta power spectrum.
    rho_ds : float, optional
        Correlation coefficient between dust and synchroton amplitudes.
    signal_filter : (nell) array. optional
        Harmonic filter that is applied to the signal (similar to beam).
    no_cmb_ee : bool, optional
        If set, set EE cmb constribution to zero. Added for backwards compatibiliy.

    Returns
    -------
    out_dict : dict
        Output dictionary with following key-value pairs:
            data : (nsplit, nfreq, npol, npix)
                Simulated data.
            gamma_dust_ell : (lmax + 1) array, optional
                Realization of the gamma_dust power spectrum
            gamma_sync_ell : (lmax + 1) array, optional
                Realization of the gamma_sync power spectrum
            
    '''

    nfreq = len(freqs)
    out = np.zeros((nsplit, nfreq, 2, minfo.npix))

    # Spawn rng for dust and noise.
    seed = np.random.default_rng(seed)
    rngs = seed.spawn(3 + nsplit)
    rng_cmb = rngs[0]
    rng_dust = rngs[1]
    rng_beta = rngs[2]
    rngs_noise = rngs[3:]

    # Generate the CMB spectra.
    cov_ell = spectra_utils.get_combined_cmb_spectrum(
        r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell)
    lmax = cov_ell.shape[-1] - 1
    assert ainfo.lmax == lmax

    if A_s_BB is not None:
        ncomp_fg = 2
    else:
        ncomp_fg = 1
    cov_fg_ell = np.zeros((ncomp_fg, ncomp_fg, lmax + 1))

    # Generate frequency-independent signal, scale with frequency later.
    cov_fg_ell[0,0] = spectra_utils.get_ell_shape(lmax, alpha_d_BB, ell_pivot=80)
    cov_fg_ell[0,0] *= A_d_BB

    if A_s_BB is not None:
        cov_fg_ell[1,1] = spectra_utils.get_ell_shape(lmax, alpha_s_BB, ell_pivot=80)
        cov_fg_ell[1,1] *= A_s_BB

        if rho_ds is not None:
            cov_fg_ell[0,1] = rho_ds * np.sqrt(cov_fg_ell[0,0] * cov_fg_ell[1,1])
            cov_fg_ell[1,0] = cov_fg_ell[0,1]

    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    if no_cmb_ee:
        cmb_alm[0] = 0

    cmb_map = np.zeros((2, minfo.npix))
    sht.alm2map(cmb_alm, cmb_map, ainfo, minfo, 2)
        
    fg_alm = alm_utils.rand_alm(cov_fg_ell, ainfo, rng_dust, dtype=np.complex128)

    if A_s_BB is not None:
        if (gamma_beta_dust != gamma_beta_sync) and None in (gamma_beta_dust, gamma_beta_sync):
            # Raises error only if one of two is None.
            raise ValueError('We only support either both dust and sync gammas or none.')

    if gamma_beta_dust is not None:
        assert amp_beta_dust is not None

        # Create real-space dust map.
        alm_tmp = np.zeros((2, ainfo.nelem), dtype=np.complex128)
        alm_tmp[1] = fg_alm[0]
        dust_map = np.zeros((2, minfo.npix))
        sht.alm2map(alm_tmp, dust_map, ainfo, minfo, 2)
        
        # Generate the dust beta map.
        beta_dust, gamma_dust_ell = sim_utils.get_beta_map(
            minfo, ainfo, beta_dust, amp_beta_dust, gamma_beta_dust, rng_beta)

        if A_s_BB is not None:
            alm_tmp[1] = fg_alm[1]
            sync_map = np.zeros((2, minfo.npix))
            sht.alm2map(alm_tmp, sync_map, ainfo, minfo, 2)
            beta_sync, gamma_sync_ell = sim_utils.get_beta_map(
                minfo, ainfo, beta_sync, amp_beta_sync, gamma_beta_sync, rng_beta)
        else:
            sync_map, beta_sync = None, None

        gen_data_per_freq = lambda freq, cov_noise_ell, b_ell: sim_utils._gen_data_per_freq_gamma(
            freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
            cmb_alm, dust_map, nsplit, rngs_noise, ainfo, minfo, b_ell, sync_map=sync_map,
            beta_sync=beta_sync, freq_pivot_sync=freq_pivot_sync)

    else:
        gen_data_per_freq = lambda freq, cov_noise_ell, b_ell: sim_utils._gen_data_per_freq_simple(
            freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
            cmb_alm, fg_alm, nsplit, rngs_noise, ainfo, minfo, b_ell, beta_sync=beta_sync,
            freq_pivot_sync=freq_pivot_sync)

        gamma_dust_ell, gamma_sync_ell = None, None

    for fidx, freq in enumerate(freqs):

        b_ell = b_ells[fidx]
        if signal_filter is not None:
            b_ell = b_ell * signal_filter
        out[:,fidx,:,:] = gen_data_per_freq(freq, cov_noise_ell[fidx], b_ell)

    out_dict = {'data' : out}
    out_dict['cmb_map'] = cmb_map    
    out_dict['dust_map'] = dust_map
    out_dict['sync_map'] = sync_map

    out_dict['beta_dust'] = beta_dust
    out_dict['beta_sync'] = beta_sync   
    
    if gamma_dust_ell is not None:
        out_dict['gamma_dust_ell'] = gamma_dust_ell
    if gamma_sync_ell is not None:
        out_dict['gamma_sync_ell'] = gamma_sync_ell

    return out_dict

def smooth_tophat(c_ell, binsize):
    '''

    Parameters
    ----------
    c_ell : (..., nell)
        Input array

    Returns
    -------
    out : (..., nell)
        Smooth copy of input.
    '''

    return uniform_filter1d(c_ell, size=binsize, axis=-1, mode='nearest')

def get_data_cov(data_alm, ainfo, binsize=15):
    '''

    Parameters
    ----------
    data_alm : (nfreq, nelem)
    
    Returns
    -------
    data_cov : (nfreq, nell)
    '''

    assert data_alm.ndim == 2
    nfreq = data_alm.shape[0]
    
    data_cov = ainfo.alm2cl(data_alm[:,None,:], data_alm[None,:,:])

    assert data_cov.shape == (nfreq, nfreq, ainfo.lmax+1)
    
    return smooth_tophat(data_cov, binsize)

cmb_temp = 2.726
hplanck = 6.62607015e-34
kboltz = 1.380649e-23
clight = 299792458.
hk_ratio = 4.799243073366221e-11
k3_hc2_ratio = 6.669547705834e-20 # kb^3 / (hc)^2
b0 = 2 * k3_hc2_ratio * cmb_temp ** 2

def get_x(freq, temp):
    return (hk_ratio * freq) / temp

def get_cmb_sed(freq):
    return 1.

def get_bprime(freq):
    xx = get_x(freq, cmb_temp)
    return b0 * xx ** 4 * np.exp(xx) / (np.exp(xx) - 1) ** 2
    
def get_dust_sed(freq, freq_ref, beta, temp):

    xx_dust = get_x(freq, temp)
    x0_dust = get_x(freq_ref, temp)    
    bprime = get_bprime(freq)
    bprime0 = get_bprime(freq_ref)

    out = (freq / freq_ref) ** (beta + 3)
    out *= (np.exp(x0_dust) - 1) / (np.exp(xx_dust) - 1)
    out *= bprime0 / bprime
    
    return out
    
def get_dust_dbeta_sed(freq, freq_ref, beta, temp):
    return np.log(freq / freq_ref) * get_dust_sed(freq, freq_ref, beta, temp)

def get_sync_sed(freq, freq_ref, beta):

    bprime = get_bprime(freq)
    bprime0 = get_bprime(freq_ref)

    out = (freq / freq_ref) ** beta
    out *= bprime0 / bprime

    return out
    
def get_sync_dbeta_sed(freq, freq_ref, beta):
    return np.log(freq / freq_ref) * get_sync_sed(freq, freq_ref, beta)
    
def get_mixing_matrix(comps, freqs):
    '''

    Parameters
    ----------
    sed_funcs : (ncomp) array-like of functions

    freqs : (nfreq) array
        Frequencies in Hz.

    Returns
    -------
    mix_mat : (nfreq, ncomp) array    
    '''

    ncomp = len(comps)
    nfreq = len(freqs)
    
    mix_mat = np.zeros((nfreq, ncomp))

    for sidx, sed in enumerate(comps):
        for fidx, freq in enumerate(freqs):
            mix_mat[fidx,sidx] = sed(freq)
    
    return mix_mat

def get_ilc_cov(data_cov, mix_mat):
    '''
    Parameters
    ----------
    data_cov : (nfreq, nfreq, nell) array

    mix_mat : (nfreq, ncomp) array


    '''

    data_icov = mat_utils.matpow(data_cov, -1)
    ilc_icov = np.einsum('ab, bcl, cd -> adl', mix_mat.T, data_icov, mix_mat)
    
    return mat_utils.matpow(ilc_icov, -1)

def get_ilc_estimate(data_alm, ainfo, data_cov, mix_mat):
    '''

    '''

    data_icov = mat_utils.matpow(data_cov, -1)
    ilc_icov = np.einsum('ab, bcl, cd -> adl', mix_mat.T, data_icov, mix_mat)

    ilc_cov = mat_utils.matpow(ilc_icov, -1)
    
    rhs = np.einsum('ab, bcl -> acl', mix_mat.T, data_icov)
    weight = np.einsum('abl, bcl -> acl', ilc_cov, rhs)

    ncomp, nfreq = weight.shape[:-1]
    out = np.zeros((ncomp, ainfo.nelem), dtype=np.complex128)

    for cidx in range(ncomp):
        for fidx in range(nfreq):
            out[cidx] += alm_c_utils.lmul(data_alm[fidx], weight[cidx,fidx], ainfo)

    return out
            
def cov2cor(cov_mat):
    '''
    Parameters
    ----------
    cov_mat : (N, N, nell)

    Returns
    -------
    cor_mat : (N, N, nell)
    '''

    diag = np.einsum('aal -> al', cov_mat)
    diag_outer = np.einsum('al, bl -> abl', diag, diag)

    return cov_mat / np.sqrt(diag_outer)

with open(opj(basedir, 'run86', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)

data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
prior, param_names = script_utils.get_prior(params_dict)
param_truths = script_utils.get_true_params(params_dict)

torch.manual_seed(0)
nsamp = 1
thetas = draw_from_prior(prior, nsamp)

wavelet_type = 'TopHatHarmonic'
fiducial_T_dust = 19.6
fiducial_beta_dust = 1.6
fiducial_beta_sync = -3

lmin = 30
lmax = 200
ells = np.arange(lmax + 1)
ells_trunc = ells[lmin:lmax+1]
bins = np.arange(30, 210, 10)
bin_centers = bins[:-1] + 5
oversample = 4
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                         2 * np.pi / (2 * oversample * lmax + 1)],
                                    variant='fejer1')

cmb_simulator = sim_utils.CMBSimulator(
    specdir, data_dict, fixed_params_dict, pyilcdir=pyilcdir, wavelet_type=wavelet_type,
    use_dust_map=True, use_dbeta_map=True, use_sync_map=True,
    use_dbeta_sync_map=True, deproj_dust=True, deproj_dbeta=True,
    deproj_sync=True, deproj_dbeta_sync=True, fiducial_beta=fiducial_beta_dust,
    fiducial_T_dust=fiducial_T_dust, fiducial_beta_sync=fiducial_beta_sync, odir=odir,
    coadd_equiv_crosses=True)

highpass_filter = np.ones((2, lmax+1)) * cmb_simulator.highpass_filter

print(f'{cmb_simulator.freq_pivot_sync=}')

print(f'{cmb_simulator.freq_strings=}')
#cmb_simulator.noise_cov_ell[2] *= 0.1
#cmb_simulator.noise_cov_ell[-1] *= 0.1
#cmb_simulator.noise_cov_ell[:] *= 10.

print(f'{cmb_simulator.freqs=}')
#cmb_simulator.freqs[0] = 15e9
#cmb_simulator.freqs[2] = 5e9
#cmb_simulator.freqs[-2] = 350e9
#cmb_simulator.freqs[-1] = 700e9


for idx, theta in enumerate(thetas):

    # NOTE NOTE
    #theta[7] = 4.
    theta = np.asarray(list(param_truths.values()))

    # NOTE set Bd and Bs to zero.
    theta[5] = 0.
    theta[10] = 0.

    # NOTE
    theta[9] = -3.

    # NOTE
    theta[4] = 1.6
    
    print(f'{idx=}, {theta=}')    
    theta_dict = dict(zip(param_names, theta))
    
    rng = np.random.default_rng(idx)

    # Create maps, save components
    odict = gen_data(
        theta_dict['A_d_BB'], theta_dict['alpha_d_BB'], theta_dict['beta_dust'],
        cmb_simulator.freq_pivot_dust, cmb_simulator.temp_dust,
        theta_dict['r_tensor'], theta_dict['A_lens'], cmb_simulator.freqs, rng,
        cmb_simulator.nsplit, cmb_simulator.noise_cov_ell,
        cmb_simulator.cov_scalar_ell, cmb_simulator.cov_tensor_ell,
        cmb_simulator.b_ells, cmb_simulator.minfo, cmb_simulator.ainfo,
        amp_beta_dust=theta_dict['amp_beta_dust'], gamma_beta_dust=theta_dict['gamma_beta_dust'],
        A_s_BB=theta_dict['A_s_BB'], alpha_s_BB=theta_dict['alpha_s_BB'], beta_sync=theta_dict['beta_sync'],
        freq_pivot_sync=cmb_simulator.freq_pivot_sync, amp_beta_sync=theta_dict['amp_beta_sync'],
        gamma_beta_sync=theta_dict['gamma_beta_sync'], rho_ds=theta_dict['rho_ds'],
        signal_filter=cmb_simulator.highpass_filter, no_cmb_ee=(cmb_simulator.mask is not None))

    omap = odict['data']

    data_alm = np.zeros((2, cmb_simulator.nfreq, 2, cmb_simulator.ainfo.nelem),
                       dtype=np.complex128) # nsplit, nfreq, npol=2.
    for split in range(cmb_simulator.nsplit):
        for f, freq_str in enumerate(cmb_simulator.freq_strings):
            sht.map2alm(omap[split,f], data_alm[split,f], cmb_simulator.minfo, cmb_simulator.ainfo, 2)

    # Reconvolve to common beam.
    print(cmb_simulator.freq_strings)
    for b_ell in cmb_simulator.b_ells:
        data_alm = alm_c_utils.lmul(data_alm, cmb_simulator.b_ells[0] / b_ell, cmb_simulator.ainfo)
            
    # Only consider first split.
    cl_data = cmb_simulator.ainfo.alm2cl(data_alm[0,:,1,:][:,None,:], data_alm[0,:,1,:][None,:,:])
    print(f'{cl_data.shape=}')

    nfreq = cmb_simulator.nfreq
    fig, axs = plt.subplots(ncols=nfreq, nrows=nfreq, dpi=300, figsize=(10, 10), constrained_layout=True)
    for idxs, ax in np.ndenumerate(axs):        
        axs[idxs].plot(ells_trunc, cl_data[idxs][lmin:lmax+1])
    fig.savefig(opj(imgdir, 'cl_data'))
    plt.close(fig)
    
    data_cov = get_data_cov(data_alm[0,:,1,:], cmb_simulator.ainfo)
    
    betas = [-3, -4, -5, -6, -7]

    def analyze(ilc_covs, ilc_cors, ilc_dets, ilc_cov_dets, part_ratios, comp_names):

        tag = '_'.join(comp_names)
        
        fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
                                sharex=True, figsize=(6, 6))
        for idxs, ax in np.ndenumerate(axs):
            if idxs[0] < idxs[1]:
                ax.set_axis_off()
                continue
            
            for bidx, fiducial_beta_sync in enumerate(betas):                
                ax.plot(ells_trunc, ilc_covs[bidx][idxs][lmin:lmax+1])
            ax.set_title(f'{comp_names[idxs[0]]} x {comp_names[idxs[1]]}')
        fig.suptitle(tag)
        fig.savefig(opj(imgdir, f'ilc_cov_{tag}'))
        plt.close(fig)

        fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
                                sharex=True, figsize=(6, 6))
        for idxs, ax in np.ndenumerate(axs):
            if idxs[0] < idxs[1]:
                ax.set_axis_off()
                continue
            
            for bidx, fiducial_beta_sync in enumerate(betas):
                ax.plot(ells_trunc, ilc_cors[bidx][idxs][lmin:lmax+1])            
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_title(f'{comp_names[idxs[0]]} x {comp_names[idxs[1]]}')
        fig.suptitle(tag)            
        fig.savefig(opj(imgdir, f'ilc_cor_{tag}'))        
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True,
                                sharex=True)
        for bidx, fiducial_beta_sync in enumerate(betas):
            ax.plot(ells_trunc, ilc_dets[bidx][lmin:lmax+1], label=f'beta_pyilc = {fiducial_beta_sync + 2}')
        ax.legend(frameon=False)
        fig.suptitle(tag)        
        fig.savefig(opj(imgdir, f'ilc_det_{tag}'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True,
                                sharex=True)
        for bidx, fiducial_beta_sync in enumerate(betas):
            ax.plot(ells_trunc, ilc_cov_dets[bidx][lmin:lmax+1], label=f'beta_pyilc = {fiducial_beta_sync + 2}')
        ax.legend(frameon=False)
        fig.suptitle(tag)        
        fig.savefig(opj(imgdir, f'ilc_cov_det_{tag}'))
        plt.close(fig)
        
        fig, ax = plt.subplots(dpi=300, constrained_layout=True,
                                sharex=True)
        for bidx, fiducial_beta_sync in enumerate(betas):
            ax.plot(ells_trunc, -np.log(ilc_dets[bidx][lmin:lmax+1]), label=f'beta_pyilc = {fiducial_beta_sync + 2}')
        ax.legend(frameon=False)
        fig.suptitle(tag)        
        fig.savefig(opj(imgdir, f'ilc_info_{tag}'))
        plt.close(fig)
        
        fig, ax = plt.subplots(dpi=300, constrained_layout=True,
                                sharex=True)
        for bidx, fiducial_beta_sync in enumerate(betas):
            ax.plot(ells_trunc, part_ratios[bidx][lmin:lmax+1], label=f'beta_pyilc = {fiducial_beta_sync + 2}')
        ax.legend(frameon=False)
        fig.suptitle(tag)        
        fig.savefig(opj(imgdir, f'ilc_pr_{tag}'))
        plt.close(fig)
        

    # c_d_dbd_s_dbs
    ilc_covs = []
    ilc_cors = []
    ilc_dets = []
    ilc_cov_dets = []    
    part_ratios = []
        
    for bidx, fiducial_beta_sync in enumerate(betas):

        comps = [lambda freq : get_cmb_sed(freq),
                 lambda freq : get_dust_sed(
                     freq, cmb_simulator.freq_pivot_dust, fiducial_beta_dust, fiducial_T_dust),
                 lambda freq : get_dust_dbeta_sed(
                     freq, cmb_simulator.freq_pivot_dust, fiducial_beta_dust, fiducial_T_dust),
                 lambda freq : get_sync_sed(freq, cmb_simulator.freq_pivot_sync, fiducial_beta_sync+2),
                 lambda freq : get_sync_dbeta_sed(freq, cmb_simulator.freq_pivot_sync, fiducial_beta_sync+2)]
        
        mix_mat =  get_mixing_matrix(comps, cmb_simulator.freqs)
        ilc_cov = get_ilc_cov(data_cov, mix_mat)
        ilc_cor = cov2cor(ilc_cov)
        ilc_det = np.linalg.det(ilc_cor.transpose(2, 0, 1))
        ilc_cov_det = np.linalg.det(ilc_cov.transpose(2, 0, 1))        
        part_ratio = np.zeros(lmax + 1)
        for lidx in range(lmax+1):            
            ev = eigh(ilc_cor[:,:,lidx], eigvals_only=True)
            part_ratio[lidx] = np.sum(ev)  ** 2 / np.sum(ev ** 2)
        ilc_covs.append(ilc_cov)
        ilc_cors.append(ilc_cor)
        ilc_dets.append(ilc_det)
        ilc_cov_dets.append(ilc_cov_det)        
        part_ratios.append(part_ratio)

        ilc_alm = get_ilc_estimate(data_alm[0,:,1,:], cmb_simulator.ainfo, data_cov, mix_mat)

        for cidx, comp in enumerate(['cmb', 'dust', 'sync', 'dbd', 'dbs']):


            #comp_enmap = reproject.healpix2map(
            #    nilc_maps[sidx,cidx], shape=shape, wcs=wcs, method='spline', order=0)
            omap = enmap.zeros(shape, wcs)
            curvedsky.alm2map(ilc_alm[cidx], omap)

            plot = enplot.plot(omap, colorbar=True, grid=30)
            enplot.write(opj(mapdir, f'out_mbfs{abs(fiducial_beta_sync)}_{comp}'), plot)
        
        
    analyze(ilc_covs, ilc_cors, ilc_dets, ilc_cov_dets, part_ratios, ['c', 'd', 'dbd', 's', 'dbs'])
            
    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_covs[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cov_c_d_dbd_s_dbs'))
    # plt.close(fig)

    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_cors[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cor_c_d_dbd_s_dbs'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, ilc_dets[bidx][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_det_c_d_dbd_s_dbs'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, part_ratios[bidx][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_pr_c_d_dbd_s_dbs'))
    # plt.close(fig)
    
    # c_d_s_dbs
    ilc_covs = []
    ilc_cors = []
    ilc_dets = []
    ilc_cov_dets = []        
    part_ratios = []
    
    for bidx, fiducial_beta_sync in enumerate(betas):

        comps = [lambda freq : get_cmb_sed(freq),
                 lambda freq : get_dust_sed(
                     freq, cmb_simulator.freq_pivot_dust, fiducial_beta_dust, fiducial_T_dust),
                 lambda freq : get_sync_sed(freq, cmb_simulator.freq_pivot_sync, fiducial_beta_sync+2), # NOTE +2.
                 lambda freq : get_sync_dbeta_sed(freq, cmb_simulator.freq_pivot_sync, fiducial_beta_sync+2)]
        
        mix_mat =  get_mixing_matrix(comps, cmb_simulator.freqs)
        ilc_cov = get_ilc_cov(data_cov, mix_mat)
        ilc_cor = cov2cor(ilc_cov)
        ilc_det = np.linalg.det(ilc_cor.transpose(2, 0, 1))
        ilc_cov_det = np.linalg.det(ilc_cov.transpose(2, 0, 1))                
        part_ratio = np.zeros(lmax + 1)
        for lidx in range(lmax+1):            
            ev = eigh(ilc_cor[:,:,lidx], eigvals_only=True)
            part_ratio[lidx] = np.sum(ev)  ** 2 / np.sum(ev ** 2)
        ilc_covs.append(ilc_cov)
        ilc_cors.append(ilc_cor)
        ilc_dets.append(ilc_det)
        ilc_cov_dets.append(ilc_cov_det)                
        part_ratios.append(part_ratio)

    analyze(ilc_covs, ilc_cors, ilc_dets, ilc_cov_dets, part_ratios, ['c', 'd', 's', 'dbs'])

        
    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_covs[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cov_c_d_s_dbs'))
    # plt.close(fig)
    
    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_cors[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cor_c_d_s_dbs'))
    # plt.close(fig)
    
    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, ilc_dets[bidx][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_det_c_d_s_dbs'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, part_ratios[bidx][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_pr_c_d_s_dbs'))
    # plt.close(fig)

    # c_d_dbd_s
    ilc_covs = []
    ilc_cors = []
    ilc_dets = []
    ilc_cov_dets = []        
    part_ratios = []
    
    for bidx, fiducial_beta_sync in enumerate(betas):

        comps = [lambda freq : get_cmb_sed(freq),
                 lambda freq : get_dust_sed(
                     freq, cmb_simulator.freq_pivot_dust, fiducial_beta_dust, fiducial_T_dust),
                 lambda freq : get_dust_dbeta_sed(
                     freq, cmb_simulator.freq_pivot_dust, fiducial_beta_dust, fiducial_T_dust),
                 lambda freq : get_sync_sed(freq, cmb_simulator.freq_pivot_sync, fiducial_beta_sync+2)]
        
        mix_mat =  get_mixing_matrix(comps, cmb_simulator.freqs)
        ilc_cov = get_ilc_cov(data_cov, mix_mat)
        ilc_cor = cov2cor(ilc_cov)
        ilc_det = np.linalg.det(ilc_cor.transpose(2, 0, 1))
        ilc_cov_det = np.linalg.det(ilc_cov.transpose(2, 0, 1))                
        part_ratio = np.zeros(lmax + 1)
        for lidx in range(lmax+1):            
            ev = eigh(ilc_cor[:,:,lidx], eigvals_only=True)
            part_ratio[lidx] = np.sum(ev)  ** 2 / np.sum(ev ** 2)
        ilc_covs.append(ilc_cov)
        ilc_cors.append(ilc_cor)
        ilc_dets.append(ilc_det)
        ilc_cov_dets.append(ilc_cov_det)                
        part_ratios.append(part_ratio)

    analyze(ilc_covs, ilc_cors, ilc_dets, ilc_cov_dets, part_ratios, ['c', 'd', 'dbd', 's'])

        
    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_covs[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cov_c_d_dbd_s'))
    # plt.close(fig)

    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_cors[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cor_c_d_dbd_s'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, ilc_dets[bidx][lmin:lmax+1], label=f'beta = {fiducial_beta_sync + 2}'))
    # ax.legend()
    # fig.savefig(opj(imgdir, 'ilc_det_c_d_dbd_s'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, part_ratios[bidx][lmin:lmax+1], label=f'beta = {fiducial_beta_sync + 2}')
    # ax.legend(frameon=False)
    # fig.savefig(opj(imgdir, 'ilc_pr_c_d_dbd_s'))
    # plt.close(fig)

    
    # c_d_s
    ilc_covs = []    
    ilc_cors = []
    ilc_dets = []
    ilc_cov_dets = []        
    part_ratios = []
    
    for bidx, fiducial_beta_sync in enumerate(betas):

        comps = [lambda freq : get_cmb_sed(freq),
                 lambda freq : get_dust_sed(
                     freq, cmb_simulator.freq_pivot_dust, fiducial_beta_dust, fiducial_T_dust),
                 lambda freq : get_sync_sed(freq, cmb_simulator.freq_pivot_sync, fiducial_beta_sync+2)] # NOTE +2.
        
        mix_mat =  get_mixing_matrix(comps, cmb_simulator.freqs)
        ilc_cov = get_ilc_cov(data_cov, mix_mat)
        ilc_cor = cov2cor(ilc_cov)
        ilc_det = np.linalg.det(ilc_cor.transpose(2, 0, 1))
        ilc_cov_det = np.linalg.det(ilc_cov.transpose(2, 0, 1))                
        part_ratio = np.zeros(lmax + 1)
        for lidx in range(lmax+1):            
            ev = eigh(ilc_cor[:,:,lidx], eigvals_only=True)            
            part_ratio[lidx] = np.sum(ev)  ** 2 / np.sum(ev ** 2)
        ilc_covs.append(ilc_cov)
        ilc_cors.append(ilc_cor)
        ilc_dets.append(ilc_det)
        ilc_cov_dets.append(ilc_cov_det)                
        part_ratios.append(part_ratio)

    analyze(ilc_covs, ilc_cors, ilc_dets, ilc_cov_dets, part_ratios, ['c', 'd', 's'])
        
    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_covs[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cov_c_d_s'))
    # plt.close(fig)
    
    # fig, axs = plt.subplots(ncols=len(comps), nrows=len(comps), dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for idxs, ax in np.ndenumerate(axs):        
    #     for bidx, fiducial_beta_sync in enumerate(betas):
    #         axs[idxs].plot(ells_trunc, ilc_cors[bidx][idxs][lmin:lmax+1])
    # fig.savefig(opj(imgdir, 'ilc_cor_c_d_s'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, ilc_dets[bidx][lmin:lmax+1], label=f'beta = {fiducial_beta_sync + 2}'))
    # ax.legend()
    # fig.savefig(opj(imgdir, 'ilc_det_c_d_s'))
    # plt.close(fig)

    # fig, ax = plt.subplots(dpi=300, constrained_layout=True,
    #                         sharex=True)
    # for bidx, fiducial_beta_sync in enumerate(betas):
    #     ax.plot(ells_trunc, part_ratios[bidx][lmin:lmax+1], label=f'beta = {fiducial_beta_sync + 2}')
    # ax.legend(frameon=False)
    # fig.savefig(opj(imgdir, 'ilc_pr_c_d_s'))
    # plt.close(fig)
    
