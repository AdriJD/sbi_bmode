import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size" : 10
})


opj = os.path.join

ilc_labels = {'nilc_dsbdbs' : 'Joint NILC\n' r'($\mathtt{c+d+s+dbd+dbs}$)',
              'hilc_dsbdbs' : 'Joint HILC\n' r'($\mathtt{c+d+s+dbd+dbs}$)'}

ilc_rundirs = {'nilc_dsbdbs' : 'run65t',
               'hilc_dsbdbs' : 'run71t'}

ilc_sampledirs = {'nilc_dsbdbs' : 'sample65t',
                  'hilc_dsbdbs' : 'sample71t'}

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
imgdir = opj(basedir, 'sample65t/img_sigma_r')
os.makedirs(imgdir, exist_ok=True)

lmin = 2
lmax = 200

def hdi_unimodal(samples, cred_mass=0.68):
    '''
    Compute the Highest Density Interval (HDI) from samples only.
    Works for unimodal 1D distributions.

    Parameters
    ----------
    samples : (nsamp) array-like
        1D posterior samples.
    cred_mass : float
        Credible mass, e.g. 0.68, 0.95.

    Returns
    -------
    (low, high) : tuple
        Lower and upper bounds of the HDI.
    '''
    
    x = np.sort(np.asarray(samples))
    
    n = x.size
    if n == 0:
        raise ValueError("No samples")
    
    k = int(np.ceil(cred_mass * n))  # ensure coverage >= cred_mass
    if k < 1 or k > n:
        raise ValueError("cred_mass leads to invalid window length")
    
    # Candidate windows [i, i+k-1].
    widths = x[k-1:] - x[:n-k+1]
    j = np.argmin(widths)
    
    return (x[j], x[j+k-1])

def get_noise_variance(amp, gamma, lmin=2, lmax=2000):
    '''
    Convert amplitude and gamma parameters to noise variance.

    Parameters
    ----------
    amp : (nsamp) array or float
        Amplitude.
    gamma : (nsamp) array or float
        Gamma power law index.
    lmin : int, optional
        Lower limit for variance computation.
    lmax : int, optional
        Upper limit for variance computation.

    Returns
    -------
    var : (nsamp, 1) or float
        Variance.    
    '''
    
    # If amp and gamma are 2d, create 2d c_ells
    amp = np.atleast_1d(np.asarray(amp))
    gamma = np.atleast_1d(np.asarray(gamma))

    assert amp.shape == gamma.shape
    if amp.size == 1:
        scalar_input = True
        amp = amp[np.newaxis,:]
        gamma = gamma[np.newaxis,:]
    else:
        amp = amp[:,np.newaxis]
        gamma = gamma[:,np.newaxis]        
        scalar_input = False
    nsamp = amp.shape[0]
        
    ell_pivot = 1
    ells = np.arange(lmin, lmax+1)
    c_ell = np.zeros((nsamp, lmax+1))
    c_ell[:,lmin:lmax+1] = amp * (ells / ell_pivot) ** gamma    
    out = np.sum((2 * ells + 1) / (4 * np.pi) * c_ell[:,lmin:lmax+1], axis=-1)

    if scalar_input:
        return out[0]
    else:
        return out

def get_arrays(idir, idir_samples):
    '''

    Parameters
    ----------
    idir : str
        Path to dir containing param_draws_test.npy file.
    idir_samples : str
        Path to dir containing samples_test.npy file.
    '''
    
    # Load posterior samples
    samples = np.load(opj(idir_samples, 'samples_test.npy'))
    # NOTE
    samples = samples[:,:500,:]

    # Load True parameters
    params = np.load(opj(idir, 'param_draws_test.npy'))

    nsamp = samples.shape[1]

    # Convert gamma parameters to sigma_gamma
    print(samples.shape)
    print(params.shape)

    nsims = samples.shape[0]
    var_beta_dust = np.zeros(nsims)
    for sidx in range(nsims):
        var_beta_dust[sidx] = get_noise_variance(params[sidx,5], params[sidx,6], lmin=lmin, lmax=lmax)

    var_beta_sync = np.zeros(nsims)
    for sidx in range(nsims):
        var_beta_sync[sidx] = get_noise_variance(params[sidx,10], params[sidx,11], lmin=lmin, lmax=lmax)

    var_beta_dust_est = np.zeros((nsims,nsamp))
    for sidx in range(nsims):
        var_beta_dust_est[sidx] = get_noise_variance(samples[sidx,:,5], samples[sidx,:,6], lmin=lmin, lmax=lmax)    

    var_beta_sync_est = np.zeros((nsims,nsamp))
    for sidx in range(nsims):
        var_beta_sync_est[sidx] = get_noise_variance(samples[sidx,:,10], samples[sidx,:,11], lmin=lmin, lmax=lmax)    

    sigma_comb = np.sqrt(var_beta_dust + var_beta_sync)
    sigma_comb_est = np.sqrt(var_beta_dust_est + var_beta_sync_est)
    sigma_comb_est_mean = np.mean(sigma_comb_est, axis=-1)

    sigma_comb_est_hdi = np.zeros((nsims, 2))
    for idx in range(nsims):
        sigma_comb_est_hdi[idx] = hdi_unimodal(sigma_comb_est[idx,:])

    sigma_r = np.zeros(nsims)
    for idx in range(nsims):
        sigma_r[idx] = np.abs(np.diff(hdi_unimodal(samples[idx,:,0]))) / 2.
    mean_r = np.mean(samples[:,:,0], axis=1)

    # Sort arrays.
    #idx_sorted = np.argsort(sigma_comb)[::-1]
    #idx_sorted = np.argsort(sigma_comb)
    #sigma_comb_est_hdi_sorted = sigma_comb_est_hdi[idx_sorted]
    #sigma_r_sorted = sigma_r[idx_sorted]

    return sigma_r, sigma_comb, sigma_comb_est

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

sigma_r_nilc, sigma_comb_nilc, sigma_comb_est_nilc = get_arrays(
    opj(basedir, ilc_rundirs['nilc_dsbdbs']), opj(basedir, ilc_sampledirs['nilc_dsbdbs']))

sigma_r_hilc, sigma_comb_hilc, sigma_comb_est_hilc = get_arrays(
    opj(basedir, ilc_rundirs['hilc_dsbdbs']), opj(basedir, ilc_sampledirs['hilc_dsbdbs']))

idx_sorted_true = np.argsort(sigma_comb_nilc)[::-1]

fig, axs =plt.subplots(nrows=2, dpi=300, constrained_layout=True, sharex=True)
axs[0].plot(sigma_comb_nilc[idx_sorted_true], sigma_r_hilc[idx_sorted_true], label='HILC')
axs[0].plot(sigma_comb_nilc[idx_sorted_true], sigma_r_nilc[idx_sorted_true], label='NILC')

axs[0].plot(sigma_comb_nilc[idx_sorted_true], smooth_tophat(sigma_r_hilc[idx_sorted_true], binsize=5), label='HILC smooth')
axs[0].plot(sigma_comb_nilc[idx_sorted_true], smooth_tophat(sigma_r_nilc[idx_sorted_true], binsize=5), label='HILC smooth')
axs[0].legend(frameon=False)

axs[1].plot(sigma_comb_nilc[idx_sorted_true],
            sigma_r_nilc[idx_sorted_true] / sigma_r_hilc[idx_sorted_true], label='NILC / HILC')
axs[1].plot(sigma_comb_nilc[idx_sorted_true],
            smooth_tophat(sigma_r_nilc[idx_sorted_true] / sigma_r_hilc[idx_sorted_true], binsize=5),
            label='NILC / HILC smooth')
axs[1].set_ylim(0.5, 1.5)
axs[0].legend(frameon=False)

axs[1].set_xscale('log')
axs[0].grid(True)
axs[1].grid(True)

fig.savefig(opj(imgdir, 'sigma_r_nilc_vs_hilc'))
plt.close(fig)
