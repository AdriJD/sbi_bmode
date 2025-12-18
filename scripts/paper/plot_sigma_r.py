import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'serif',
         'size'   : 12,
         'serif':  'cmr10'
         }

matplotlib.rc('font', **font)

opj = os.path.join

#idir = '/u/adriaand/project/so/20240521_sbi_bmode/run52bt'
##idir_samples = '/u/adriaand/project/so/20240521_sbi_bmode/sample52bt'
##idir_samples = '/u/adriaand/project/so/20240521_sbi_bmode/sample52bt_b'
#idir = '/u/adriaand/project/so/20240521_sbi_bmode/run52bt2'
#idir_samples = '/u/adriaand/project/so/20240521_sbi_bmode/sample52bt'

idir = '/u/adriaand/project/so/20240521_sbi_bmode/run65t'
idir_samples = '/u/adriaand/project/so/20240521_sbi_bmode/sample65t'

imgdir = opj(idir_samples, 'img_sigma_r')
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
    #out = get_noise_variance_from_c_ell(c_ell, lmin, lmax)
    if scalar_input:
        return out[0]
    else:
        return out

def get_noise_variance_from_c_ell(c_ell, lmin, lmax):
    
    c_ell = c_ell[lmin:lmax+1]
    ells = np.arange(lmin, lmax+1)
    
    return np.sum((2 * ells + 1) / (4 * np.pi) * c_ell)

# Load posterior samples
samples = np.load(opj(idir_samples, 'samples_test.npy'))
# NOTE
#samples = samples[:,:1000,:]

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

#sigma_comb_real = np.sqrt(var_beta_dust_real + var_beta_sync_real)
#sigma_r = np.std(samples[:,:,0], axis=1)
sigma_r = np.zeros(nsims)
for idx in range(nsims):
    sigma_r[idx] = np.abs(np.diff(hdi_unimodal(samples[idx,:,0]))) / 2.
mean_r = np.mean(samples[:,:,0], axis=1)

print(sigma_comb_est.shape)
print(sigma_comb_est[0])
fig, ax = plt.subplots(dpi=300)
for idx in range(10):
    ax.hist(sigma_comb_est[idx,:], density=True, bins=50, histtype='step')
    hdi_min, hdi_max = hdi_unimodal(sigma_comb_est[idx,:])
    ax.axvline(hdi_min, color=f'C{idx}')
    ax.axvline(hdi_max, color=f'C{idx}')    
fig.savefig(opj(imgdir, f'sigma_comb_est'))
plt.close(fig)

# NOTE
#sigma_comb = np.sqrt(var_beta_sync)
#sigma_comb_est = np.sqrt(var_beta_sync_est)
#sigma_comb_real = np.sqrt(var_beta_sync_real)

# plot
for sidx in range(10):
    print(np.sqrt(get_noise_variance(params[sidx,5], params[sidx,6])))


# NOTE Change from sigma to HPD region or whateer it's called
# Color points based on gamma.

#print(sigma_comb[:10])
#print(sigma_comb_real[:10])

fig, ax = plt.subplots(dpi=300)
#sp = ax.scatter(sigma_comb, sigma_r, c=params[:,6], cmap='viridis', s=0.5)
#sp = ax.scatter(sigma_comb, sigma_r, c=np.mean(samples[:,:,5], axis=1), cmap='viridis', s=0.5)
#sp = ax.scatter(sigma_comb, sigma_r, c=sigma_comb_est - sigma_comb, cmap='viridis', s=0.5)
sp = ax.scatter(sigma_comb, sigma_r, c=sigma_comb_est_mean, cmap='viridis', s=0.5)
cbar = fig.colorbar(sp, ax=ax)
ax.set_xscale('log')
#ax.set_xlim(0.01)
ax.set_yscale('log')
#ax.set_ylim(0.0001)
#for i in range(nsims):
#    plt.text(sigma_comb[i], sigma_r[i], str(i), fontsize=4, ha='right', va='bottom')

fig.savefig(opj(imgdir, 'sigma_r'))
plt.close(fig)

# fig, ax = plt.subplots(dpi=300)
# sp = ax.scatter(sigma_comb_real, sigma_r, c=sigma_comb, cmap='viridis', s=0.5)
# cbar = fig.colorbar(sp, ax=ax)
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig.savefig(opj(imgdir, 'sigma_r_real'))
# plt.close(fig)

# Sort arrays.
idx_sorted = np.argsort(sigma_comb)
sigma_comb_est_hdi_sorted = sigma_comb_est_hdi[idx_sorted]
sigma_r_sorted = sigma_r[idx_sorted]

fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(4.5, 3))
#sp = ax.scatter(sigma_comb_est, sigma_r, s=1, edgecolors='black', facecolors='none', alpha=0.8, linewidths=0.5)
for idx in range(nsims):
    x_arr = np.asarray(sigma_comb_est_hdi_sorted[idx])
    y_arr = np.zeros(2) + sigma_r_sorted[idx]
    ax.plot(x_arr, y_arr, color='grey', lw=0.1, zorder=0)
sp = ax.scatter(sigma_comb_est_mean[idx_sorted], sigma_r[idx_sorted], s=3,
                c=sigma_comb[idx_sorted],
                alpha=1, linewidths=0.5,
                edgecolor='face',
                norm=matplotlib.colors.LogNorm(vmin=sigma_comb.min(), vmax=1.))
cbar = fig.colorbar(sp, ax=ax)
cbar.set_label(r'$\sqrt{\sigma^2_{\beta_\mathrm{d}} + \sigma^2_{\beta_\mathrm{s}}}$')
ax.set_xscale('log', base=10)
ax.set_xlim(0.07, 1)
#ax.grid(color='gray', linestyle='-', linewidth=0.5, zorder=0, which='both')
ax.set_yscale('log')
ax.set_ylim(6e-4, 2.5e-2)
ax.set_xlabel(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$')
ax.set_ylabel(r'$\sigma_{r}$')
ax.tick_params('both', which='both', direction='in', right=True, top=True)
fig.savefig(opj(imgdir, 'sigma_r_var_est'))
plt.close(fig)

# fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(4, 3))
# sp = ax.scatter(sigma_comb_est, sigma_r, s=1, c=sigma_comb_real, alpha=0.8, linewidths=0.5)
# cbar = fig.colorbar(sp, ax=ax)
# ax.grid(color='black', linestyle='-', linewidth=0.5, zorder=0)
# ax.set_yscale('log')
# ax.set_xlabel(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$')
# ax.set_ylabel(r'$\sigma_{r}$')
# fig.savefig(opj(imgdir, 'sigma_r_var_real'))
# plt.close(fig)


fig, ax = plt.subplots(dpi=300)
ax.errorbar(sigma_comb, mean_r, yerr=sigma_r, ls='none', fmt='o', elinewidth=0.2, ms=0.5)
#for i in range(nsims):
#    plt.text(sigma_comb[i], mean_r[i], str(i), fontsize=4, ha='right', va='bottom')
ax.set_xscale('log')
#ax.set_xlim(0.005)
#ax.set_yscale('log')
#ax.set_ylim(0.0001)
ax.grid('true')
fig.savefig(opj(imgdir, 'mean_r'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)
ax.errorbar(sigma_comb_est_mean, mean_r, yerr=sigma_r, ls='none', fmt='o', elinewidth=0.2, ms=0.5)
#for i in range(nsims):
#    plt.text(sigma_comb[i], mean_r[i], str(i), fontsize=4, ha='right', va='bottom')
ax.set_xscale('log')
ax.grid('true')
fig.savefig(opj(imgdir, 'mean_r_var_est'))
plt.close(fig)

