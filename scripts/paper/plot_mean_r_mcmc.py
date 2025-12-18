import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size" : 10
})

#font = {'family' : 'serif',
#         'size'   : 10,
#         'serif':  'cmr10'
#         }

#matplotlib.rc('font', **font)

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
imgdir = opj(basedir, 'sample65t/img_sigma_r')
os.makedirs(imgdir, exist_ok=True)

ilc_rundir = 'run65t'
ilc_sampledir = 'sample65t'
mcmc_sampledir = 'mcmc65t_test2'

idir = opj(basedir, ilc_rundir)
idir_samples_ilc = opj(basedir, ilc_sampledir)
idir_samples_mcmc = opj(basedir, mcmc_sampledir)

np.random.seed(0)
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

# Load posterior samples
samples_ilc = np.load(opj(idir_samples_ilc, 'samples_test.npy'))
samples_mcmc = np.load(opj(idir_samples_mcmc, 'samples.npy'))

print(samples_mcmc[0,0,:20,0])
print(samples_mcmc.shape)

# NOTE
#samples_ilc = samples_ilc[::10,:,:]
#samples_mcmc = samples_mcmc[::10,:,:,:]

# Rescale r.
samples_ilc[...,0] *= 100
samples_mcmc[...,0] *= 100

# Flatten chain dimension.
samples_mcmc = samples_mcmc.reshape(
    samples_mcmc.shape[0], samples_mcmc.shape[1] * samples_mcmc.shape[2], samples_mcmc.shape[3])

# Load True parameters
params = np.load(opj(idir, 'param_draws_test.npy'))

nsamp_ilc = samples_ilc.shape[1]
nsamp_mcmc = samples_mcmc.shape[1]

# Convert gamma parameters to sigma_gamma
nsims = samples_ilc.shape[0]

var_beta_dust = np.zeros(nsims)
for sidx in range(nsims):
    var_beta_dust[sidx] = get_noise_variance(params[sidx,5], params[sidx,6], lmin=lmin, lmax=lmax)

var_beta_sync = np.zeros(nsims)
for sidx in range(nsims):
    var_beta_sync[sidx] = get_noise_variance(params[sidx,10], params[sidx,11], lmin=lmin, lmax=lmax)

var_beta_dust_est = np.zeros((nsims, nsamp_ilc))
for sidx in range(nsims):
    var_beta_dust_est[sidx] = get_noise_variance(samples_ilc[sidx,:,5], samples_ilc[sidx,:,6], lmin=lmin, lmax=lmax)

var_beta_sync_est = np.zeros((nsims, nsamp_ilc))
for sidx in range(nsims):
    var_beta_sync_est[sidx] = get_noise_variance(samples_ilc[sidx,:,10], samples_ilc[sidx,:,11], lmin=lmin, lmax=lmax)

sigma_comb = np.sqrt(var_beta_dust + var_beta_sync)
sigma_comb_est = np.sqrt(var_beta_dust_est + var_beta_sync_est)
sigma_comb_est_mean = np.mean(sigma_comb_est, axis=-1)


sigma_comb_est_hdi = np.zeros((nsims, 2))
for idx in range(nsims):
    sigma_comb_est_hdi[idx] = hdi_unimodal(sigma_comb_est[idx,:])

hdi_r_ilc = np.zeros((nsims, 2))
hdi_r_mcmc = np.zeros((nsims, 2))
for idx in range(nsims):
    hdi_r_ilc[idx] = hdi_unimodal(samples_ilc[idx,:,0])
    hdi_r_mcmc[idx] = hdi_unimodal(samples_mcmc[idx,:,0])
mean_r_ilc = np.mean(samples_ilc[:,:,0], axis=1)
mean_r_mcmc = np.mean(samples_mcmc[:,:,0], axis=1)

# Sort arrays.
#idx_sorted = np.argsort(sigma_comb)[::-1]
idx_sorted = np.argsort(sigma_comb_est_mean)[::-1]
print(f'{idx_sorted=}')
sigma_comb_est_mean_sorted = sigma_comb_est_mean[idx_sorted]

#sigma_comb_sorted = sigma_comb[idx_sorted]

#sigma_comb_est_hdi_sorted = sigma_comb_est_hdi[idx_sorted]
hdi_r_ilc_sorted = hdi_r_ilc[idx_sorted]
hdi_r_mcmc_sorted = hdi_r_mcmc[idx_sorted]

mean_r_ilc_sorted = mean_r_ilc[idx_sorted]
mean_r_mcmc_sorted = mean_r_mcmc[idx_sorted]


sigma_mcmc = np.sum(np.abs(hdi_r_mcmc_sorted - mean_r_mcmc_sorted[:,np.newaxis]), axis=-1) / 2
sigma_mcmc = sigma_mcmc.ravel()
print(sigma_mcmc.shape)
mcmc_mask = sigma_mcmc > (1.4 * sigma_mcmc[-1])
sigma_mcmc_masked = sigma_mcmc.copy()
sigma_mcmc_masked[mcmc_mask] = np.nan
print(sigma_mcmc[-1])
fig, ax = plt.subplots(dpi=300)
ax.plot(sigma_mcmc)
ax.plot(sigma_mcmc_masked)
ax.set_ylim(-0.1, 0.5)
fig.savefig(opj(imgdir, 'hdi_r_mcmc_sorted'))
plt.close(fig)

nbins = 15
n_per_bin = 10
bins = np.logspace(np.log10(sigma_comb_est_mean_sorted.min()),
                   np.log10(sigma_comb_est_mean_sorted.max()), nbins + 1)
selected_indices = []

for i in range(nbins):
    if i < nbins - 1:
        # For all but the last bin, include lower edge, exclude upper edge
        in_bin = np.where((sigma_comb_est_mean_sorted >= bins[i]) & (sigma_comb_est_mean_sorted < bins[i + 1]))[0]
    else:
        # For the last bin, include the upper edge too
        in_bin = np.where((sigma_comb_est_mean_sorted >= bins[i]) & (sigma_comb_est_mean_sorted <= bins[i + 1]))[0]

    if len(in_bin) >= n_per_bin:
        chosen = np.random.choice(in_bin, n_per_bin, replace=False)
    else:
        chosen = in_bin

    selected_indices.append(chosen)

selected_indices = np.concatenate(selected_indices)

sigma_comb_est_mean_sorted = sigma_comb_est_mean_sorted[selected_indices]
hdi_r_ilc_sorted = hdi_r_ilc_sorted[selected_indices]
hdi_r_mcmc_sorted = hdi_r_mcmc_sorted[selected_indices]
mean_r_ilc_sorted = mean_r_ilc_sorted[selected_indices]
mean_r_mcmc_sorted = mean_r_mcmc_sorted[selected_indices]
mcmc_mask = mcmc_mask[selected_indices]


for idx in range(10):
    print(hdi_r_ilc_sorted[-idx], mean_r_ilc_sorted[-idx])
print(np.abs(hdi_r_ilc_sorted[:10] - mean_r_ilc_sorted[:10][:,None]))
print(np.abs((hdi_r_ilc_sorted.T)[:,:10] - mean_r_ilc_sorted[np.newaxis,:10]))

#print(sigma_comb_sorted)
fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(7.1, 3.5),
                        sharex=True, sharey=True)

for idx in range(sigma_comb_est_mean_sorted.size):
    if mcmc_mask[idx]:
        continue
    x_arr = np.zeros(2) + np.asarray(sigma_comb_est_mean_sorted[idx])
    y_arr = np.asarray([mean_r_ilc_sorted[idx], mean_r_mcmc_sorted[idx]])
    #ax.plot(x_arr, y_arr, color='dimgrey', lw=0.2, zorder=0, alpha=0.8)
    ax.plot(x_arr, y_arr, color='grey', lw=0.2, zorder=0, alpha=0.8)

norm = matplotlib.colors.LogNorm(vmin=sigma_comb.min(), vmax=1.)
#ax.scatter(sigma_comb_sorted, mean_r_ilc_sorted, s=3,
#           c=sigma_comb_est_mean[idx_sorted],
#           alpha=0.7, linewidths=0.5,
#           edgecolor='face',
#           norm=norm, cmap='viridis_r')

sigma_d1s5 = np.sqrt(0.0844 ** 2 + 0.0285 ** 2)
sigma_d10s5 = np.sqrt(0.0844 ** 2 + 0.1319 ** 2)
sigma_d12s5 = np.sqrt(0.0844 ** 2 + 0.1041 ** 2)

ax.axvline(sigma_d1s5, ls='dashed', color='gray', lw=1)
ax.axvline(sigma_d10s5, ls='dashed', color='gray', lw=1)
ax.axvline(sigma_d12s5, ls='dashed', color='gray', lw=1)

ax.text(0.99 * sigma_d1s5 , 5.1, r'$\mathtt{d1\_s5}$', rotation='vertical', horizontalalignment='right',
        verticalalignment='top')
ax.text(0.99 * sigma_d10s5 , 5.1, r'$\mathtt{d10\_s5}$', rotation='vertical', horizontalalignment='right',
        verticalalignment='top')
ax.text(0.99 * sigma_d12s5 , 5.1, r'$\mathtt{d12\_s5}$', rotation='vertical', horizontalalignment='right',
        verticalalignment='top')

err = ax.errorbar(sigma_comb_est_mean_sorted[~mcmc_mask], mean_r_mcmc_sorted[~mcmc_mask],
                  yerr=np.abs((hdi_r_mcmc_sorted[~mcmc_mask]).T - (mean_r_mcmc_sorted[~mcmc_mask])[np.newaxis,:]),
                  fmt='x', color='crimson', capsize=2, elinewidth=0.1, capthick=0.5,
                  markersize=2, label='Multi-freq. likelihood (w/o moment marg.)')


ax.errorbar(sigma_comb_est_mean_sorted, mean_r_ilc_sorted,
            yerr=np.abs(hdi_r_ilc_sorted.T - mean_r_ilc_sorted[np.newaxis,:]),
            fmt='o', color='black', capsize=2, elinewidth=0.5, capthick=0.5,
            markersize=1.5, label='Joint NILC SBI')
            #s=3,
            #    c=sigma_comb_est_mean[idx_sorted],
            #    alpha=0.7, linewidths=0.5,
            #    edgecolor='face',
            #    norm=norm, cmap='viridis_r')

#sp = ax.scatter(sigma_comb_sorted, mean_r_mcmc_sorted, s=3,
#           c=sigma_comb_est_mean_sorted,
#           alpha=0.7, linewidths=0.5,
#           norm=norm, cmap='viridis_r', edgecolor='face')
# sp = ax.scatter(sigma_comb_est_mean_sorted, mean_r_mcmc_sorted, s=3,
#            c=sigma_comb_sorted,
#            alpha=0.7, linewidths=0.5,
#            norm=norm, cmap='viridis_r', edgecolor='face')

#for bar in err[2]:
#    bar.set_alpha(0.6)

#ax.legend(frameon=False, loc=(0.40, 0.82))
ax.legend(frameon=False, loc=(0.43, 0.82))
#for b in bins:
#    ax.axvline(b)
    
#cbar = fig.colorbar(sp, ax=ax, fraction=0.05, pad=0.02, shrink=0.5)
#cbar.set_label(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$', labelpad=-8,
#               size=12)

ax.set_xscale('log', base=10)
ax.set_xlim(7e-2, 1)
#axs[-1,-1].set_xlim(0.07, 1)

#axs[-1,-1].set_yscale('log')
#axs[-1,-1].set_ylim(6e-4, 2.5e-2)
#axs.set_xlabel(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$')
#ax.set_ylabel(r'$\sigma_{r}$')
ax.set_xlabel(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$', fontsize=12)
#fig.supylabel(r'$\sigma_{r}$', x=0.02)
ax.set_ylabel(r'$100r$', fontsize=12)
ax.set_ylim(-0.001 * 100, 0.052 * 100)
#for ax in axs.ravel():
ax.tick_params('both', which='both', direction='in', right=True, top=True)
ax.axhline(0.01 * 100, color='black', lw=0.5, ls='dotted')
#ax.grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
ax.set_axisbelow(True)
fig.savefig(opj(imgdir, 'mean_r_mcmc'))
fig.savefig(opj(imgdir, 'mean_r_mcmc.pdf'))
plt.close(fig)
