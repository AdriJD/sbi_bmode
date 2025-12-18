import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#font = {'family' : 'serif',
#         'size'   : 10,
#         'serif':  'cmr10'
#         }

#matplotlib.rc('font', **font)

matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size" : 10
})


opj = os.path.join


#ilc_labels = {'nilc_dsbdbs' : 'Joint NILC\n(' r'$\mathrm{CMB}$' ', ' r'$\mathrm{d}$, ' r'$\mathrm{s}$,' \
#              '\n' r'$\delta \beta_{\mathrm{d}}$, $\delta \beta_{\mathrm{s}}$)',
#              'hilc_dsbdbs' : 'Joint HILC\n(' r'$\mathrm{CMB}$' ', ' r'$\mathrm{d}$, ' r'$\mathrm{s}$,' \
#              '\n' r'$\delta \beta_{\mathrm{d}}$, $\delta \beta_{\mathrm{s}}$)',
#              'nilc_ds' : 'Joint NILC\n(' r'$\mathrm{CMB}$' ', ' r'$\mathrm{d}$, ' r'$\mathrm{s}$)',
#              'nilc' : 'NILC \n(' r'$\mathrm{CMB}$' ')',              
#              'nilc_cds' : 'Constrained NILC\n(' r'$\mathrm{CMB}$' ', ' r'$\mathrm{d}$, ' r'$\mathrm{s}$)',
#              'nilc_cdsbdbs' : 'Constrained NILC\n(' r'$\mathrm{CMB}$' ', ' r'$\mathrm{d}$, ' r'$\mathrm{s}$,' \
#              '\n' r'$\delta \beta_{\mathrm{d}}$, $\delta \beta_{\mathrm{s}}$)'
#              }

ilc_labels = {'nilc_dsbdbs' : 'Joint NILC\n' r'($\mathtt{c+d+s+dbd+dbs}$)',
              'hilc_dsbdbs' : 'Joint HILC\n' r'($\mathtt{c+d+s+dbd+dbs}$)',
              'nilc_ds'     : 'Joint NILC\n' r'($\mathtt{c+d+s}$)',
              'nilc' : 'NILC \n' r'($\mathtt{c}$)',              
              'nilc_cds' : 'Constrained NILC\n' r'($\mathtt{c+d+s}$',
              'nilc_cdsbdbs' : 'Constrained NILC\n' r'($\mathtt{c+d+s+dbd+dbs}$)'
              }

ilc_rundirs = {'nilc_dsbdbs' : 'run65t',
               'hilc_dsbdbs' : 'run71t',
               'nilc' : 'run67t',
               #'nilc_ds' : 'run68t',
               #'nilc_cds' : 'run69t',
               'nilc_ds' : 'run87t',
               'nilc_cds' : 'run88t',               
               'nilc_cdsbdbs' : 'run70t'}

ilc_sampledirs = {'nilc_dsbdbs' : 'sample65t',
                  'hilc_dsbdbs' : 'sample71t',
                  'nilc' : 'sample67t',
                  #'nilc_ds' : 'sample68t',
                  #'nilc_cds' : 'sample69t',
                  'nilc_ds' : 'sample87t',
                  'nilc_cds' : 'sample88t',                  
                  'nilc_cdsbdbs' : 'sample70t'}

sigma_d1s5 = np.sqrt(0.0844 ** 2 + 0.0285 ** 2)
sigma_d10s5 = np.sqrt(0.0844 ** 2 + 0.1319 ** 2)
sigma_d12s5 = np.sqrt(0.0844 ** 2 + 0.1041 ** 2)
models = ['d1_s5', 'd10_s5', 'd12_s5']
sigma_per_model = {'d1_s5' : sigma_d1s5, 'd10_s5' : sigma_d10s5, 'd12_s5' : sigma_d12s5}
sigma_r_per_model = np.zeros((6, 3)) # n_nilc, n_model.

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

    
fig, axs = plt.subplots(dpi=300, nrows=3, ncols=2, constrained_layout=True, figsize=(7.1, 8.3),
                        sharex=True, sharey=True)

for aidx, (ax, ilc_type) in enumerate(zip(axs.ravel(), ilc_labels.keys())):

    print(ax, ilc_type)
    idir = opj(basedir, ilc_rundirs[ilc_type])
    idir_samples = opj(basedir, ilc_sampledirs[ilc_type])    
    
    # Load posterior samples
    samples = np.load(opj(idir_samples, 'samples_test.npy'))
    # NOTE
    #samples = samples[:,:500,:]

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
    idx_sorted = np.argsort(sigma_comb)
    sigma_comb_est_hdi_sorted = sigma_comb_est_hdi[idx_sorted]
    sigma_r_sorted = sigma_r[idx_sorted]

    for idx in range(nsims):
        x_arr = np.asarray(sigma_comb_est_hdi_sorted[idx])
        y_arr = np.zeros(2) + sigma_r_sorted[idx]
        #ax.plot(x_arr, y_arr, color='dimgrey', lw=0.2, zorder=0, alpha=0.8)
        ax.plot(x_arr, y_arr, color='grey', lw=0.2, zorder=0, alpha=0.8)        
        
    if aidx == 0:
        # Determine colorbar normalization, should be the same for all panels because the
        # true sigma parameters are the same for all.
        norm = matplotlib.colors.LogNorm(vmin=sigma_comb.min(), vmax=1.)
    sp = ax.scatter(sigma_comb_est_mean[idx_sorted], sigma_r[idx_sorted], s=3,
                    c=sigma_comb[idx_sorted],
                    #alpha=0.7, linewidths=0.5,
                    alpha=1, linewidths=0.5,                    
                    edgecolor='face',
                    norm=norm, cmap='viridis_r')
    ax.text(
        #0.9765, 0.024, ilc_labels[ilc_type],
        0.95, 0.05, ilc_labels[ilc_type],         
        transform=ax.transAxes,    # position in axes coords (0–1)
        fontsize=10, #fontweight="bold",
        ha='right', va='bottom', 
        bbox=dict(facecolor="white", edgecolor="black", pad=3.5),
        multialignment="right"
    )

    # Determine average sigma_r for some of the PySM models.
    for midx, model in enumerate(models):

        s_b = sigma_per_model[model]
        mask = (sigma_comb > 0.90 * s_b) & (sigma_comb < 1.10 * s_b)
        print(ilc_type, model, s_b, mask.sum(), np.mean(sigma_r[mask]), np.std(sigma_r[mask]))
        sigma_r_per_model[aidx,midx] = np.mean(sigma_r[mask])

with open(opj(imgdir, 'sigma_r_table.txt'), "w") as f:
    
    f.write("\t" + "\t".join(models) + "\n")
    for label, row in zip(ilc_labels.keys(), sigma_r_per_model):
        row_str = "\t".join(f"{val:.6g}" for val in row)
        f.write(f"{label}\t{row_str}\n")
    
cbar = fig.colorbar(sp, ax=axs, fraction=0.05, pad=0.02, shrink=0.5)
cbar.set_label(r'$\sqrt{\sigma^2_{\beta_\mathrm{d}} + \sigma^2_{\beta_\mathrm{s}}}$', labelpad=-8,
               size=12)
    
axs[-1,-1].set_xscale('log', base=10)
axs[-1,-1].set_xlim(0.07, 1)

axs[-1,-1].set_yscale('log')
axs[-1,-1].set_ylim(6e-4, 2.5e-2)
#axs.set_xlabel(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$')
#ax.set_ylabel(r'$\sigma_{r}$')
fig.supxlabel(r'$\sqrt{\hat{\sigma}^2_{\beta_\mathrm{d}} + \hat{\sigma}^2_{\beta_\mathrm{s}}}$',
              x=0.46)
#fig.supylabel(r'$\sigma_{r}$', x=0.02)
fig.supylabel(r'$\sigma_{r}$', y=0.54)

for ax in axs.ravel():
    ax.tick_params('both', which='both', direction='in', right=True, top=True)
    ax.grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
    ax.set_axisbelow(True)
fig.savefig(opj(imgdir, 'sigma_r_var_est_panel'))
fig.savefig(opj(imgdir, 'sigma_r_var_est_panel.pdf'))
plt.close(fig)
