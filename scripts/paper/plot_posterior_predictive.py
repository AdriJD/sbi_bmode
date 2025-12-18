import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.gridspec as gridspec
import torch

from sbi_bmode import compress_utils

matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size" : 10
})

opj = os.path.join

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

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
priordir = opj(basedir, 'run65')
covdir = opj(basedir, 'run65cov')

fg_combs = ['d1_s5', 'd1_s7', 'd4_s5', 'd10_s5', 'd10alt_s5', 'd12_s5']
fg_names = ['d1\_s5', 'd1\_s7', 'd4\_s5', 'd10\_s5', 'd10x1.6\_s5', 'd12\_s5']
subdir_dict = {'d1_s5' : {'postdir' : 'run65post_d1s5', 'datadir' : 'run76t_d1s5'},
               'd1_s7' : {'postdir' : 'run65post_d1s7', 'datadir' : 'run76t_d1s7'},
               'd4_s5' : {'postdir' : 'run65post_d4s5', 'datadir' : 'run76t_d4s5'}, 
               'd10_s5' : {'postdir' : 'run65post_d10s5', 'datadir' : 'run76t_d10s5'},
               'd10alt_s5' : {'postdir' : 'run65post_d10alts5', 'datadir' : 'run76t_d10alts5'},
               'd12_s5' : {'postdir' : 'run65post_d12s5', 'datadir' : 'run76t_d12s5'}}
title_dict = {'d1_s5' : r'$\mathtt{d1\_s5}$', 'd1_s7' : r'$\mathtt{d1\_s7}$',
              'd4_s5' : r'$\mathtt{d4\_s5}$', 'd10_s5' : r'$\mathtt{d10\_s5}$',
              'd10alt_s5' : r'$\mathtt{d10x1.6\_s5}$', 'd12_s5' : r'$\mathtt{d12\_s5}$'}

spec_names = [r'$(\mathtt{c}$,$\mathtt{c})$', r'$(\mathtt{c}$,$\mathtt{d})$', r'$(\mathtt{c}$,$\mathtt{dbd})$',
              r'$(\mathtt{c}$,$\mathtt{s})$', r'$(\mathtt{c}$,$\mathtt{dbs})$', r'$(\mathtt{d}$,$\mathtt{d})$',
              r'$(\mathtt{d}$,$\mathtt{dbd})$', r'$(\mathtt{d}$,$\mathtt{s})$', r'$(\mathtt{d}$,$\mathtt{dbs})$',
              r'$(\mathtt{dbd}$,$\mathtt{dbd})$', r'$(\mathtt{dbd}$,$\mathtt{s})$', r'$(\mathtt{dbd}$,$\mathtt{dbs})$',
              r'$(\mathtt{s}$,$\mathtt{s})$', r'$(\mathtt{s}$,$\mathtt{dbs})$', r'$(\mathtt{dbs}$,$\mathtt{dbs})$']

imgdir = opj(basedir, 'pysm_compared')

mean = np.load(opj(priordir, 'data_mean.npy'))
std = np.load(opj(priordir, 'data_std.npy'))
prior_data = np.load(opj(priordir, 'data_draws_round_000.npy'))
posterior = np.load(opj(basedir, 'run65_optuna/trial_0120/posterior.pkl'), allow_pickle=True)

prior_data = compress_utils.unnormalize_simple(prior_data, mean, std)
nsim_prior = prior_data.shape[0]
prior_data = prior_data.reshape(nsim_prior, 15, -1)

alpha = 0.68
data_idx = 3
data_dict = {}

def get_hpd_mask(logprobs, alpha):
    '''
    Compute boolean mask that is True for samples inside HPD region.

    Parameters
    ----------
    logprobs : (nsim) array
        Log probabilities.
    alpha : float
        Level of the HPD region 0 <= alpha <= 1.
    
    Returns
    -------
    mask : (nsim) bool array
        Mask selecting samples inside the HPD region.
    '''

    assert logprobs.ndim == 1
    nsim = logprobs.size
    
    order = np.argsort(logprobs)[::-1]
    cum_fraction = np.arange(1, nsim + 1) / nsim

    idx = np.searchsorted(cum_fraction, alpha) + 1
    top_idxs = order[:idx]
    mask = np.zeros(nsim, dtype=bool)
    mask[top_idxs] = True

    return mask    

for fg_comb in fg_combs:

    postdir = opj(basedir, subdir_dict[fg_comb]['postdir'])
    datadir = opj(basedir, subdir_dict[fg_comb]['datadir'])    
    
    data = np.load(opj(datadir, 'data_draws_test.npy'))[data_idx]
    post_data = np.load(opj(postdir, 'data_draws_test.npy'))
    post_params = torch.as_tensor(np.load(opj(postdir, 'param_draws_test.npy')))

    # Determine 95% HPD.
    logprobs = posterior.potential(post_params, torch.as_tensor(data))
    logprobs = np.asarray(logprobs)    
    hpd_mask = get_hpd_mask(logprobs, alpha)
        
    data = compress_utils.unnormalize_simple(data, mean, std)
    post_data = compress_utils.unnormalize_simple(post_data, mean, std)

    data = data.reshape(15, -1)
    nsim_post = post_data.shape[0]
    post_data = post_data.reshape(nsim_post, 15, -1)

    data_dict[fg_comb] = {}
    data_dict[fg_comb]['data'] = data
    data_dict[fg_comb]['post_data'] = post_data
    data_dict[fg_comb]['hpd_mask'] = hpd_mask

    print(np.sum(hpd_mask))
    
nbins = prior_data.shape[-1]
bins = np.linspace(30, 200, num=nbins)
x_ticks = [50, 150]
x_labels = ["50", "150"]

#lidxs = np.asarray(np.tril_indices(5)).T
# Plot.

ylims = np.zeros((15, 2))
scalings = np.zeros(15)

for idx in range(15):    
    delta_data = np.max(data_dict['d1_s5']['data'][idx]) - np.min(data_dict['d1_s5']['data'][idx])
    scalings[idx] = 10 ** int(round(np.log10(delta_data), 0))
    
    ylims[idx] = np.asarray([np.min(data_dict['d1_s5']['data'][idx]) - 0.1 * delta_data,
                             np.max(data_dict['d1_s5']['data'][idx]) + 0.2 * delta_data]) / scalings[idx]
    
fig = plt.figure(figsize=(7.1, 8.5), dpi=300)
gs = gridspec.GridSpec(15, 6, figure=fig, wspace=0.0, hspace=0.0)
axs = np.empty((15, 6), dtype=object)

print(gs)

for aidx in range(15):
    #for ajdx in range(6):
    for ajdx, fg_comb in enumerate(fg_combs):

        print(aidx,ajdx)
        ax = fig.add_subplot(gs[aidx,ajdx])
        axs[aidx,ajdx] = ax
        #if ajdx == 0:
        for idx in range(0, nsim_prior, 100):
            axs[aidx,ajdx].plot(bins, prior_data[idx,aidx] / scalings[aidx], color='C0', alpha=0.2, lw=0.5)

        postdata2plot = data_dict[fg_comb]['post_data'][:,aidx]
            
        #for idx in range(nsim_post):            
        #    axs[aidx,ajdx].plot(bins, data_dict[fg_comb]['post_data'][idx,aidx]  / scalings[aidx],
        #                        color='black', alpha=0.2, lw=0.5)
        for pd in postdata2plot[~data_dict[fg_comb]['hpd_mask']]:
            axs[aidx,ajdx].plot(bins, pd  / scalings[aidx],
                                color='C1', alpha=0.5, lw=0.5)
        for pd in postdata2plot[data_dict[fg_comb]['hpd_mask']]:
            axs[aidx,ajdx].plot(bins, pd  / scalings[aidx],
                                #color='lavender', alpha=1, lw=0.5)
                                color='black', alpha=1, lw=0.5)            

        axs[aidx,ajdx].plot(bins, data_dict[fg_comb]['data'][aidx]  / scalings[aidx], color='crimson', lw=1, ls='dashed')

        if ajdx != 0:
            axs[aidx,ajdx].set_yticklabels([])
            axs[aidx,ajdx].set_ylabel('')             

        axs[aidx,ajdx].set_xticks(x_ticks)            
        if aidx == 14:
            axs[aidx,ajdx].set_xticklabels(x_labels)
        else:
            axs[aidx,ajdx].set_xticklabels([])            
            axs[aidx,ajdx].set_xlabel('')
            
        axs[aidx,ajdx].set_ylim(*ylims[aidx])
        
for ax in axs.flat:
    ax.tick_params('both', direction='in', right=True, top=True)
    
for ajdx in range(6):
    axs[0,ajdx].set_title(title_dict[fg_combs[ajdx]])

for aidx in range(15):
    scaling = scalings[aidx]
    if scaling != 1:
        scaling_str = r'$10^{' + str(int(np.log10(1 / scaling))) + '}$'
    else:
        scaling_str = '1'
    axs[aidx,0].text(
        0.982, 0.925, scaling_str + r'$\cdot$' + spec_names[aidx],
        transform=axs[aidx,0].transAxes,    # position in axes coords (0–1)
        fontsize=10, #fontweight="bold",
        ha='right', va='top', 
        bbox=dict(facecolor="white", edgecolor="black", pad=2.5),
        multialignment="right"
    )

    
fig.supxlabel(r'Bin center $b$', x=0.55, fontsize=12)
fig.supylabel(r'$\hat{C}_{b}$', y=0.52, fontsize=12)

    
fig.subplots_adjust(left=0.1, right=0.995, top=0.97, bottom=0.06)
    
fig.savefig(opj(imgdir, 'corner_post_pred'))
fig.savefig(opj(imgdir, 'corner_post_pred.pdf'))
plt.close(fig)

# Plot of chi^2
data_cov = np.load(opj(covdir, 'data_draws_test.npy'))
#data_cov = compress_utils.unnormalize_simple(data_cov, mean, std)
cov = np.cov(data_cov, rowvar=False)
d_mat = np.diag(np.diag(cov) ** -0.5)
cor = d_mat @ cov @ d_mat

# NOTE
#cov = np.diag(np.diag(cov))

icov = np.linalg.inv(cov)

fig, ax = plt.subplots(dpi=300)
ax.imshow(cov)
fig.savefig(opj(imgdir, 'cov'))
plt.close()

fig, ax = plt.subplots(dpi=300)
ax.imshow(cor)
fig.savefig(opj(imgdir, 'cor'))
plt.close()

fig, ax = plt.subplots(dpi=300)
ax.imshow(icov)
fig.savefig(opj(imgdir, 'icov'))
plt.close()

#fig, axs = plt.subplots(dpi=300, nrows=3, ncols=2, figsize=(3.35, 5), constrained_layout=True,
#                        sharex=True, sharey=True)
fig = plt.figure(figsize=(3.35, 5), dpi=300)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.0, hspace=0.0)
axs = np.empty((3, 2), dtype=object)

for aidx in range(3):
    for ajdx in range(2):
        axs[aidx,ajdx] = fig.add_subplot(gs[aidx,ajdx])
ax_flat = axs.ravel()

for ajdx, fg_comb in enumerate(fg_combs):

    #ax = fig.add_subplot(gs[aidx,ajdx])    
    ax = ax_flat[ajdx]
    post_data_flat = data_dict[fg_comb]['post_data']
    nsim_post = post_data_flat.shape[0]
    post_data_flat = post_data_flat.reshape(nsim_post, -1)
    post_data_flat = compress_utils.normalize_simple(post_data_flat, mean, std)

    data_flat = data_dict[fg_comb]['data']    
    data_flat = data_flat.reshape(-1)
    data_flat = compress_utils.normalize_simple(data_flat, mean, std)    
    
    chi_sq_post = np.einsum('ia, ab, ib -> i', post_data_flat, icov, post_data_flat)
    chi_sq_data = np.einsum('a, ab, b', data_flat, icov, data_flat)
    print(chi_sq_data)
    print(chi_sq_post)
    chi_sq_post = chi_sq_post[chi_sq_post < 1000]
    ax.hist(chi_sq_post, bins=np.linspace(200, 1000, 35),
            density=True, histtype='step', stacked=True)
    ax.axvline(chi_sq_data, color='gray', lw=1, ls='dashed')
    ax.set_xlim(200, 1000)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])

    ax.xaxis.set_ticks([300, 600, 900])

    ax.tick_params('both', direction='in', right=False, top=True, labelsize=8)

    loc = (0.31, 0.925) if 'd4' in fg_comb else (0.965, 0.925)
    ax.text(
        *loc, r'$\mathtt{' + fg_names[ajdx] + '}$',         
        transform=ax.transAxes,    # position in axes coords (0–1)
        fontsize=10, #fontweight="bold",
        ha='right', va='top', 
        #bbox=dict(facecolor="white", edgecolor="black", pad=2.5),
        multialignment="right"
    )

    
for aidx in range(2):
    for ajdx in range(2):
        axs[aidx,ajdx].xaxis.set_ticklabels([])
        #axs[aidx,ajdx].xaxis.set_ticks([])
        

fig.supxlabel(r'$\chi^2$', x=0.52, y=0.02, fontsize=12)

#gs.tight_layout(fig)
fig.savefig(opj(imgdir, 'chi_sq_post_pred_test'), bbox_inches='tight')
fig.savefig(opj(imgdir, 'chi_sq_post_pred_test.pdf'), bbox_inches='tight')
plt.close(fig)

    
# Load extra sims
# get covariance matrix
#ndata = 
# Compute chi2 for all
# 6 panel plot.
