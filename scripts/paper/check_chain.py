import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples

from sbi_bmode import script_utils

opj = os.path.join

anadir = '/u/adriaand/local/cca_project/scripts'
basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
#idir = opj(basedir, 'mcmc66_rerun')
idir = opj(basedir, 'mcmc65t_test_debug')
#idir = opj(basedir, 'mcmc65t_test2')
imgdir = opj(idir, 'img')
os.makedirs(imgdir, exist_ok=True)

chains = np.load(opj(idir, 'samples.npy'))

# NOTE
#chains = chains[3,0:1]
chains = chains[0]
print(chains.shape)

oname = '65'
#with open(opj(basedir, 'run65', 'config.yaml'), 'r') as yfile:
#    config = yaml.safe_load(yfile)
with open(opj(anadir, 'configs', 'config28.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)

if chains.ndim == 2:
    chains = chains[np.newaxis,:,:]
num_chains = chains.shape[0]


param_names = ['r_tensor', 'A_lens', 'A_d_BB', 'alpha_d_BB', 'beta_dust', 'A_s_BB',
               'alpha_s_BB', 'beta_sync', 'rho_ds']
param_labels = [r'$r$', r'$A_L$', r'$A_d^{BB}$', r'$\alpha_d^{BB}$', r'$\beta_d$',
                r'$A_s^{BB}$', r'$\alpha_s^{BB}$', r'$\beta_s$', r'$\rho_{ds}$']

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $
#param_truths = [0.01, 0.45, 4., -0.3, 1.55, 1.5, -0.6, -2.8, 0.1]


_, _, params_dict = script_utils.parse_config(config)
#prior, param_names = script_utils.get_prior(params_dict)
#param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)
param_truths.pop('amp_beta_dust', None)
param_truths.pop('gamma_beta_dust', None)
param_truths.pop('amp_beta_sync', None)
param_truths.pop('gamma_beta_sync', None)

print(param_truths)

samples_g = []
for cidx in range(num_chains):

    samples = chains[cidx]
    print(np.mean(samples[:,0]))
    print(np.std(samples[:,0]))    

    samples = getdist_MCSamples(
        samples=samples, names=param_labels, labels=param_labels_g,
        ranges={'$r$': [0, None]})
    samples_g.append(samples)

g = getdist_plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(samples_g, filled=True,
                markers=param_truths,
                contour_colors=['C%d'%i for i in range(num_chains)],
                )
g.export(opj(imgdir, f'{oname}_corner_per_chain.png'), dpi=300)

if num_chains > 2:
    # Compute Gelman-Rubin statistic.
    ll = chains.shape[1]
    chain_means = np.mean(chains, axis=1)
    chain_means_variance = np.var(chain_means, axis=0, ddof=1)
    bb = chain_means_variance
    ww = np.mean(np.var(chains, axis=1, ddof=1), axis=0)

    rr  = ((ll - 1) * ww / ll + bb) / ww

# trace plot for each parameter
num_params = chains.shape[-1]
fig, axs = plt.subplots(dpi=300, nrows=num_params, constrained_layout=True, figsize=(3, num_params),
                        sharex=True)
for pidx in range(num_params):
    axs[pidx].plot(chains[:,:,pidx].reshape(-1), lw=0.5)
    axs[pidx].set_ylabel(param_labels[pidx])
    if num_chains > 2:
        axs[pidx].text(0.6, 0.05, f'R-1={1-rr[pidx]:.3f}', transform=axs[pidx].transAxes)
axs[-1].set_xlabel('sample number')
fig.savefig(opj(imgdir, 'trace'))
plt.close(fig)

