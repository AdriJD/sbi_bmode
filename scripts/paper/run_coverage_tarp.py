import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#import tarp FIXME todo
from scipy.stats import norm

import torch
import sbi
from sbi.diagnostics import check_sbc, check_tarp, run_sbc, run_tarp

opj = os.path.join

idir = '/u/adriaand/project/so/20240521_sbi_bmode/run56'
idir_post = '/u/adriaand/project/so/20240521_sbi_bmode/run56_optuna/trial_0104'

imgdir = opj(idir_post, 'img')
os.makedirs(imgdir, exist_ok=True)

quantiles = np.arange(0.05, 1.05, 0.05)            # which quantiles to plot
param_label_dict = {'r_tensor' : r'$r$',
                    'A_lens' : r'$A_L$',
                    'A_d_BB' : r'$A_d^{BB}$',
                    'alpha_d_BB' : r'$\alpha_d^{BB}$',
                    'beta_dust' : r'$\beta_d$',
                    'amp_beta_dust' : r'$A_{\beta}$',
                    'gamma_beta_dust' : r'$\gamma_{\beta}$'}
param_names = [param_label_dict[key] for key in param_label_dict.keys()]

with open(opj(idir_post, 'posterior.pkl'), 'rb') as f:
    posterior = pickle.load(f)

# load prior draws and generated data
param_draws = np.load(opj(idir, 'param_draws_test.npy'))
data_draws = np.load(opj(idir, 'data_draws_test.npy'))

num_tarp_samples = param_draws.shape[0]
print(num_tarp_samples)
nthreads = len(os.sched_getaffinity(0))
print(nthreads)
ecp, alpha = run_tarp(
    torch.as_tensor(param_draws),
    torch.as_tensor(data_draws),
    posterior,
    num_workers=nthreads,
    references=None,  # will be calculated automatically.
    num_posterior_samples=1000,
)

atc, ks_pval = check_tarp(ecp, alpha)

print(atc, "Should be close to 0")
print(ks_pval, "Should be larger than 0.05")


fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
ax.plot(alpha, ecp, color="blue", label="TARP")
ax.plot(alpha, alpha, color="black", linestyle="--", label="ideal")
ax.set_xlabel(r"Credibility Level $\alpha$")
ax.set_ylabel(r"Expected Coverage Probability")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'coverage_tarp'))
plt.close(fig)
