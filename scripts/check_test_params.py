import os
import yaml
import numpy as np
from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples
import torch

from sbi_bmode import script_utils

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'

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
    
param_label_dict = {'r_tensor' : r'$r$',
                    'A_lens' : r'$A_{\mathrm{lens}}$',
                    'A_d_BB' : r'$A_{\mathrm{d}}$',
                    'alpha_d_BB' : r'$\alpha_{\mathrm{d}}$',
                    'beta_dust' : r'$\beta_{\mathrm{d}}$',
                    'amp_beta_dust' : r'$B_{\mathrm{d}}$',
                    'gamma_beta_dust' : r'$\gamma_{\mathrm{d}}$',
                    'A_s_BB' : r'$A_{\mathrm{s}}$',
                    'alpha_s_BB' : r'$\alpha_{\mathrm{s}}$',
                    'beta_sync' : r'$\beta_{\mathrm{s}}$',
                    'amp_beta_sync' : r'$B_{\mathrm{s}}$',
                    'gamma_beta_sync' : r'$\gamma_{\mathrm{s}}$',
                    'rho_ds' : r'$\rho_{\mathrm{ds}}$'}

oname = '49b_test'
dirs2compare = ['run49b']
labels = ['test set HILC']

with open(opj(basedir, 'run49', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)
    
data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

param_limits = script_utils.get_param_limits(prior, param_names)

# Sample from prior.
torch.manual_seed(0)
nsamp = 50_000
prior_draw = draw_from_prior(prior, nsamp)

colors = ['C%d'%i for i in range(len(dirs2compare))]

samples_g = []

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits)

for sidx, (subdir, label) in enumerate(zip(dirs2compare, labels)):

    idir = opj(basedir, subdir)
    if sidx == len(dirs2compare) - 1:
        imgdir = opj(idir, 'img')
        os.makedirs(imgdir, exist_ok=True)
    #samples = np.load(opj(idir, 'samples.npy'))
    samples = np.load(opj(idir, 'param_draws_test.npy'))


    print(samples.shape)
    
    samples = getdist_MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=f'{subdir}: {label}')
    
    samples_g.append(samples)
    

    
g = getdist_plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(samples_g, filled=True,
                markers=param_truths,
                contour_colors=colors,
                )

for i, name in enumerate(param_names):
    # This will add a line to the 1D marginal only
    g.add_1d(prior_samples, param=name, ls='--', color='gray', label='prior', ax=g.subplots[i, i])

g.export(opj(imgdir, f'{oname}_corner.png'), dpi=300)
