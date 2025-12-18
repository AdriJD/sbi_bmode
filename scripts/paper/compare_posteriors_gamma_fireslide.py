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
                    #'A_lens' : r'$A_{\mathrm{lens}}$',
                    #'A_d_BB' : r'$A_{\mathrm{d}}$',
                    #'alpha_d_BB' : r'$\alpha_{\mathrm{d}}$',
                    'beta_dust' : r'$\beta_{\mathrm{d}}$',
                    #'amp_beta_dust' : r'$B_{\mathrm{d}}$',
                    #'gamma_beta_dust' : r'$\gamma_{\mathrm{d}}$',
                    #'A_s_BB' : r'$A_{\mathrm{s}}$',
                    #'alpha_s_BB' : r'$\alpha_{\mathrm{s}}$',
                    #'beta_sync' : r'$\beta_{\mathrm{s}}$',
                    #'amp_beta_sync' : r'$B_{\mathrm{s}}$',
                    #'gamma_beta_sync' : r'$\gamma_{\mathrm{s}}$',
                    #'rho_ds' : r'$\rho_{\mathrm{ds}}$'}
                    }

oname = 'comparison_with_ani_beta'
#dirs2compare = ['mcmc54', 'run56_optuna/trial_0104', 'run59_optuna/trial_0143', 'run63']
#labels = ['MF', 'NILC CMB', 'cNILC CMB', 'NILC']
#dirs2compare = ['run59_optuna/trial_0143', 'mcmc54',  'run63']
#labels = ['cNILC CMB', 'MF', 'NILC']
dirs2compare = ['run56_optuna/trial_0104', 'run59_optuna/trial_0143', 'mcmc54',  'run63']
labels = ['NILC CMB', 'cNILC CMB', 'MF', 'NILC']

with open(opj(basedir, 'run52', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)
    
data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)

params_dict.pop('A_lens')
params_dict.pop('A_d_BB')
params_dict.pop('alpha_d_BB')
#params_dict.pop('beta_dust')
params_dict.pop('A_s_BB')
params_dict.pop('alpha_s_BB')
params_dict.pop('beta_sync')
params_dict.pop('rho_ds')

params_dict.pop('amp_beta_dust')
params_dict.pop('gamma_beta_dust')
params_dict.pop('amp_beta_sync')
params_dict.pop('gamma_beta_sync')

prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

param_limits = script_utils.get_param_limits(prior, param_names)

# Sample from prior.
print(param_names)
print(param_labels)
print(param_truths)
print(param_limits)

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
    samples = np.load(opj(idir, 'samples.npy'))

    print(samples.shape)

    #if sidx > 0:
    #    samples = samples[:,[0, 1, 2, 3, 4, 7, 8, 9, 12]]
    # NOTE
    samples = samples[:,[0, 4]]

    print(np.std(samples))
    
    samples = getdist_MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        #label=f'{subdir}: {label}')
        label=f'{label}')    
    
    samples_g.append(samples)
    
#g = getdist_plots.get_subplot_plotter(width_inch=8)
g = getdist_plots.get_subplot_plotter(width_inch=3)

g.triangle_plot(samples_g, filled=False,
                markers=param_truths,
                contour_colors=colors,
                param_limits={'r_tensor': [0., 0.025], 'beta_dust': [1., 2.2]}
                )

# NOTE
#for i, name in enumerate(param_names):
#    # This will add a line to the 1D marginal only
#    g.add_1d(prior_samples, param=name, ls='--', color='gray', label='prior', ax=g.subplots[i, i])

#g.export(opj(imgdir, f'{oname}_corner.png'), dpi=300)
g.export(opj(imgdir, f'{oname}_corner.png'), dpi=450)
