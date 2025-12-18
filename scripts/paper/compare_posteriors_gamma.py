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

#param_names = ['r_tensor', 'A_lens', 'A_d_BB', 'alpha_d_BB', 'beta_dust', 'amp_beta_dust', 'gamma_beta_dust']
#param_labels = [r'$r$', r'$A_L$', r'$A_d^{BB}$', r'$\alpha_d^{BB}$', r'$\beta_d$', r'$A_{\beta}$', r'$\gamma_{\beta}$']

#oname = '32c_33a_33b_33c'
#dirs2compare = ['run32c', 'run33a', 'run33b', 'run33c']
#labels = ['cmb + dust + dbeta + deproj dust', 'cmb', 'cmb + deproj. dust', 'cmb + deproj. (dbeta + dust)']

#oname = '36_36b_36c'
#dirs2compare = ['run36', 'run36b', 'run36c']
#labels = ['old', 'new', 'new, 3x sims']
#oname = 'mcmc54_52bt71'
#dirs2compare = ['mcmc54', 'run52_optuna/trial_0071']
#labels = ['multifreq mcmc', 'NILC']

#oname = '52bt40_63'
#dirs2compare = ['run52b_optuna/trial_0040', 'run63']
#labels = ['NPE', 'SNPE']
#oname = '52bt40_61t63_63_64'
#oname = 'prior_only'
#oname = '65t120_74'
#dirs2compare = ['run65_optuna/trial_0120', 'run74']
#labels = ['NPE', 'SNPE']

oname = '86_86beta_86beta4'
dirs2compare = ['run86_optuna/trial_0020', 'run86_beta4_optuna/trial_0101', 'run86_beta_extra_optuna/trial_0071']
labels = ['beta_pyilc=-1', 'beta_pyilc=-2', 'beta_pyilc=-3']

#with open(opj(basedir, dirs2compare[0], 'config.yaml'), 'r') as yfile:
#    config = yaml.safe_load(yfile)
#with open(opj(basedir, 'run52', 'config.yaml'), 'r') as yfile:
#    config = yaml.safe_load(yfile)
with open(opj(basedir, 'run86', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)
#with open(opj('/u/adriaand/local/cca_project/scripts/configs', 'config30.yaml'), 'r') as yfile:
#    config = yaml.safe_load(yfile)
    
#configdir = '/mnt/home/aduivenvoorden/local/cca_project/scripts/configs'
#with open(opj(configdir, 'config13.yaml'), 'r') as yfile:
#    config = yaml.safe_load(yfile)


data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

#param_truths = [0.005, 1, 5, -0.2, 1.59]
#param_truths = [0.01, 1.05, 4, -0.3, 1.55, 0.4, -3.5]

#param_limits = {'r_tensor' : (0., None), 'A_lens' : (0., None), 'A_d_BB' : (0., None),
#                'amp_beta_dust' : (0., 1.59), 'gamma_beta_dust' : (-6., -1.5)}
param_limits = script_utils.get_param_limits(prior, param_names)

# Sample from prior.

print(f'{param_names=}')
print(f'{param_labels=}')
print(f'{param_truths=}')
print(f'{param_limits=}')

torch.manual_seed(0)
nsamp = 50_000
prior_draw = draw_from_prior(prior, nsamp)

colors = ['C%d'%i for i in range(len(dirs2compare))]
#colors = ['black'] + colors

samples_g = []
#for sidx, subdir in enumerate(dirs2compare):

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits,
    label=f'prior')
#samples_g.append(prior_samples) # NOTE

for sidx, (subdir, label) in enumerate(zip(dirs2compare, labels)):

    idir = opj(basedir, subdir)
    if sidx == len(dirs2compare) - 1:
        imgdir = opj(idir, 'img')
        os.makedirs(imgdir, exist_ok=True)
    samples = np.load(opj(idir, 'samples.npy'))
    #samples = np.load(opj(idir, 'samples_round_000.npy'))

    print(samples.shape)
    
    samples = getdist_MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=f'{subdir}: {label}')
    
    samples_g.append(samples)
        
g = getdist_plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(samples_g, filled=True,
                markers=param_truths,
                contour_colors=colors
                )

for i, name in enumerate(param_names):
    # This will add a line to the 1D marginal only
    g.add_1d(prior_samples, param=name, ls='--', color='gray', label='prior', ax=g.subplots[i, i])

g.export(opj(imgdir, f'{oname}_corner.png'), dpi=300)
