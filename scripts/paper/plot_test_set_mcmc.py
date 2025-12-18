import os
import yaml
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import getdist
from getdist import plots as getdist_plots
from getdist import MCSamples
from mpi4py import MPI

from sbi_bmode import script_utils

comm = MPI.COMM_WORLD

opj = os.path.join

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

def plot_posterior(opath, samples, params_dict, param_truths):
    '''
    Plot corner plot of output posterior.

    Parameters
    ----------
    opath : str
        Path to output png file.
    samples :  (n_samples, n_parameters) array
        Posterior samples.
    param_truths
    '''

    param_label_dict = {'r_tensor' : r'$r$',
                        'A_lens' : r'$A_{\mathrm{lens}}$',
                        'A_d_BB' : r'$A_{\mathrm{d}}$',
                        'alpha_d_BB' : r'$\alpha_{\mathrm{d}}$',
                        'beta_dust' : r'$\beta_{\mathrm{d}}$',
                        'A_s_BB' : r'$A_{\mathrm{s}}$',
                        'alpha_s_BB' : r'$\alpha_{\mathrm{s}}$',
                        'beta_sync' : r'$\beta_{\mathrm{s}}$',
                        'rho_ds' : r'$\rho_{\mathrm{ds}}$'}
            
    prior, param_names = script_utils.get_prior(params_dict)

    param_labels = [param_label_dict[p] for p in param_names]
    param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $.

    torch.manual_seed(0)
    nsamp = 500_000
    prior_draw = draw_from_prior(prior, nsamp)
    
    param_limits = script_utils.get_param_limits(prior, param_names)

    prior_samples = MCSamples(
        samples=prior_draw, names=param_names, labels=param_labels_g,
        ranges=param_limits)
    prior_samples.smooth_scale_1D = 0.2
    
    samples = MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits)

    g = getdist_plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(samples, filled=True,
                    markers=param_truths)

    for i, name in enumerate(param_names):
        # This will add a line to the 1D marginal only
        g.add_1d(prior_samples, param=name, ls='--', color='gray', label='prior',
                 ax=g.subplots[i, i])

    print(opath)
    g.export(opath, dpi=300)
    plt.close(g.fig)

if __name__ == '__main__':

    anadir = '/u/adriaand/local/cca_project/scripts'
    basedir = '/u/adriaand/project/so/20240521_sbi_bmode'    
    idir = opj(basedir, 'mcmc65t_test2')
    imgdir = opj(idir, 'img_corner')
    #config_path = opj(basedir, 'run65', 'config.yaml')
    config_path = opj(anadir, 'configs', 'config28.yaml')    
    samples_path = opj(basedir, 'mcmc65t_test2', 'samples.npy')
    test_params_path = opj(basedir, 'run65t', 'param_draws_test.npy')
    seed = 0
    
    if comm.rank == 0:

        os.makedirs(imgdir, exist_ok=True)        
        with open(config_path, 'r') as yfile:
            config = yaml.safe_load(yfile)
            
        params = np.load(test_params_path)        
        samples = np.load(samples_path)
    else:
        config = None
        params = None
        samples = None
        
    config = comm.bcast(config, root=0)
    params = comm.bcast(params, root=0)    
    samples = comm.bcast(samples, root=0)            

    _, _, params_dict = script_utils.parse_config(config)

    params_dict.pop('amp_beta_dust')
    params_dict.pop('gamma_beta_dust')
    params_dict.pop('amp_beta_sync')
    params_dict.pop('gamma_beta_sync')    
    
    if samples.ndim == 4:
        samples = samples.reshape(
            samples.shape[0], samples.shape[1] * samples.shape[2], -1)

    if samples.shape[-1] == 13:
        samples = samples[:,[0, 1, 2, 3, 4, 7, 8, 9, 12]]
    print(samples.shape)
    print(params.shape)
    
    # All rank get the same seed, should be fine in this case.
    rng = np.random.default_rng(seed)
    
    nparam = params.shape[-1]    
    nsim = samples.shape[0]
    
    idxs = np.arange(nsim)
    idxs_per_rank = np.array_split(idxs, comm.size)
        
    idxs_on_rank = idxs_per_rank[comm.rank]
    
    for ridx in idxs_on_rank:
    
        plot_posterior(opj(imgdir, f'corner_{ridx:03d}.png'), samples[ridx],
                       params_dict, params[ridx])
