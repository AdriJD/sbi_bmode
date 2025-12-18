import os
import pickle
import yaml
import signal
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import sbi
import getdist
from getdist import plots as getdist_plots
from getdist import MCSamples

from sbi_bmode import script_utils

opj = os.path.join

#idir = '/u/adriaand/project/so/20240521_sbi_bmode/run49b'
#idir_post = '/u/adriaand/project/so/20240521_sbi_bmode/run50_optuna/trial_0141'

#idir = '/u/adriaand/project/so/20240521_sbi_bmode/run45d'
idir = '/u/adriaand/project/so/20240521_sbi_bmode/run56'
#idir_post = '/u/adriaand/project/so/20240521_sbi_bmode/run48_optuna_b/trial_0060'
idir_post = '/u/adriaand/project/so/20240521_sbi_bmode/run56_optuna/trial_0104'
#idir_post = '/u/adriaand/project/so/20240521_sbi_bmode/run51'

imgdir = opj(idir_post, 'img')
imgdir_corner = opj(imgdir, 'corner_plots')
os.makedirs(imgdir, exist_ok=True)
os.makedirs(imgdir_corner, exist_ok=True)

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

def plot_posterior(opath, samples, config, param_label_dict, param_truths):
    '''
    Plot corner plot of output posterior.

    Parameters
    ----------
    opath : str
        Path to output png file.
    samples :  (n_samples, n_parameters) array
        Posterior samples.
    prior_samples :  (n_prior_samples, n_parameters) array
        Prior samples.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    '''

    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior, param_names = script_utils.get_prior(params_dict)
    param_labels = [param_label_dict[p] for p in param_names]
    #param_truths = script_utils.get_true_params(params_dict)
    param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $.

    torch.manual_seed(0)
    nsamp = 10_000
    prior_draw = draw_from_prior(prior, nsamp)
    
    param_limits = script_utils.get_param_limits(prior, param_names)

    prior_samples = MCSamples(
        samples=prior_draw, names=param_names, labels=param_labels_g,
        ranges=param_limits)

    print(prior_draw.shape)
    print(samples.shape)
    samples = MCSamples(
        samples=samples.T, names=param_names, labels=param_labels_g,
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

quantiles = np.arange(0.05, 1.05, 0.05)            # which quantiles to plot
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


param_names = [param_label_dict[key] for key in param_label_dict.keys()]

with open(opj(idir_post, 'posterior.pkl'), 'rb') as f:
    posterior = pickle.load(f)

# load prior draws and generated data
param_draws = np.load(opj(idir, 'param_draws_test.npy'))
data_draws = np.load(opj(idir, 'data_draws_test.npy'))

config_file = opj(idir, 'config.yaml')
with open(config_file, 'r') as yfile:
    config = yaml.safe_load(yfile)

# NOTE
#param_draws = param_draws[1:]
#data_draws = data_draws[1:]

print(np.all(np.isfinite(param_draws)))
print(np.all(np.isfinite(data_draws)))
print(posterior.prior)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def _handle_timeout(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and an alarm.
    original_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Cancel the alarm and restore original handler.
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def compute_coverage_mean_std(posterior, param_draws, data_draws, quantiles=[.95]):
    """
    Generate a coverage plot based on posterior samples from SBI, where true parameters are part of the dataloader.

    Parameters:
    - posterior: SBI posterior object
    - param_draws: [num_draws, nparams]
    - data_draws: [num_draws, ndimdata]  simulated data from the simulator
    - quantiles: Desired coverage levels (default [.95]).

    Returns:
    - coverage (shape = nquants, nparams)
    - means    (shape = num_draws, nparams)
    - stds     (shape = num_draws, nparams)
    """
    
    num_draws = param_draws.shape[0]  # number of prior draws
    num_samples = 1000                # number of posterior samples to draw

    # To track the coverage for each parameter dimension
    total_coverage = []
    means = []
    stds = []

    for i in range(num_draws):
        print(i)
        # Draw posterior samples (num_samples could be adjusted)

        print(f'idx = {i}: {param_draws[i]}')        
        #try:
        #    with timeout(5):        
        posterior_samples = posterior.sample((num_samples,), x=data_draws[i], show_progress_bars=True).numpy().T
        true_params = param_draws[i]
        #except TimeoutException:
        #    print(f'skipping {i}')
        #    continue
        #plot_posterior(opj(imgdir_corner, f'corner_{i}.png'), posterior_samples, config, param_label_dict,
        #               param_draws[i])
        
        # Calculate coverage for each parameter in the true_params
        coverage_per_param_quantile = np.zeros((len(quantiles), true_params.shape[0]))
        for qi,quantile in enumerate(quantiles):
            coverage_per_param = np.zeros(len(true_params))
            for pi in range(len(true_params)):  # Iterate over the parameter dimensions
                # Calculate the quantile bounds for the posterior samples of the i-th parameter
                lower_bound = np.quantile(posterior_samples[pi], (1 - quantile) / 2)
                upper_bound = np.quantile(posterior_samples[pi], 1 - (1 - quantile) / 2)

                # Check if the true parameter is within the bounds
                coverage_per_param[pi] = ((true_params[pi] >= lower_bound) &
                                        (true_params[pi] <= upper_bound))

            coverage_per_param_quantile[qi] = coverage_per_param

        total_coverage.append(coverage_per_param_quantile)
        means.append(np.mean(posterior_samples, axis=-1))
        stds.append(np.std(posterior_samples, axis=-1))

    coverage = np.mean(total_coverage, axis=0)   # taking the mean over all sims. This will have shape = num_params
    return coverage, np.array(means), np.array(stds)



# compute coverage, means, stds
coverages, means, stds = compute_coverage_mean_std(posterior, param_draws, data_draws, quantiles=quantiles)

# plot all coverage information. may want to change bin_edges_params for each panel below
ncols = param_draws.shape[1]
fig, ax = plt.subplots(4, ncols, figsize=(2*ncols,7), squeeze=False, dpi=300, constrained_layout=True)
true_params = param_draws.T

# coverage plot
for pi in range(ncols):
    ax[0,pi].set_title(param_names[pi])
    ax[0,pi].plot(quantiles, coverages[:,pi])
    ax[0,pi].plot(quantiles, quantiles, color='k', linestyle=':')
    ax[0,pi].set_xlabel('Nominal Coverage')

# residual plot
bin_edges_params = [np.linspace(-2, 2, 20)]*ncols
for pi in range(ncols):
    bin_edges = bin_edges_params[pi]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    res = means[:,pi] - true_params[pi]
    MSE = np.mean(res**2)
    h, _ = np.histogram(res, bins=bin_edges, density=1)
    ax[1,pi].plot(bin_centers, h, label='MSE=%.1e'%MSE)
    ax[1,pi].set_xlabel('residual')
    ax[1,pi].legend(fontsize=8)
    ax[1,pi].vlines(0,h.min(),h.max(), color='gray', linestyle=':')

# var plot
bin_edges_params = [np.linspace(0, 1, 40)]*ncols
for pi in range(ncols):
    bin_edges = bin_edges_params[pi]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    sig = stds[:,pi]
    avgsig2 = np.mean(sig**2)
    h, _ = np.histogram(sig, bins=bin_edges, density=1)
    ax[2,pi].plot(bin_centers, h, label=r'$\bar{\sigma}^2$=%.1e'%avgsig2)
    ax[2,pi].set_xlabel(r'$\sigma$')
    ax[2,pi].legend(fontsize=8)
    ax[2,pi].vlines(0,h.min(),h.max(), color='gray', linestyle=':')

# chi plot
bin_edges_params = [np.linspace(-5, 5, 20)]*ncols
for pi in range(ncols):
    bin_edges = bin_edges_params[pi]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    res = means[:,pi] - true_params[pi]
    sig = stds[:,pi]
    chi = res/sig
    avgchisq = np.mean(chi**2)
    h, _ = np.histogram(chi, bins=bin_edges, density=1)
    ax[3,pi].plot(bin_centers, h, label=r'$\bar{\chi}^2$=%.1f'%avgchisq)
    _z = np.linspace(-5,5,1000)
    ax[3,pi].plot(_z, norm.pdf(_z), label='normal', color='k', linestyle='--')
    ax[3,pi].set_xlabel(r'$\chi$=residual/$\sigma$')
    ax[3,pi].legend(fontsize=8)
    ax[3,pi].vlines(0,h.min(),h.max(), color='gray', linestyle=':')


ax[0,0].set_ylabel('Empirical Coverage')
ax[1,0].set_ylabel('prob')
ax[2,0].set_ylabel('prob')
ax[3,0].set_ylabel('prob')

fig.savefig(opj(imgdir, 'coverage.png'))
plt.close(fig)
