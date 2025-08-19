import os
import pickle
import yaml
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import sbi
from sbi.utils.sbiutils import seed_all_backends
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

def plot_posterior(opath, samples, config, param_truths, cosmo_only=False):
    '''
    Plot corner plot of output posterior.

    Parameters
    ----------
    opath : str
        Path to output png file.
    samples :  (n_samples, n_parameters) array
        Posterior samples.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    param_truths
    '''

    if cosmo_only:
        param_label_dict = {'r_tensor' : r'$r$',
                            'A_lens' : r'$A_{\mathrm{lens}}$'}
    else:    
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
        
    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior, param_names = script_utils.get_prior(params_dict)

    if cosmo_only:
        prior = prior[0:2]
        param_names = param_names[0:2]
    
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

def sample(posterior, data_obs, nsamp=10000):
    '''

    Parameters
    ----------

    Returns
    -------
    samples : (nsamp, nparam)
        Posterior draws for each input dataset.
    '''

    samples = posterior.sample(
        (nsamp,), x=data_obs, show_progress_bars=False)
    samples = np.asarray(samples, dtype=np.float64)
    
    return samples

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--posterior', help='Path to posterior .pkl file')
    parser.add_argument('--test-params', help='Path to .npy file containing test set parameters')
    parser.add_argument('--test-data', help='Path to .npy file containing test set data draws')
    parser.add_argument('--nsamp', type=int, default=10000, help='number of posterior samples to draw')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cosmo-only', action='store_true', help='Use posterior that only predicts r and A_lens.')        

    args = parser.parse_args()

    odir = args.odir
    imgdir = opj(args.odir, 'img')
    if comm.rank == 0:

        print(f'Running with arguments: {args}')
        print(f'Running with {comm.size} MPI rank(s)')

        os.makedirs(odir, exist_ok=True)
        os.makedirs(imgdir, exist_ok=True)        
        with open(args.config, 'r') as yfile:
            config = yaml.safe_load(yfile)
            
        # load files
        params = np.load(args.test_params)
        data = np.load(args.test_data)
        
        with open(opj(args.posterior), 'rb') as f:
            posterior = pickle.load(f)        
    else:
        config = None
        params = None
        data = None
        posterior = None
        
    config = comm.bcast(config, root=0)
    params = comm.bcast(params, root=0)    
    if args.cosmo_only:
        params = params[:,:2]
    data = comm.bcast(data, root=0)            
    posterior = comm.bcast(posterior, root=0)

    # All rank get the same seed, should be fine in this case.
    rng = np.random.default_rng(args.seed)    
    seed_all_backends(int(rng.integers(2 ** 32 - 1)))
    
    nparam = params.shape[-1]    
    nsim = data.shape[0]
    
    idxs = np.arange(nsim)
    idxs_per_rank = np.array_split(idxs, comm.size)
    num_sims_per_rank = np.zeros(comm.size, dtype=int)
    for ridx in range(comm.size):
        num_sims_per_rank[ridx] = len(idxs_per_rank[ridx])
        
    idxs_on_rank = idxs_per_rank[comm.rank]
    samples_on_rank = np.zeros((len(idxs_on_rank), args.nsamp, nparam))
    
    for idx, ridx in enumerate(idxs_on_rank):
    
        samples_on_rank[idx] = sample(posterior, data[ridx], nsamp=args.nsamp)
    
        plot_posterior(opj(imgdir, f'corner_{ridx:03d}.png'), samples_on_rank[idx],
                       config, params[ridx], cosmo_only=args.cosmo_only)
                   
    # Save all samples in one array.
    if comm.rank == 0:
        samples_full = np.zeros(nsim * args.nsamp * nparam)
    else:
        samples_full = None

    offsets = np.zeros(comm.size, dtype=int)
    offsets[1:] = np.cumsum(num_sims_per_rank * args.nsamp * nparam)[:-1]
    comm.Gatherv(
        sendbuf=samples_on_rank,
        recvbuf=(samples_full, np.array(num_sims_per_rank * args.nsamp * nparam, dtype=int),
                 np.array(offsets, dtype=int), MPI.DOUBLE), root=0)
    
    # Reorder.
    if comm.rank == 0:
        samples_full_reordered = np.zeros((nsim, args.nsamp, nparam))
        for ridx in range(comm.size):
            
            offset = offsets[ridx]
            tot_size_on_rank = num_sims_per_rank[ridx] * args.nsamp * nparam
            
            samples_full_reordered[idxs_per_rank[ridx],:,:] = \
                samples_full[offset:offset+tot_size_on_rank].reshape(
                    int(num_sims_per_rank[ridx]), args.nsamp, nparam)
        
        np.save(opj(args.odir, 'samples_test'), samples_full_reordered)
        
