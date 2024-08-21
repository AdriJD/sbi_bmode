import os
import yaml
import pickle
import argparse

from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np

from pixell.utils import eigpow
from sbi_bmode import (spectra_utils, sim_utils, so_utils, likelihood_utils,
                       script_utils)

comm = MPI.COMM_WORLD
opj = os.path.join

def get_prior(params_dict):
    '''
    Parse parameter dictionary and return log prior function.

    Parameters
    ----------
    params_dict : dict
        Dictionary with parameters that we sample.    

    Returns
    -------
    logprior : callable
        Callable that return the logprior.
    param_names : list of str
        List of parameter names in same order as prior.
    bounds : list of tuples
        List with (min, max) per param.
    '''
    
    prior = []
    param_names = []
    bounds = []
    for param, prior_dict in params_dict.items():
    
        if prior_dict['prior_type'].lower() == 'normal':
            prior.append(likelihood_utils.Normal(
                *prior_dict['prior_params']))
            bounds.append((None, None))
            
        elif prior_dict['prior_type'].lower() == 'halfnormal':
            prior.append(likelihood_utils.Normal(
                0., *prior_dict['prior_params'], halfnormal=True))
            bounds.append((0., None))
            
        else:
            raise ValueError(f"{prior_dict['prior_type']=} not understood")
        
        param_names.append(param)
        
    prior_combined = likelihood_utils.MultipleIndependent(prior)
        
    return prior_combined, param_names, bounds

def main(odir, config, specdir, seed):

    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior_combined, param_names, bounds = get_prior(params_dict)

    cmb_simulator = sim_utils.CMBSimulator(specdir, data_dict, fixed_params_dict)

    #print(cmb_simulator.bins)
    #exit()
    
    # Get prior mean.
    mean = prior_combined.get_mean()
    
    # Get covariance matrix
    mean_dict = {}
    for idx, name in enumerate(param_names):
        mean_dict[name] = mean[idx]
    
    signal_spectra = cmb_simulator.get_signal_spectra(
        mean_dict['r_tensor'], mean_dict['A_lens'], mean_dict['A_d_BB'],
        mean_dict['alpha_d_BB'], mean_dict['beta_dust'])

    noise_spectra = cmb_simulator.get_noise_spectra()
    cov = likelihood_utils.get_cov(
        np.asarray(signal_spectra), noise_spectra, cmb_simulator.bins, cmb_simulator.lmin,
        cmb_simulator.lmax, cmb_simulator.nsplit, cmb_simulator.nfreq)    

    # Draw signal spectra.
    num_sims = 12800
    #num_sims = 12_800
    #print(f'{cmb_simulator.size_data=}')
    #sims = np.zeros((num_sims, cmb_simulator.size_data))

    sim_idxs = np.arange(num_sims)
    sim_idxs_per_rank = np.array_split(sim_idxs, comm.size)
    #print(f'{sim_idxs_per_rank=}')
    nsim_per_rank = sim_idxs_per_rank[comm.rank].size
    sims_per_rank = np.zeros((nsim_per_rank, cmb_simulator.size_data))
    
    #rng = np.random.default_rng(seed=seed)
    rng = np.random.default_rng(seed=comm.rank)    

    
    
    for idx in range(nsim_per_rank):        
        #print(comm.rank, idx)
        sims_per_rank[idx] = cmb_simulator.draw_data(
            r_tensor=mean_dict['r_tensor'], A_lens=mean_dict['A_lens'],
            A_d_BB=mean_dict['A_d_BB'], alpha_d_BB=mean_dict['alpha_d_BB'],
            beta_dust=mean_dict['beta_dust'], seed=rng)

    #print(comm.rank, sims)
    sendcounts = np.array(comm.gather(nsim_per_rank * cmb_simulator.size_data, 0))

    #print(comm.rank, f'{sendcounts=}')
    
    recvbuf = None
    if comm.rank == 0:
        #recvbuf = np.zeros_like(sims)
        recvbuf = np.empty(sum(sendcounts), dtype=sims_per_rank.dtype)

        
    #comm.Gather(sims, recvbuf, root=0)

    comm.Gatherv(sendbuf=sims_per_rank, recvbuf=(recvbuf, sendcounts), root=0)
    
    if comm.rank == 0:
               
        sims = recvbuf.reshape(-1,  cmb_simulator.size_data)
        
        cov_mc = np.cov(sims.T)

        tri_indices = sim_utils.get_tri_indices(cmb_simulator.nsplit, cmb_simulator.nfreq)
        cov_ext = np.ones((tri_indices.shape[0], tri_indices.shape[0], cov.shape[-1], cov.shape[-1]))
        cov_ext[:] = cov[:,:,:,np.newaxis]
        cov_ext[:] *= np.eye(cov.shape[-1])[np.newaxis,np.newaxis,:,:]

        cov_ext = np.transpose(cov_ext, (0, 2, 1, 3))
        cov_ext = cov_ext.reshape(cov_mc.shape)


        print(f'{np.linalg.cond(cov_ext)=}')

        # Also check if the mean is correct.
        model = cmb_simulator.get_signal_spectra(
            mean_dict['r_tensor'], mean_dict['A_lens'], mean_dict['A_d_BB'], mean_dict['alpha_d_BB'],
            mean_dict['beta_dust'])
        sims_mean = np.mean(sims, axis=0)

        print(f'{model.shape=}')
        print(f'{sims_mean.shape=}')        

        #data.reshape(tri_indices.shape[0], -1)
        
        diff = likelihood_utils.get_diff(
            sims_mean.reshape(tri_indices.shape[0], -1), model, tri_indices)

        nsamp = 50
        diff_samples = np.zeros((nsamp, sims_mean.size))
        for idx in range(nsamp):
            diff_samples[idx] = likelihood_utils.get_diff(
            sims[idx].reshape(tri_indices.shape[0], -1), model, tri_indices).reshape(-1)
        
        fig, ax = plt.subplots(dpi=300)
        for idx in range(nsamp):
            ax.plot(diff_samples[idx], color='black', alpha=0.3, lw=0.5)
        ax.plot(diff.reshape(-1), color='red', lw=0.5)
        fig.savefig(opj(odir, 'diff_spec'))
        plt.close(fig)
        
        
        #cov_prod = np.dot(np.linalg.inv(cov_ext), cov_mc)
        isqrt_cov_ext = eigpow(cov_ext, -0.5)
        cov_prod = isqrt_cov_ext @ cov_mc @ isqrt_cov_ext
        
        sim_mean = np.mean(sims, axis=0)

        fig, axs = plt.subplots(nrows=2, dpi=300, sharex=True)
        for idx in range(10):
            axs[0].plot(sims[idx], lw=0.5)        
            axs[1].plot(sims[idx] - sim_mean, lw=0.5)
        fig.savefig(opj(odir, 'data_draws'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        im = ax.imshow(np.log10(np.abs(cov_mc)), vmin=-14, vmax=-7.5)
        #im = ax.imshow(cov_mc)
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cov_mc'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        #im = ax.imshow(cov_prod - np.eye(cov_prod.shape[0]) / np.sqrt(num_sims))
        im = ax.imshow((cov_prod - np.eye(cov_prod.shape[0])) * np.sqrt(num_sims))        
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cov_prod'))
        plt.close(fig)
        
        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        im = ax.imshow(np.log10(np.absolute(cov_ext, out=np.ones_like(cov_ext) * 1e-20, where=cov_ext != 0)), vmin=-14, vmax=-7.5)
        #im = ax.imshow(cov_ext)
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cov'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        #im = ax.imshow(cov_ext - cov_mc)
        im = ax.imshow(np.log10(np.abs(cov_ext - cov_mc)), vmin=-14, vmax=-7.5)
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cov_diff'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        im = ax.imshow(np.divide(cov_mc, cov_ext, where=cov_ext != 0, out=np.ones_like(cov_ext)),
                       vmin=0.9, vmax=1.1)
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cov_ratio'))
        plt.close(fig)

        fig, axs = plt.subplots(nrows=2, dpi=300)
        axs[0].plot(np.diag(cov_mc))
        axs[0].plot(np.diag(cov_ext))
        axs[1].plot(np.diag(cov_ext) / np.diag(cov_mc))
        axs[1].set_ylim(0.9, 1.1)
        fig.savefig(opj(odir, 'cov_diag'))
        plt.close(fig)

        cor = np.diag(1 / np.sqrt(np.diag(cov_ext))) @ cov_ext @  np.diag(1 / np.sqrt(np.diag(cov_ext)))
        cor_mc = np.diag(1 / np.sqrt(np.diag(cov_mc))) @ cov_mc @  np.sqrt(np.diag(1 / np.diag(cov_mc)))

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        im = ax.imshow(cor_mc)
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cor_mc'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300, constrained_layout=True)
        im = ax.imshow(cor)
        fig.colorbar(im, ax=ax)
        fig.savefig(opj(odir, 'cor'))
        plt.close(fig)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')    
    parser.add_argument('--seed', type=int, default=65489873156946,
                        help="Random seed for the sampler.")

    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    with open(args.config, 'r') as yfile:
        config = yaml.safe_load(yfile)

    main(args.odir, config, args.specdir, args.seed)
