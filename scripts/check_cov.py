import os
import yaml
import pickle
import argparse

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from pixell import curvedsky
from optweight import mat_utils
from mclmc.sampler import Sampler
from mclmc.boundary import Boundary

from sbi_bmode import (spectra_utils, sim_utils, so_utils, likelihood_utils,
                       script_utils)

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

    print(cmb_simulator.bins)
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
    #num_sims = 1000
    num_sims = 1000
    print(cmb_simulator.size_data)
    sims = np.zeros((num_sims, cmb_simulator.size_data))

    rng = np.random.default_rng(seed=seed)
    
    for idx in range(num_sims):
        print(idx)
        sims[idx] = cmb_simulator.draw_data(
            r_tensor=mean_dict['r_tensor'], A_lens=mean_dict['A_lens'],
            A_d_BB=mean_dict['A_d_BB'], alpha_d_BB=mean_dict['alpha_d_BB'],
            beta_dust=mean_dict['beta_dust'], seed=rng)

    cov_mc = np.cov(sims.T)

    print(cov.shape)
    print(cov_mc.shape)


    # prefactor = likelihood_utils.get_cov_prefactor(cmb_simulator.bins, cmb_simulator.lmin, cmb_simulator.lmax)

    # print(prefactor)
    # exit()



    # #rand_arr = np.random.randn(cmb_simulator.lmax + 1) * 10
    # rand_arr = np.ones(cmb_simulator.lmax + 1) * 10    

    # br = spectra_utils.bin_spectrum(rand_arr, np.arange(cmb_simulator.lmax + 1),
    #                                 cmb_simulator.bins, cmb_simulator.lmin, cmb_simulator.lmax,
    #                                 use_jax=False)

    # br2 = spectra_utils.bin_spectrum(rand_arr, np.arange(cmb_simulator.lmax + 1),
    #                                  cmb_simulator.bins, cmb_simulator.lmin, cmb_simulator.lmax,
    #                                  use_jax=True)
    # print(br)
    # print(br2)
    # exit()
    
    # NOTE
    #cov_mc *= 10
    
    tri_indices = sim_utils.get_tri_indices(cmb_simulator.nsplit, cmb_simulator.nfreq)
    #data = data.reshape(tri_indices.shape[0], -1)
    print(tri_indices.shape)
    cov_ext = np.ones((tri_indices.shape[0], tri_indices.shape[0], cov.shape[-1], cov.shape[-1]))
    cov_ext[:] = cov[:,:,:,np.newaxis]
    cov_ext[:] *= np.eye(cov.shape[-1])[np.newaxis,np.newaxis,:,:]

    cov_ext = np.transpose(cov_ext, (0, 2, 1, 3))
    cov_ext = cov_ext.reshape(cov_mc.shape)

    print(cov_ext.shape)

    sim_mean = np.mean(sims, axis=0)
    
    fig, axs = plt.subplots(nrows=2, dpi=300, sharex=True)
    for idx in range(10):
        axs[0].plot(sims[idx], lw=0.5)        
        axs[1].plot(sims[idx] - sim_mean, lw=0.5)
    fig.savefig(opj(odir, 'data_draws'))
    plt.close(fig)
    
    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    #im = ax.imshow(np.log10(np.abs(cov_mc)))
    im = ax.imshow(cov_mc)
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(odir, 'cov_mc'))
    plt.close(fig)

    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    #im = ax.imshow(np.log10(np.abs(cov_ext)))
    im = ax.imshow(cov_ext)
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(odir, 'cov'))
    plt.close(fig)
 
    fig, ax = plt.subplots(dpi=300, constrained_layout=True)
    im = ax.imshow(cov_ext - cov_mc)
    fig.colorbar(im, ax=ax)
    fig.savefig(opj(odir, 'cov_diff'))
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
