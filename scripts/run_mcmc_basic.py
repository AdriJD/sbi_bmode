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

def real2norm(params):

    return params * jnp.asarray([1e3, 1e2, 1e1, 1e2, 1e4])

def norm2real(params):

    return params / jnp.asarray([1e3, 1e2, 1e1, 1e2, 1e4])

def main(odir, config, specdir, data_file, seed, n_samples, n_chains):

    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior_combined, param_names, bounds = get_prior(params_dict)

    cmb_simulator = sim_utils.CMBSimulator(specdir, data_dict, fixed_params_dict)
    
    # Get prior mean.
    mean = prior_combined.get_mean()
    
    # Load data.
    data = jnp.asarray(np.load(data_file))

    # Get covariance matrix
    print(param_names)

    mean_dict = {}
    for idx, name in enumerate(param_names):
        mean_dict[name] = mean[idx]
    
    signal_spectra = cmb_simulator.get_signal_spectra(
        mean_dict['r_tensor'], mean_dict['A_lens'], mean_dict['A_d_BB'],
        mean_dict['alpha_d_BB'], mean_dict['beta_dust'])

    #def get_signal_spectra(self, r_tensor, A_lens, A_d_BB, alpha_d_BB, beta_dust):        

    noise_spectra = cmb_simulator.get_noise_spectra()
    cov = likelihood_utils.get_cov(
        np.asarray(signal_spectra), noise_spectra, cmb_simulator.bins, cmb_simulator.lmin,
        cmb_simulator.lmax, cmb_simulator.nsplit, cmb_simulator.nfreq)
    
    # Invert matrix
    icov = jnp.asarray(mat_utils.matpow(np.asarray(cov), -1))
    
    # Init sampler
    key = jax.random.key(seed)

    #print(mean)
    #print(prior_combined.sample(key))
    #key, subkey = jax.random.split(key)
    #print(prior_combined.sample(key))    

    tri_indices = sim_utils.get_tri_indices(cmb_simulator.nsplit, cmb_simulator.nfreq)

    data = data.reshape(tri_indices.shape[0], -1)

    # NOTE
    #icov = jnp.ones((tri_indices.shape[0], tri_indices.shape[0], data.shape[-1])) * jnp.eye(tri_indices.shape[0])[:,:,None]

    
    def logdens(params):

        params = norm2real(params)
        model = cmb_simulator.get_signal_spectra(*params)
        loglike = likelihood_utils.loglike(
            model, data, icov, tri_indices)
        
        logprior = prior_combined.log_prob(params)

        #jax.debug.print('{x}', x=params)
        #jax.debug.print('{x}', x=-(loglike + logprior))
        
        return -(loglike + logprior)
        
    class BSampler():

        def __init__(self, d):

            self.d = d
            self.grad_nlogp = jax.jit(jax.value_and_grad(logdens))

        def transform(self, x):
            return x

        def prior_draw(self, key):

            #return prior_combined.sample(key)
            return real2norm(prior_combined.sample(key))

    target = BSampler(mean.size)

    positive_indices = jnp.asarray([0])
    
    boundary = Boundary(
        target.d, where_positive=positive_indices)

    sampler = Sampler(target, varEwanted=5e-3, boundary=boundary, diagonal_preconditioning=True,
                     frac_tune1=0.4, frac_tune2=0.1, frac_tune3=0.1)

    print(logdens(mean))
    print(jax.grad(logdens)(mean))
    
    samples = sampler.sample(n_samples, num_chains=n_chains)

    samples = samples.reshape(n_samples * n_chains, -1)
    print(samples)
    mean = jnp.mean(samples, axis=0)
    print(mean)
    print(jnp.std(samples, axis=0))

    fig, axs = plt.subplots(nrows=5, dpi=300, constrained_layout=True, sharex=True, figsize=(4, 8))
    for idx in range(5):
        axs[idx].plot(samples[::100,idx] - mean[idx], label=param_names[idx])
        axs[idx].legend()
    fig.savefig(opj(odir, 'samples'))
    plt.close(fig)

    samples = norm2real(samples)
    
    np.save(opj(odir, 'samples_mcmc.npy'), np.asarray(samples))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')    
    parser.add_argument('--data', help="Path to data .npy file.")
    parser.add_argument('--seed', type=int, default=65489873156946,
                        help="Random seed for the sampler.")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples per chain.")
    parser.add_argument('--n_chains', type=int, default=1, help="number of independent chains.")

    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    with open(args.config, 'r') as yfile:
        config = yaml.safe_load(yfile)

    main(args.odir, config, args.specdir, args.data, args.seed, args.n_samples, args.n_chains)
