import os
import yaml
import pickle
import argparse

import jax
import jax.numpy as jnp
from jax.scipy.special import logit, expit
import blackjax
import matplotlib.pyplot as plt
import numpy as np
from pixell import curvedsky
from optweight import mat_utils

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
    transforms : dict
        Dictionary containing a transformation instance per parameter.
    '''
    
    prior = []
    param_names = []
    transforms = {}
    for param, prior_dict in params_dict.items():
    
        if prior_dict['prior_type'].lower() == 'normal':
            prior.append(likelihood_utils.Normal(
                *prior_dict['prior_params']))
            transforms[param] = likelihood_utils.UnityTransform()
            
        elif prior_dict['prior_type'].lower() == 'halfnormal':
            prior.append(likelihood_utils.HalfNormal(
                *prior_dict['prior_params']))
            transforms[param] = likelihood_utils.LogTransform(0.)
            
        elif prior_dict['prior_type'].lower() == 'truncatednormal':
            prior.append(likelihood_utils.TruncatedNormal(
                *prior_dict['prior_params']))
            transforms[param] = likelihood_utils.LogOddsTransform(
                *prior_dict['prior_params'][2:])            
        else:
            raise ValueError(f"{prior_dict['prior_type']=} not understood")
        
        param_names.append(param)
        
    prior_combined = likelihood_utils.MultipleIndependent(prior)
        
    return prior_combined, param_names, transforms

def main(odir, config, specdir, data_file, seed, n_samples, n_chains,
         coadd_equiv_crosses=True):
    '''
    Run mcmc script.

    Parameters
    ----------
    odir : str
        Path to output directory.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    specdir : str
        Path to data directory containing power spectrum files.
    data_file : str
        path to .npy file containting observed spectra.
    seed : int
        Global seed from which all RNGs are seeded.    
    n_samples : int
        Number of mcmc samples per chain.
    n_chains : int
        Number of mcmc chains.
    coadd_equiv_crosses : bool, optional
        If set, assume we have used the mean of e.g. comp1 x comp2 and comp2 x comp1 spectra.
    '''
    
    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)

    print(f'{data_dict=}')
    print(f'{fixed_params_dict=}')
    print(f'{params_dict=}')
    params_dict.pop('amp_beta_dust')
    params_dict.pop('gamma_beta_dust')    

    print(params_dict)
    prior_combined, param_names, transforms = get_prior(params_dict)

    cmb_simulator = sim_utils.CMBSimulator(specdir, data_dict, fixed_params_dict,
                                           coadd_equiv_crosses=coadd_equiv_crosses)
    
    # Get prior mean.
    mean = prior_combined.get_mean()
    
    # Load data.
    data = jnp.asarray(np.load(data_file))

    # Get covariance matrix
    print(param_names)
                
    # NOTE actually better if we use the true parameters for the fiducial spectrum.
    true_params = {}
    for param_name, pd in params_dict.items():
        true_params[param_name] = pd['true_value']    

    print(f'{true_params=}')
        
    signal_spectra = cmb_simulator.get_signal_spectra(
        true_params['r_tensor'], true_params['A_lens'], true_params['A_d_BB'],
        true_params['alpha_d_BB'], true_params['beta_dust'])
    noise_spectra = cmb_simulator.get_noise_spectra()
    
    if coadd_equiv_crosses:    
        coadd_mat = likelihood_utils.get_coadd_transform_matrix(
            cmb_simulator.sels_to_coadd, sim_utils.get_ntri(cmb_simulator.nsplit, cmb_simulator.nfreq))
    else:
        coadd_mat = None

    cov = likelihood_utils.get_cov(
        np.asarray(signal_spectra), noise_spectra, cmb_simulator.bins, cmb_simulator.lmin,
        cmb_simulator.lmax, cmb_simulator.nsplit, cmb_simulator.nfreq, coadd_matrix=coadd_mat)
    
    # Invert matrix
    icov = jnp.asarray(mat_utils.matpow(np.asarray(cov), -1))
    
    tri_indices = sim_utils.get_tri_indices(cmb_simulator.nsplit, cmb_simulator.nfreq)
    if coadd_equiv_crosses:
        data = data.reshape(coadd_mat.shape[0], -1)
    else:
        data = data.reshape(tri_indices.shape[0], -1)

    def _logprob(params):
        '''
        Evaluate the posterior for a given set of parameters.

        Parameters
        ----------
        params : dict
            Dictionary with parameter value for each parameter.

        Returns
        -------
        logprob : float
            Posterior probability for input parameters.
        '''
        
        model = cmb_simulator.get_signal_spectra(
            params['r_tensor'], params['A_lens'], params['A_d_BB'],
            params['alpha_d_BB'], params['beta_dust'], A_s_BB=params.get('A_s_BB'),
            alpha_s_BB=params.get('alpha_s_BB'), beta_sync=params.get('beta_sync'),
            rho_ds=params.get('rho_ds'))
        
        loglike = likelihood_utils.loglike(
            model, data, icov, tri_indices, coadd_matrix=coadd_mat)

        ordered_params = jnp.array([params[k] for k in param_names])
        logprior = prior_combined.log_prob(ordered_params)
        
        return loglike + logprior

    def logprob_transformed(params):
        '''

        '''
        
        # Loop over parameters, transform each of them and compute log oab jacobian for each.
        params = {k: transforms[k].inv_func(v) for k, v in params.items()}

        clipped_abs_jac = lambda transform, y : jnp.clip(
            transform.abs_jac(y), jnp.finfo(y.dtype).smallest_normal)
        
        log_abs_jac = {k: clipped_abs_jac(transforms[k], v) for k, v in params.items()}
        
        sum_log_abs_jac = jax.tree.reduce(
            lambda acc, x: acc + jnp.sum(x), log_abs_jac, initializer=0.0)
        
        logprob = _logprob(params)
        
        return logprob 

    def get_prior_draw_transformed(rng_key):
        '''
        Returns
        -------
        draw : dict
            Parameter draw for each parameter
        '''
        
        draw = prior_combined.sample(rng_key)

        return {k: transforms[k].func(v) for k, v in zip(param_names, draw)}

    num_steps = 32
    num_chains = 1

    # Init sampler
    rng_key = jax.random.key(seed)
    rng_key, init_key = jax.random.split(rng_key)

    initial_position = get_prior_draw_transformed(init_key)
    print(f'{initial_position=}')    
    print('start warmup')
    print(logprob_transformed(initial_position))
    print(jax.grad(logprob_transformed)(initial_position))

    warmup = blackjax.window_adaptation(
        blackjax.hmc, logprob_transformed, is_mass_matrix_diagonal=True,
        num_integration_steps=num_steps,
        initial_step_size=1e-5, target_acceptance_rate=0.8)
    
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=512)

    print(state)
    print(parameters)
        
    hmc = blackjax.hmc(logprob_transformed, **parameters)
    hmc_kernel = jax.jit(hmc.step)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    rng_key, sample_key = jax.random.split(rng_key)
    states = inference_loop(sample_key, hmc_kernel, state, n_samples)

    mcmc_samples = states.position

    def apply_per_param_transform(mcmc_samples, transforms):
        return {k: transforms[k].inv_func(v) for k, v in mcmc_samples.items()}

    transform_fn = lambda position: apply_per_param_transform(position, transforms)
    transformed = jax.vmap(transform_fn)(mcmc_samples)

    print('alpha_d_BB')
    print(transformed['alpha_d_BB'])
    print(jnp.mean(transformed['alpha_d_BB']))
    print(jnp.std(transformed['alpha_d_BB']))

    print('r_tensor')
    print(transformed['r_tensor'])
    print(jnp.mean(transformed['r_tensor']))
    print(jnp.std(transformed['r_tensor']))

    print('A_lens')
    print(transformed['A_lens'])
    print(jnp.mean(transformed['A_lens']))
    print(jnp.std(transformed['A_lens']))

    print('A_d_BB')
    print(transformed['A_d_BB'])
    print(jnp.mean(transformed['A_d_BB']))
    print(jnp.std(transformed['A_d_BB']))

    print('beta_dust')
    print(transformed['beta_dust'])
    print(jnp.mean(transformed['beta_dust']))
    print(jnp.std(transformed['beta_dust']))

    # Save samples in numpy array similar to run_sbi_basic output.
    samples = np.zeros((n_samples, len(param_names)))
    for pidx, pname in enumerate(param_names):
        samples[:,pidx] = transformed[pname]
                       
    np.save(opj(odir, 'samples.npy'), samples)
        
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
    parser.add_argument('--no-coadd-equiv-crosses', action='store_true',
                        help='Do not coadd comp1 x comp2 and comp2 x comp1 cross spectra in data vector')

    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    with open(args.config, 'r') as yfile:
        config = yaml.safe_load(yfile)

    main(args.odir, config, args.specdir, args.data, args.seed, args.n_samples, args.n_chains,
         coadd_equiv_crosses=not args.no_coadd_equiv_crosses)
