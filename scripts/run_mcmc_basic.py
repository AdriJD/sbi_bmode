import os
import yaml
import pickle
import argparse

import jax
import jax.numpy as jnp
import blackjax
import matplotlib.pyplot as plt
import numpy as np
from pixell import curvedsky
from optweight import mat_utils
from mpi4py import MPI

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
         coadd_equiv_crosses=True, param_file=None, data_index_slice=None):
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
        path to .npy file containing observed spectra. Can be one set of spectra or
        multiple.
    seed : int
        Global seed from which all RNGs are seeded.    
    n_samples : int
        Number of mcmc samples per chain.
    n_chains : int
        Number of mcmc chains.
    coadd_equiv_crosses : bool, optional
        If set, assume we have used the mean of e.g. comp1 x comp2 and comp2 x comp1
        spectra.
    param_file : str, optional
        Path to .npy file containing true parameters. Can be one set or mutliple
        (same number as data_file). If not provided, uses true parameters from config.
    data_index_slice : slice, optional
        Slice into the indices of the input data set. Only used when `data_file`
        refers to a set of spectra.
    '''
    
    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)

    param_names_full = [n for n in params_dict.keys()]
    if comm.rank == 0:
        print(f'{data_dict=}')
        print(f'{fixed_params_dict=}')
        print(f'{params_dict=}')
    params_dict.pop('amp_beta_dust', None)
    params_dict.pop('gamma_beta_dust', None)  
    params_dict.pop('amp_beta_sync', None)
    params_dict.pop('gamma_beta_sync', None)

    prior_combined, param_names, transforms = get_prior(params_dict)

    cmb_simulator = sim_utils.CMBSimulator(specdir, data_dict, fixed_params_dict,
                                           coadd_equiv_crosses=coadd_equiv_crosses)
    
    # Load data.
    if data_index_slice is None:
        data_index_slice = slice(None, None, None)
    data_arr = np.load(data_file)

    if data_arr.ndim == 1:
        data_arr = data_arr[np.newaxis,:]    
    data_arr = data_arr[data_index_slice]
    if data_arr.ndim == 1:
        data_arr = data_arr[np.newaxis,:]
    
    data_arr = jnp.asarray(data_arr)
        
    num_sims = data_arr.shape[0]
    
    # Use the true parameters for the fiducial spectrum needed for the covariance matrix.    
    if param_file is not None:
        params = np.load(param_file)

        if params.ndim == 1:
            params[np.newaxis,:]
        params = params[data_index_slice]
        if params.ndim == 1:
            params[np.newaxis,:]

        params = jnp.asarray(params)
            
        assert params.shape[0] == num_sims

        # Turn this into a list of dicts. We can use param_names_full because those
        # are in the same order as the parameters.
        true_params_list = [dict(zip(param_names_full, p)) for p in params]

    else:
        # Use config.
        true_params = {}
        for param_name, pd in params_dict.items():
            true_params[param_name] = pd['true_value']    
        true_params_list = None

    idxs = np.arange(num_sims)
    idxs_per_rank = np.array_split(idxs, comm.size)
    num_sims_per_rank = [x.size for x in idxs_per_rank]

    if comm.rank == 0:
        print(f'{num_sims_per_rank=}')
    
    # Array to gather chains per rank.
    chains_on_rank = np.zeros(
        (num_sims_per_rank[comm.rank], n_chains, n_samples, len(param_names)),
        dtype=np.float32)
    
    for idx, ridx in enumerate(idxs_per_rank[comm.rank]):

        data = data_arr[ridx]
        if true_params_list is not None:
            true_params = true_params_list[ridx]

        print(f'rank : {comm.rank}, {true_params=}')
        
        signal_spectra = cmb_simulator.get_signal_spectra(
            true_params['r_tensor'], true_params['A_lens'], true_params['A_d_BB'],
            true_params['alpha_d_BB'], true_params['beta_dust'],
            A_s_BB=true_params.get('A_s_BB'), alpha_s_BB=true_params.get('alpha_s_BB'),
            beta_sync=true_params.get('beta_sync'), rho_ds=true_params.get('rho_ds'))
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

            # Loop over parameters, transform each of them and compute log abs Jacobian for each.
            params = {k: transforms[k].inv_func(v) for k, v in params.items()}

            clipped_abs_jac = lambda transform, y : jnp.clip(
                transform.abs_jac(y), jnp.finfo(y.dtype).smallest_normal)

            log_abs_jac = {k: clipped_abs_jac(transforms[k], v) for k, v in params.items()}

            sum_log_abs_jac = jax.tree.reduce(
                lambda acc, x: acc + jnp.sum(x), log_abs_jac, initializer=0.0)

            logprob = _logprob(params)

            return logprob + sum_log_abs_jac

        def get_prior_draw_transformed(rng_key):
            '''

            Parameters
            ----------
            rng_key : 
                RNG key.
            
            Returns
            -------
            draw : dict
                Parameter draw for each parameter
            '''

            draw = prior_combined.sample(rng_key)

            return {k: transforms[k].func(v) for k, v in zip(param_names, draw)}

        num_steps = 32

        # Init sampler
        rng_key = jax.random.key(seed)
        rng_key, *init_keys = jax.random.split(rng_key, n_chains + 1)
        init_keys = jnp.stack(init_keys)

        @jax.vmap
        def initialize_chain(rng_key):
            return get_prior_draw_transformed(rng_key)

        initial_positions = initialize_chain(init_keys)

        print(f'rank : {comm.rank}, {initial_positions=}')    

        def run_warmup(rng_key, initial_position):
            warmup = blackjax.window_adaptation(
                blackjax.hmc, logprob_transformed, is_mass_matrix_diagonal=True,
                num_integration_steps=num_steps,
                initial_step_size=1e-5, target_acceptance_rate=0.8)

            (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=512)
            return state, parameters

        rng_key, *warmup_keys = jax.random.split(rng_key, n_chains + 1)
        warmup_keys = jnp.stack(warmup_keys)
        states, parameters = jax.vmap(run_warmup)(warmup_keys, initial_positions)

        print(f'rank : {comm.rank}, {states=}')
        print(f'rank : {comm.rank}, {parameters=}')

        params_0 = jax.tree.map(lambda x: x[0], parameters)
        hmc = blackjax.hmc(logprob_transformed, **params_0)

        hmc_kernel = jax.jit(hmc.step)

        def inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        rng_key, *sample_keys = jax.random.split(rng_key, n_chains + 1)
        sample_keys = jnp.stack(sample_keys)

        batched_inference_loop = jax.vmap(
            lambda rng, state: inference_loop(rng, hmc_kernel, state, n_samples),
            in_axes=(0, 0))
        states = batched_inference_loop(sample_keys, states)

        print(f'rank : {comm.rank}, {states=}')

        mcmc_samples = states.position

        def apply_per_param_transform(mcmc_samples, transforms):
            return {k: transforms[k].inv_func(v) for k, v in mcmc_samples.items()}

        transform_fn = lambda position: apply_per_param_transform(position, transforms)
        transformed = jax.vmap(jax.vmap(transform_fn))(mcmc_samples)

        # Save samples in numpy array similar to run_sbi_basic output.
        for pidx, pname in enumerate(param_names):
            chains_on_rank[idx,:,:,pidx] = transformed[pname].astype(np.float32)

    samples = script_utils.gatherv_array(chains_on_rank, comm)

    if comm.rank == 0:
        np.save(opj(odir, 'samples.npy'), samples)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')    
    parser.add_argument('--data', help="Path to data .npy file. Can either contain a single \
                        or multiple datasets: (num_sims, ndata) shaped.")
    parser.add_argument('--params', help="Path to param .npy file. Can either contain a single \
                        or multiple parameters: (num_sims, nparam) shaped. If not provided, use true \
                        parameters from config file.")
    parser.add_argument('--seed', type=int, default=65489873156946,
                        help="Random seed for the sampler.")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples per chain.")
    parser.add_argument('--n_chains', type=int, default=1, help="number of independent chains.")
    parser.add_argument('--no-coadd-equiv-crosses', action='store_true',
                        help='Do not coadd comp1 x comp2 and comp2 x comp1 cross spectra in data vector')
    parser.add_argument('--data-index-slice', type=str, default=':',
                        help='Slice into data and params, e.g. "5" or ":10:2"')

    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    with open(args.config, 'r') as yfile:
        config = yaml.safe_load(yfile)

    main(args.odir, config, args.specdir, args.data, args.seed, args.n_samples, args.n_chains,
         coadd_equiv_crosses=not args.no_coadd_equiv_crosses, param_file=args.params,
         data_index_slice=script_utils.str_to_slice(args.data_index_slice))
