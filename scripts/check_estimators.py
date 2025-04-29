import os
import errno
import yaml
import pickle
import argparse

import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import map_utils
import torch
from torch.distributions import Normal, HalfNormal
from sbi.inference import SNPE, simulate_for_sbi, FMPE
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
    MultipleIndependent
)
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets import posterior_nn, flowmatching_nn

import sys
sys.path.append('..')
from sbi_bmode import (
    sim_utils, script_utils, compress_utils, custom_distributions)

opj = os.path.join

def normalize_simple(data, data_mean, data_std):
    '''
    Normalize data vector.

    Parameters
    ----------
    data : (nsim, ndata) array
        Data vector.
    data_mean : (ndata) array
        Mean over simulations
    data_std : (ndata) array
        Standard deviation over simulations.

    Returns
    -------
    data_norm : (nsim, ndata) array
        Normalized data.
    '''

    shape = data.shape
    if data.ndim == 1:
        data = data[np.newaxis,:]

    return ((data - data_mean) / data_std).reshape(shape)

def unnormalize_simple(data_norm, data_mean, data_std):
    '''
    Undo the normalization of a data vector.

    Parameters
    ----------
    data_norm : (nsim, ndata) array
        Normalized data vector.
    data_mean : (ndata) array
        Mean over simulations
    data_std : (ndata) array
        Standard deviation over simulations.

    Returns
    -------
    data : (nsim, ndata) array
        Unnormalized data.
    '''

    shape = data_norm.shape
    if data_norm.ndim == 1:
        data_norm = data_norm[np.newaxis,:]

    return (data_norm * data_std + data_mean).reshape(shape)

def get_prior(params_dict):
    '''
    Parse parameter dictionary and return pytorch prior distribution.

    Parameters
    ----------
    params_dict : dict
        Dictionary with parameters that we sample.

    Returns
    -------
    prior : list of torch.distributions objects
        Prior distributions for each parameter.
    param_names : list of str
        List of parameter names in same order as prior.
    '''

    prior = []
    param_names = []
    for param, prior_dict in params_dict.items():

        if prior_dict['prior_type'].lower() == 'normal':
            prior.append(Normal(*prior_dict['prior_params']))
        elif prior_dict['prior_type'].lower() == 'halfnormal':
            prior.append(HalfNormal(*prior_dict['prior_params']))
        elif prior_dict['prior_type'].lower() == 'truncatednormal':
            prior.append(custom_distributions.TruncatedNormal(*prior_dict['prior_params']))
        else:
            raise ValueError(f"{prior_dict['prior_type']=} not understood")
        param_names.append(param)

    # sbi needs the distributions to not be scalar.
    #return [p.expand(torch.Size([1])) for p in prior], param_names
    prior_list = [p.expand(torch.Size([1])) for p in prior]
    return MultipleIndependent(prior_list), param_names

def get_true_params(params_dict):
    '''
    Extract the true values of the parameters.

    Parameters
    ----------
    params_dict : dict
        Dictionary with parameters that we sample.

    Returns
    -------
    true_params : dict
        Dictionary with params names and values.
    '''

    true_params = {}
    for param_name, pd in params_dict.items():
        true_params[param_name] = pd['true_value']

    return true_params

def main(odir, imgdir config, n_samples, odir_compare=None, embed=False, embed_num_layers=2, embed_num_hiddens=25,
         fmpe=False, e_moped=False, n_moped=None, density_estimator_type='maf'):
    '''
    Run SBI.

    Parameters
    ----------
    odir : str
        Path to output directory.

    imgdir : 

    odir_compare :

    
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    n_samples : int
        Number of posterior samples to draw.
    embed : bool, optional
        Use an embedding network.
    embed_num_layers : int, optional
        Number of layers of embedding network
    embed_num_hiddens : int, optional
        Number of features in each hidden layer.
    fmpe : bool, optional
        Use Flow-Matching Posterior Estimation.
    e_moped : bool, optional
        Use e-MOPED compression for the data vector.
    n_moped : int, optional
        Number of simulations used for e-MOPED compression matrix.
    density_estimator_type : str, optional
        String denoting density estimator for NPE.
    '''

    ridx = 0
    theta = np.load(opj(odir, f'param_draws_round_{ridx:03d}'))
    x = np.load(opj(odir, f'data_draws_round_{ridx:03d}'))
    x_obs =  np.load(opj(odir, 'data_norm.npy'))
    
    # If needed, load posterior to compare against
    if odir_compare is not None:
        samples2compare = np.load(opj(odir, f'samples_round_{ridx:03d}.npy'))
    
    if embed:
        embedding_net = FCEmbedding(
           input_dim=x_obs.size,
           output_dim=num_parameters,
           num_layers=embed_num_layers,
           num_hiddens=embed_num_hiddens)
        neural_posterior = posterior_nn(model=density_estimator_type,
                                        embedding_net=embedding_net)
        inference = SNPE(prior=prior, density_estimator=neural_posterior)
    elif fmpe:
        net_builder = flowmatching_nn(
            model="resnet",
            num_blocks=3,
            hidden_features=24
        )
        inference = FMPE(prior, density_estimator=net_builder)
    else:
        neural_posterior = posterior_nn(model=density_estimator_type)
        inference = SNPE(prior, density_estimator=neural_posterior)

    ## Train the SNPE.
    #theta, x = simulate_for_sbi_mpi(
    #    cmb_simulator, proposal, param_names, n_train, data_size,
    #    rng_sims, comm, score_compress, mat_compress=mat_compress)

    # LOAD x and theta
    
    if fmpe:
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal,
        ).train(show_train_summary=True, training_batch_size=200, learning_rate=5e-4)
    else:
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal,
        ).train(show_train_summary=True, use_combined_loss=True)

    posterior = inference.build_posterior(density_estimator)
    proposal = posterior.set_default_x(x_obs)
    samples = posterior.sample((n_samples,), x=x_obs)


    # if comm.rank == 0:
    #     with open(opj(odir, 'posterior.pkl'), "wb") as handle:
    #         pickle.dump(posterior, handle)
    #     symlink_force(opj(odir, f'samples_round_{ridx:03d}.npy'), opj(odir, f'samples.npy'))
    #     np.save(opj(odir, 'data_uncompressed.npy'), x_obs_full)
    #     if norm_params:
    #         np.save(opj(odir, 'data_norm.npy'), x_obs)
    #         np.save(opj(odir, 'data.npy'), cmb_simulator.get_unnorm_data(x_obs))
    #     elif norm_simple:
    #         np.save(opj(odir, 'data_norm.npy'), x_obs)
    #         np.save(opj(odir, 'data.npy'), unnormalize_simple(x_obs, data_mean, data_std))
    #     else:
    #         np.save(opj(odir, 'data.npy'), x_obs_full)
    #     with open(opj(odir, 'config.yaml'), "w") as handle:
    #         yaml.safe_dump(config, handle)
    #     np.save(opj(odir, 'training_loss.npy'), np.asarray(inference.summary['training_loss']))
    #     np.save(opj(odir, 'validation_loss.npy'), np.asarray(inference.summary['validation_loss']))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')

    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for the training data.")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples of posterior")

    parser.add_argument('--embed', action='store_true',
                        help="Estimate and apply embedding (compression) network")
    parser.add_argument('--embed-num-layers', type=int, default=2,
                        help="Number of layers in embedding nework")
    parser.add_argument('--embed-num-hiddens', type=int, default=25,
                        help="Number of hidden units in each layer of the embedding network")
    parser.add_argument('--fmpe', action='store_true', help="Use Flow-Matching Posterior Estimation.")
    parser.add_argument('--density-estimator-type', type=str, default='maf',
                        help="pick from 'nsf', 'maf', 'mdn', 'made', 'zuko_maf' or 'zuko_nsf'")
                        
    args = parser.parse_args()

    odir = args.odir
    if comm.rank == 0:

        print(f'Running with arguments: {args}')
        print(f'Running with {comm.size} MPI rank(s)')

        os.makedirs(odir, exist_ok=True)
        with open(args.config, 'r') as yfile:
            config = yaml.safe_load(yfile)
    else:
        config = None
    config = comm.bcast(config, root=0)

    main(odir, config, args.specdir, args.r_true, args.seed, args.n_train,
         args.n_samples, args.n_rounds, args.pyilcdir, args.use_dbeta_map, args.deproj_dust,
         args.deproj_dbeta, args.fiducial_beta, args.fiducial_T_dust,
         no_norm=args.no_norm, score_compress=args.score_compress, embed=args.embed,
         embed_num_layers=args.embed_num_layers, embed_num_hiddens=args.embed_num_hiddens,
         fmpe=args.fmpe, e_moped=args.e_moped, n_moped=args.n_moped,
         density_estimator_type=args.density_estimator_type)
