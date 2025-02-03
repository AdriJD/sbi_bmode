import os
import yaml
import pickle
import argparse

import numpy as np
import healpy as hp
from mpi4py import MPI
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
)
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets import posterior_nn, flowmatching_nn

import sys
sys.path.append('..')
from sbi_bmode import sim_utils, script_utils, compress_utils

opj = os.path.join
comm = MPI.COMM_WORLD

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
        else:
            raise ValueError(f"{prior_dict['prior_type']=} not understood")
        param_names.append(param)

    # sbi needs the distributions to not be scalar.
    return [p.expand(torch.Size([1])) for p in prior], param_names

def simulate_for_sbi_mpi(simulator, proposal, param_names, num_sims, ndata, seed, comm,
                         score_compress, mat_compress=None):
    '''
    Draw parameters from proposal and simulate data.

    Parameters
    ----------
    simulator : CMBSimulator object
        Instance of simulator class.
    proposal : any
        Proposal distribution for parameters, must have `sample` method.
    param_names : list of str
        List if the parameters in same order as samples from the proposal.
    num_sims : int
        Number of simulations to produce.
    ndata : int
        Size of the data vector
    seed : int, np.random._generator.Generator object
        Seed or random number generator object.
    comm : mpi4py.MPI.Intracomm object
        MPI communicator.
    score_compress : bool
        Apply score compression.
    mat_compress : (ntheta, ndata) array, optional
        Apply this compression matrix to the data vectors.

    Returns
    -------
    theta : (num_sims, ntheta) torch tensor, None
        Parameters draws, only on root rank.
    sims : (num_sims, ndata) torch tensor, None
        Data draws, only on root rank.
    '''

    div, mod = np.divmod(num_sims, comm.size)
    num_sims_per_rank = np.full(comm.size, div, dtype=int)
    num_sims_per_rank[:mod] += 1

    thetas = proposal.sample((num_sims_per_rank[comm.rank],))

    thetas = thetas.numpy().astype(np.float64)
    ntheta = thetas.shape[-1]
    sims = np.zeros((num_sims_per_rank[comm.rank], ndata))

    for idx, theta in enumerate(thetas):
        #print(comm.rank, idx)
        theta_dict = dict(zip(param_names, theta))

        draw = simulator.draw_data(
            r_tensor=theta_dict['r_tensor'], A_lens=theta_dict['A_lens'],
            A_d_BB=theta_dict['A_d_BB'], alpha_d_BB=theta_dict['alpha_d_BB'],
            beta_dust=theta_dict['beta_dust'], seed=seed)

        if mat_compress is not None:
            draw = np.dot(mat_compress, draw)
        if score_compress:
            draw = simulator.score_compress(draw)
        sims[idx] = draw

    if comm.rank == 0:
        thetas_full = np.zeros(num_sims * ntheta, dtype=np.float64)
        sims_full = np.zeros(num_sims * ndata, dtype=np.float64)
    else:
        thetas_full = None
        sims_full = None

    offsets_theta = np.zeros(comm.size)
    offsets_theta[1:] = np.cumsum(num_sims_per_rank * ntheta)[:-1]

    offsets_sims = np.zeros(comm.size)
    offsets_sims[1:] = np.cumsum(num_sims_per_rank * ndata)[:-1]

    comm.Gatherv(
        sendbuf=thetas,
        recvbuf=(thetas_full, num_sims_per_rank * ntheta, offsets_theta, MPI.DOUBLE), root=0)
    comm.Gatherv(
        sendbuf=sims,
        recvbuf=(sims_full, num_sims_per_rank * ndata, offsets_sims, MPI.DOUBLE), root=0)

    if comm.rank == 0:
        thetas_full = torch.as_tensor(thetas_full.reshape(num_sims, ntheta).astype(np.float32))
        sims_full = torch.as_tensor(sims_full.reshape(num_sims, ndata).astype(np.float32))

    return thetas_full, sims_full

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

def main(odir, config, specdir, r_true, seed, n_train, n_samples, n_rounds, pyilcdir, use_dbeta_map,
         deproj_dust, deproj_dbeta, no_norm=False, score_compress=False, embed=False, embed_num_layers=2,
         embed_num_hiddens=25, fmpe=False, e_moped=False, n_moped=None):
    '''
    Run SBI.

    Parameters
    ----------
    odir : str
        Path to output directory.
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.
    specdir : str
        Path to data directory containing power spectrum files.
    r_true : float
        Assumed true r parameter used to draw data.
    seed : int
        Global seed from which all RNGs are seeded.
    n_train : int
        Number of simulations to draw.
    n_samples : int
        Number of posterior samples to draw.
    n_rounds : int
        Number of simulation rounds, if 1: NPE, if >1, SNPE.
    pyilcdir: str
        Path to pyilc repository. If None, nilc not performed.
    use_dbeta_map: Bool
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of first moment w.r.t. beta and include it in
        auto- and cross-spectra in the data vector
    deproj_dust: Bool
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject dust in CMB NILC map.
    deproj_dbeta: Bool
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject first moment of dust w.r.t. beta in CMB NILC map.
    no_norm : bool, optional
        Apply no normalization to the data vector.
    score_compress : bool, optional
        Apply score-compression to the data.
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
    '''

    if score_compress and e_moped:
        raise ValueError("Cannot have both score and e-MOPED compression.")

    # Seed SBI. Annoyingly, this is using a bunch of global seeds. Every rank
    # gets a unique global seed.
    if comm.rank == 0:
        seed_global = np.random.SeedSequence(seed)
        seed_per_rank = seed_global.spawn(comm.size)
    else:
        seed_per_rank = None

    seed_per_rank = comm.scatter(seed_per_rank, root=0)
    # Per rank create one rng for SBI backend and one for drawing simulations.
    rng_sbi, rng_sims = [np.random.default_rng(s) for s in seed_per_rank.spawn(2)]
    seed_all_backends(int(rng_sbi.integers(2 ** 32 - 1)))

    data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
    prior, param_names = get_prior(params_dict)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    print(f'{prior.mean=}')
    print(f'{prior.stddev=}')    
    mean_dict = {}
    for idx, name in enumerate(param_names):
        mean_dict[name] = float(prior.mean[idx])

    norm_params = None
    norm_simple = False
    if not pyilcdir and not no_norm:
        norm_params = mean_dict         
    elif pyilcdir and not no_norm:
        norm_simple = True
        
    if r_true is not None:
        params_dict['r_tensor']['true_value'] = r_true
    true_params = get_true_params(params_dict)

    if score_compress:
        # For now, compute the score around the true parameter values.
        score_params = true_params
    else:
        score_params = None

    cmb_simulator = sim_utils.CMBSimulator(
        specdir, data_dict, fixed_params_dict, pyilcdir, use_dbeta_map, deproj_dust, deproj_dbeta,
        odir, norm_params=norm_params, score_params=score_params)

    proposal = prior

    mat_compress = None
    if e_moped:
        # Draw some simulations from the prior to estimate the compression matrix.
        theta, x = simulate_for_sbi_mpi(
            cmb_simulator, proposal, param_names, n_moped, cmb_simulator.size_data,
            rng_sims, comm, score_compress)

        if comm.rank == 0:
            mat_compress = compress_utils.get_e_moped_matrix(x.numpy(), theta.numpy())
        del theta, x
        mat_compress = comm.bcast(mat_compress, root=0)
        data_size = num_parameters
        
    elif score_compress:
        data_size = num_parameters
    else:
        data_size = cmb_simulator.size_data
        
    if norm_simple:
        # Draw some simulations from the prior to find a normalization.
        # Ideally this would be done during the first round of inference
        # but easier for now to do here. We shouldn't need too many sims.
        n_norm = 128
        _, x_norm = simulate_for_sbi_mpi(
            cmb_simulator, proposal, param_names, n_norm, data_size,
            rng_sims, comm, score_compress, mat_compress=mat_compress)
        if comm.rank == 0:
            data_mean = np.mean(np.asarray(x_norm), axis=0)
            data_std = np.std(np.asarray(x_norm), axis=0)        
        else:
            data_mean, data_std = None, None
        data_mean = comm.bcast(data_mean, root=0)
        data_std = comm.bcast(data_std, root=0)
        
    # Define observations. Important that all ranks agree on this.
    if comm.rank == 0:
        x_obs = cmb_simulator.draw_data(
            r_tensor=true_params['r_tensor'], A_lens=true_params['A_lens'],
            A_d_BB=true_params['A_d_BB'], alpha_d_BB=true_params['alpha_d_BB'],
            beta_dust=true_params['beta_dust'], seed=rng_sims)
    else:
        x_obs = None
    x_obs = comm.bcast(x_obs, root=0)
    
    x_obs_full = x_obs.copy() # We always want to save the full data vector.

    if e_moped:
        x_obs = np.dot(mat_compress, x_obs)
    if score_compress:
        x_obs = np.asarray(cmb_simulator.score_compress(x_obs))
    if comm.rank == 0:
        print(f'{x_obs.size=}')

    if norm_simple:
        x_obs = normalize_simple(x_obs, data_mean, data_std)
        
    if embed:
        embedding_net = FCEmbedding(
           input_dim=x_obs.size,
           output_dim=num_parameters,
           num_layers=embed_num_layers,
           num_hiddens=embed_num_hiddens)
        neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net)
        inference = SNPE(prior=prior, density_estimator=neural_posterior)
    elif fmpe:
        net_builder = flowmatching_nn(
            model="resnet",
            num_blocks=3,
            hidden_features=24
        )
        inference = FMPE(prior, density_estimator=net_builder)
    else:
        inference = SNPE(prior, density_estimator='maf')

    # Train the SNPE.
    for ridx in range(n_rounds):
        theta, x = simulate_for_sbi_mpi(
            cmb_simulator, proposal, param_names, n_train, data_size,
            rng_sims, comm, score_compress, mat_compress=mat_compress)

        if comm.rank == 0:

            if norm_simple:
                x = normalize_simple(x, torch.as_tensor(data_mean), torch.as_tensor(data_std))
            
            print(f'param draws : {theta}')
            print(f'data draws : {x}')
                       
            # Save parameters and data draws to disk for debugging.
            np.save(opj(odir, f'param_draws_round_{ridx:03d}'), np.asarray(theta))
            np.save(opj(odir, f'data_draws_round_{ridx:03d}'), np.asarray(x))

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

        proposal = comm.bcast(proposal, root=0)

    if comm.rank == 0:
        with open(opj(odir, 'posterior.pkl'), "wb") as handle:
            pickle.dump(posterior, handle)
        samples = posterior.sample((n_samples,), x=x_obs)
        np.save(opj(odir, 'samples.npy'), samples)
        np.save(opj(odir, 'data_uncompressed.npy'), x_obs_full)
        if norm_params:
            np.save(opj(odir, 'data_norm.npy'), x_obs)
            np.save(opj(odir, 'data.npy'), cmb_simulator.get_unnorm_data(x_obs))
        elif norm_simple:
            np.save(opj(odir, 'data_norm.npy'), x_obs)
            np.save(opj(odir, 'data.npy'), unnormalize_simple(x_obs, data_mean, data_std))
        else:
            np.save(opj(odir, 'data.npy'), x_obs_full)
        with open(opj(odir, 'config.yaml'), "w") as handle:
            yaml.safe_dump(config, handle)

    comm.Barrier()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')
    parser.add_argument('--pyilcdir', default=None, help="Path to pyilc repository. "\
                        "Set to None to use multifrequency PS instead of NILC PS.")
    parser.add_argument('--use_dbeta_map', default=False, help="Whether to build map of \
                        1st moment w.r.t. beta. Only relevant if usng NILC PS.")
    parser.add_argument('--deproj_dust', default=False, help="Whether to derpoject dust \
                       in CMB NILC map. Only relevant if usng NILC PS.")
    parser.add_argument('--deproj_dbeta', default=False, help="Whether to derpoject first  \
                    moment of dust w.r.t. beta in CMB NILC map. Only relevant if usng NILC PS.")
    
    parser.add_argument('--r_true', type=float, default=None, help="True value of r.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for the training data.")
    parser.add_argument('--n_train', type=int, default=1000, help="training samples for SNPE")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples of posterior")
    parser.add_argument('--n_rounds', type=int, default=1, help="number of sequential rounds")

    parser.add_argument('--score-compress', action='store_true',
                        help="Compress data vector with score compression")
    parser.add_argument('--no-norm', action='store_true', help="Do not normalize the data vector")
    parser.add_argument('--embed', action='store_true',
                        help="Estimate and apply embedding (compression) network")
    parser.add_argument('--embed-num-layers', type=int, default=2,
                        help="Number of layers in embedding nework")
    parser.add_argument('--embed-num-hiddens', type=int, default=25,
                        help="Number of hidden units in each layer of the embedding network")
    parser.add_argument('--fmpe', action='store_true', help="Use Flow-Matching Posterior Estimation.")
    parser.add_argument('--e-moped', action='store_true', help="Use e-MOPED to compress the data vector")
    parser.add_argument('--n-moped', type=int, help="Number of sims used to estimate e-moped matrix",
                        default=1000)

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
         args.n_samples, args.n_rounds, args.pyilcdir, args.use_dbeta_map, args.deproj_dust, args.deproj_dbeta,
         no_norm=args.no_norm, score_compress=args.score_compress, embed=args.embed,
         embed_num_layers=args.embed_num_layers, embed_num_hiddens=args.embed_num_hiddens,
         fmpe=args.fmpe, e_moped=args.e_moped, n_moped=args.n_moped)
