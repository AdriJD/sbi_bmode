import os
import yaml
import pickle
import argparse

import numpy as np
from mpi4py import MPI
import torch
from sbi.inference import SNPE, simulate_for_sbi, FMPE
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    MultipleIndependent
)
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets import posterior_nn, flowmatching_nn

from sbi_bmode import (
    sim_utils, script_utils, compress_utils, custom_distributions)

opj = os.path.join
comm = MPI.COMM_WORLD

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
        print(f'{comm.rank=}, {idx=}, {theta=}')
        theta_dict = dict(zip(param_names, theta))

        draw = simulator.draw_data(
            theta_dict['r_tensor'],
            theta_dict['A_lens'],
            theta_dict['A_d_BB'],
            theta_dict['alpha_d_BB'],
            theta_dict['beta_dust'],
            seed,
            amp_beta_dust=theta_dict.get('amp_beta_dust'),
            gamma_beta_dust=theta_dict.get('gamma_beta_dust'),
            A_s_BB=theta_dict.get('A_s_BB'),
            alpha_s_BB=theta_dict.get('alpha_s_BB'),
            beta_sync=theta_dict.get('beta_sync'),
            amp_beta_sync=theta_dict.get('amp_beta_sync'),
            gamma_beta_sync=theta_dict.get('gamma_beta_sync'),
            rho_ds=theta_dict.get('rho_ds'))
        
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
        recvbuf=(thetas_full, np.array(num_sims_per_rank * ntheta, dtype=int),
                 np.array(offsets_theta, dtype=int), MPI.DOUBLE), root=0)
    comm.Gatherv(
        sendbuf=sims,
        recvbuf=(sims_full, np.array(num_sims_per_rank * ndata, dtype=int),
                 np.array(offsets_sims, dtype=int), MPI.DOUBLE), root=0)

    if comm.rank == 0:
        thetas_full = torch.as_tensor(thetas_full.reshape(num_sims, ntheta).astype(np.float32))
        sims_full = torch.as_tensor(sims_full.reshape(num_sims, ndata).astype(np.float32))

    return thetas_full, sims_full

def estimate_data_mean_and_std(n_norm, cmb_simulator, proposal, param_names, ndata,
                               seed, comm, score_compress, mat_compress=None):
    '''
    Draw simulations and estimate mean and std of the data distribution.

    Parameters
    ----------
    n_norm : int
        Number of simulations to draw.
    cmb_simulator : sim_utils.cmb_simulator instance.
        Simulator instance.
    proposal : any
        Proposal distribution for parameters, must have `sample` method.
    param_names : list of str
        List if the parameters in same order as samples from the proposal.
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
    data_mean : (ndata) ndarray
        Mean of data vector.
    data_std : (ndata) ndarray
        Standard deviations per element of data vector.
    '''

    _, x_norm = simulate_for_sbi_mpi(
        cmb_simulator, proposal, param_names, n_norm, ndata,
        rng_sims, comm, score_compress, mat_compress=mat_compress)
    if comm.rank == 0:
        data_mean = np.mean(np.asarray(x_norm), axis=0)
        data_std = np.std(np.asarray(x_norm), axis=0)
    else:
        data_mean, data_std = None, None
    data_mean = comm.bcast(data_mean, root=0)
    data_std = comm.bcast(data_std, root=0)

    return data_mean, data_std

def main(odir, config, specdir, seed, n_train, n_samples, n_rounds, pyilcdir, use_dust_map,
         use_dbeta_map, deproj_dust, deproj_dbeta, fiducial_beta, fiducial_T_dust,         
         use_sync_map=False, use_dbeta_sync_map=False, deproj_sync=False, deproj_dbeta_sync=False,
         fiducial_beta_sync=None, no_norm=False, score_compress=False, embed=False, embed_num_layers=2,
         embed_num_hiddens=25, embed_num_output_fact=3, fmpe=False, e_moped=False, n_moped=None,
         density_estimator_type='maf', coadd_equiv_crosses=True, apply_highpass_filter=True, n_test=None,
         previous_seed_file=None, data_mean_file=None, data_std_file=None, previous_data_obs_file=None,
         previous_data_file=None, previous_params_file=None):
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
    use_dust_map: bool
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of dust and include it in
        auto- and cross-spectra in the data vector    
    use_dbeta_map: bool
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of first moment w.r.t. beta and include it in
        auto- and cross-spectra in the data vector
    deproj_dust: bool
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject dust in CMB NILC map.
    deproj_dbeta: bool
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject first moment of dust w.r.t. beta in CMB NILC map.
    fiducial_beta: float, optional
        Only relevant if using nilc (pyilc dir is not None). If not None,
        use this value for beta when building nilc maps. Otherwise, use a separate value
        for each simulation.
    fiducial_T_dust: float, optional
        Only relevant if using nilc (pyilc dir is not None). If not None,
        use this value for T_dust when building nilc maps. Otherwise, use a separate value
        for each simulation.
    use_sync_map: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of synchrotron and include it in auto- and cross-spectra in
        the data vector
    use_dbeta_sync_map: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of first moment w.r.t. beta_synchrotron and include it in
        auto- and cross-spectra in the data vector
    deproj_sync: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject synchrotron in CMB NILC map.
    deproj_dbeta_sync: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject first moment of synchrotron w.r.t. beta in CMB NILC map.
    fiducial_beta_sync: float, optional
        Use this value for beta_synchrotron when building nilc maps.
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
    embed_num_output_fact : float, optional
        Number of output nodes for embedding network: int(embed_num_output_fact * nparam).
    fmpe : bool, optional
        Use Flow-Matching Posterior Estimation.
    e_moped : bool, optional
        Use e-MOPED compression for the data vector.
    n_moped : int, optional
        Number of simulations used for e-MOPED compression matrix.
    density_estimator_type : str, optional
        String denoting density estimator for NPE.
    coadd_equiv_crosses : bool, optional
        Whether to use the mean of e.g. comp1 x comp2 and comp2 x comp1 spectra.
    apply_highpass_filter: bool, optional
        Filter out signal modes below lmin in the simulations.
    n_test : int, optional
        Number of additional simulations written to disk for later testing.
    previous_seed_file: str, optional
        Path to .npy file containing seed(s) from previous run to which we are adding data.
    data_mean_file : str, optional
        Path to .npy file containing mean of data distribution used for previous run.
    data_std_file : str, optional
        Path to .npy file containing std of data distribution used for previous run.
    previous_data_obs_file : str, optional
        Path to .npy file containing previous (normalized) observed data vector.
    previous_data_file : str, optional
        Path to .npy file containing previous (normalized) data vectors.
    previous_params_file : str, optional
        Path to .npy file containing previous parameter vectors.
    '''

    if previous_seed_file:
        previous_seed = np.load(previous_seed_file)
        if seed in previous_seed:
            raise ValueError(f'Cannot re-use seeds: {previous_seed}, {seed}')        

    if data_mean_file or data_std_file:
        assert None not in (data_mean_file, data_std_file), 'Both mean and std files required'
        if comm.rank == 0:
            data_mean = np.load(data_mean_file)
            data_std = np.load(data_std_file)
        else:
            data_mean, data_std = None, None
        data_mean = comm.bcast(data_mean, root=0)
        data_std = comm.bcast(data_std, root=0)        

    if previous_data_obs_file:
        if comm.rank == 0:
            x_obs = np.load(previous_data_obs_file)
        else:
            x_obs = None
        x_obs = comm.bcast(x_obs)

    if previous_data_file or previous_params_file:
        assert None not in (previous_data_file, previous_params_file), 'Both data and params required'
        if comm.rank == 0:
            # Only need this on root rank.
            previous_data = np.load(previous_data_file)
            previous_params = np.load(previous_params_file)
            
    if score_compress or e_moped:
        raise ValueError('Not supported right now.')
        
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
    prior, param_names = script_utils.get_prior(params_dict)    
    prior = MultipleIndependent(prior)
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    config['param_order'] = param_names # To save the order that we actually used.

    if comm.rank == 0:
        print(f'{prior.mean=}')
        print(f'{prior.stddev=}')
    mean_dict = {}
    for idx, name in enumerate(param_names):
        mean_dict[name] = float(prior.mean[idx])

    norm_params = None
    norm_simple = False
    #if not pyilcdir and not no_norm:
    #    norm_params = mean_dict
    #elif pyilcdir and not no_norm:
    #    norm_simple = True
    if not no_norm:
        norm_simple = True

    true_params = script_utils.get_true_params(params_dict)

    if score_compress:
        # For now, compute the score around the true parameter values.
        score_params = true_params
    else:
        score_params = None

    cmb_simulator = sim_utils.CMBSimulator(
        specdir, data_dict, fixed_params_dict, pyilcdir=pyilcdir, use_dust_map=use_dust_map,
        use_dbeta_map=use_dbeta_map, use_sync_map=use_sync_map, use_dbeta_sync_map=use_dbeta_sync_map,
        deproj_dust=deproj_dust, deproj_dbeta=deproj_dbeta, deproj_sync=deproj_sync,
        deproj_dbeta_sync=deproj_dbeta_sync, fiducial_beta=fiducial_beta,
        fiducial_T_dust=fiducial_T_dust, fiducial_beta_sync=fiducial_beta_sync, odir=odir,
        norm_params=norm_params, score_params=score_params,
        coadd_equiv_crosses=coadd_equiv_crosses)

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

    if comm.rank == 0:
        print(f'{data_size=}')

    if norm_simple:
        # Draw some simulations from the prior to find a normalization.
        # Ideally this would be done during the first round of inference
        # but easier for now to do here. We shouldn't need too many sims.
        if data_mean_file is None:
            n_norm = 144
            data_mean, data_std = estimate_data_mean_and_std(
                n_norm, cmb_simulator, proposal, param_names, data_size,
                rng_sims, comm, score_compress, mat_compress=mat_compress)
            
    # Define observations. Important that all ranks agree on this.
    if previous_data_obs_file is None:
        if comm.rank == 0:
            x_obs = cmb_simulator.draw_data(
                true_params['r_tensor'],
                true_params['A_lens'],
                true_params['A_d_BB'],
                true_params['alpha_d_BB'],
                true_params['beta_dust'],
                rng_sims,
                amp_beta_dust=true_params.get('amp_beta_dust'),
                gamma_beta_dust=true_params.get('gamma_beta_dust'),
                A_s_BB=true_params.get('A_s_BB'),
                alpha_s_BB=true_params.get('alpha_s_BB'),
                beta_sync=true_params.get('beta_sync'),
                amp_beta_sync=true_params.get('amp_beta_sync'),
                gamma_beta_sync=true_params.get('gamma_beta_sync'),
                rho_ds=true_params.get('rho_ds'))
        else:
            x_obs = None
        x_obs = comm.bcast(x_obs, root=0)

        x_obs_full = x_obs.copy() # We always want to save the full data vector.

        if e_moped:
            x_obs = np.dot(mat_compress, x_obs)
        if score_compress:
            x_obs = np.asarray(cmb_simulator.score_compress(x_obs))
        if norm_simple:
            x_obs = compress_utils.normalize_simple(x_obs, data_mean, data_std)
    else:
        x_obs_full = None # We have loaded x_obs from disk, so no access to full vector.
            
    if comm.rank == 0:
        print(f'{x_obs.size=}')
            
    if embed:
        embedding_net = FCEmbedding(
           input_dim=x_obs.size,
           output_dim=int(embed_num_output_fact * num_parameters),
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

    # Train the SNPE.
    for ridx in range(n_rounds):
        theta, x = simulate_for_sbi_mpi(
            cmb_simulator, proposal, param_names, n_train, data_size,
            rng_sims, comm, score_compress, mat_compress=mat_compress)

        if comm.rank == 0:

            if norm_simple:
                x = compress_utils.normalize_simple(x, torch.as_tensor(data_mean), torch.as_tensor(data_std))

            if previous_params_file and ridx == 0:
                x = torch.cat([torch.as_tensor(previous_data), x], dim=0)
                theta = torch.cat([torch.as_tensor(previous_params), theta], dim=0)            
            
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
            samples = posterior.sample((n_samples,), x=x_obs)
            np.save(opj(odir, f'samples_round_{ridx:03d}.npy'), samples)

        proposal = comm.bcast(proposal, root=0)

    if n_test is not None:
        # We are using the proposal from the latest round for the test set.
        theta_test, x_test = simulate_for_sbi_mpi(
            cmb_simulator, proposal, param_names, n_test, data_size,
            rng_sims, comm, score_compress, mat_compress=mat_compress)

        if comm.rank == 0:
            if norm_simple:
                x_test = compress_utils.normalize_simple(x_test, torch.as_tensor(data_mean), torch.as_tensor(data_std))
        
    if comm.rank == 0:
        if previous_seed_file:
            seed = np.append(previous_seed, seed)
        np.save(opj(odir, 'random_seed.npy'), np.asarray(seed))
        with open(opj(odir, 'posterior.pkl'), "wb") as handle:
            pickle.dump(posterior, handle)
        script_utils.symlink_force(opj(odir, f'samples_round_{ridx:03d}.npy'), opj(odir, f'samples.npy'))
        if x_obs_full is not None:
            np.save(opj(odir, 'data_uncompressed.npy'), x_obs_full)
        if norm_params:
            np.save(opj(odir, 'data_norm.npy'), x_obs)
            np.save(opj(odir, 'data.npy'), cmb_simulator.get_unnorm_data(x_obs))
        elif norm_simple:
            np.save(opj(odir, 'data_norm.npy'), x_obs)
            np.save(opj(odir, 'data.npy'), compress_utils.unnormalize_simple(x_obs, data_mean, data_std))            
            np.save(opj(odir, 'data_mean.npy'), data_mean)
            np.save(opj(odir, 'data_std.npy'), data_std)            
        else:
            np.save(opj(odir, 'data.npy'), x_obs_full)
        with open(opj(odir, 'config.yaml'), "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        np.save(opj(odir, 'training_loss.npy'), np.asarray(inference.summary['training_loss']))
        np.save(opj(odir, 'validation_loss.npy'), np.asarray(inference.summary['validation_loss']))

        if n_test is not None:
            np.save(opj(odir, 'param_draws_test.npy'), np.asarray(theta_test))
            np.save(opj(odir, 'data_draws_test.npy'), np.asarray(x_test))
        
    comm.Barrier()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')

    parser.add_argument('--pyilcdir', default=None, help="Path to pyilc repository. "\
                        "Set to None to use multifrequency PS instead of NILC PS.")
    parser.add_argument('--use_dbeta_map', action='store_true', help="Whether to build map of \
                        1st moment w.r.t. beta. Only relevant if usng NILC PS.")
    parser.add_argument('--deproj_dust', action='store_true', help="Whether to deproject dust \
                        in CMB NILC map. Only relevant if usng NILC PS.")
    parser.add_argument('--deproj_dbeta', action='store_true', help="Whether to deproject first  \
                        moment of dust w.r.t. beta in CMB NILC map. Only relevant if usng NILC PS.")
    parser.add_argument('--fiducial_beta', type=float, default=None, help="Use this fiducial beta \
                         value to build NILC maps, required if dust/dbeta maps/deprojection are used.")
    parser.add_argument('--fiducial_T_dust', type=float, default=None, help="Use this fiducial temperature \
                         value to build NILC maps, required if dust/dbeta maps/deprojection are used.")
    parser.add_argument('--no-dust-map', action='store_true', help="Whether to build map of \
                        dust. Only relevant if usng NILC PS.")    
    parser.add_argument('--use_sync_map', action='store_true', help="Whether to build map of \
                        synchrotron. Only relevant if usng NILC PS.")
    parser.add_argument('--use_dbeta_sync_map', action='store_true', help="Whether to build map of \
                        1st moment w.r.t. beta synchrotron. Only relevant if usng NILC PS.")    
    parser.add_argument('--deproj_sync', action='store_true', help="Whether to deproject synchrotron \
                        in CMB NILC map. Only relevant if usng NILC PS.")
    parser.add_argument('--deproj_dbeta_sync', action='store_true', help="Whether to deproject first  \
                        moment of dust w.r.t. synchrotron beta in CMB NILC map. Only relevant if usng NILC PS.")
    parser.add_argument('--fiducial_beta_sync', type=float, default=None, help="Use this fiducial beta synchrotron \
                         value to build NILC maps, required if sync/dbeta_sync maps/deprojection are used.")    
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for the training data.")
    parser.add_argument('--n_train', type=int, default=1000, help="training samples for SNPE")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples of posterior")
    parser.add_argument('--n_rounds', type=int, default=1, help="number of sequential rounds")
    parser.add_argument('--n_test', type=int, help='Number of additional data draws written to disk.')    
    parser.add_argument('--score-compress', action='store_true',
                        help="Compress data vector with score compression")
    parser.add_argument('--no-norm', action='store_true', help="Do not normalize the data vector")
    parser.add_argument('--embed', action='store_true',
                        help="Estimate and apply embedding (compression) network")
    parser.add_argument('--embed-num-layers', type=int, default=2,
                        help="Number of layers in embedding nework")
    parser.add_argument('--embed-num-hiddens', type=int, default=25,
                        help="Number of hidden units in each layer of the embedding network")
    parser.add_argument('--embed-num-output-fact', type=int, default=3,
                        help="Number of output nodes for embedding network: int(embed_num_output_fact * nparam).")
    parser.add_argument('--fmpe', action='store_true', help="Use Flow-Matching Posterior Estimation.")
    parser.add_argument('--e-moped', action='store_true', help="Use e-MOPED to compress the data vector")
    parser.add_argument('--n-moped', type=int, help="Number of sims used to estimate e-moped matrix",
                        default=1000)
    parser.add_argument('--density-estimator-type', type=str, default='maf',
                        help="pick from 'nsf', 'maf', 'mdn', 'made', 'zuko_maf' or 'zuko_nsf'")
    parser.add_argument('--no-coadd-equiv-crosses', action='store_true',
                        help='Do not coadd comp1 x comp2 and comp2 x comp1 cross spectra in data vector')
    parser.add_argument('--no-highpass-filter', action='store_true',
                        help='Do not remove signal below lmin in simulated maps.')

    # Options for adding simulations to previous run.
    parser.add_argument('--previous-data', type=str,
                        help='Path to .npy file containing previous (normalized) data vectors.')
    parser.add_argument('--previous-params', type=str,
                        help='Path to .npy file containing previous parameter vectors.')
    parser.add_argument('--data-obs', type=str,
                        help='Path to .npy file containing previous (normalized) observed data vector.')
    parser.add_argument('--data-mean', type=str,
                        help='Path to .npy file containing mean of data distribution used for previous run.')
    parser.add_argument('--data-std', type=str,
                        help='Path to .npy file containing std of data distribution used for previous run.')    
    parser.add_argument('--previous-seed-file', type=str,
                        help='Path to .npy file containing seed integer(s)')
    
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

    main(odir, config, args.specdir, args.seed, args.n_train,
         args.n_samples, args.n_rounds, args.pyilcdir, not args.no_dust_map, args.use_dbeta_map,
         args.deproj_dust, args.deproj_dbeta, args.fiducial_beta, args.fiducial_T_dust,
         use_sync_map=args.use_sync_map, use_dbeta_sync_map=args.use_dbeta_sync_map,
         deproj_sync=args.deproj_sync, deproj_dbeta_sync=args.deproj_dbeta_sync,
         fiducial_beta_sync=args.fiducial_beta_sync,         
         no_norm=args.no_norm, score_compress=args.score_compress, embed=args.embed,
         embed_num_layers=args.embed_num_layers, embed_num_hiddens=args.embed_num_hiddens,
         embed_num_output_fact=args.embed_num_output_fact, fmpe=args.fmpe, e_moped=args.e_moped, n_moped=args.n_moped,
         density_estimator_type=args.density_estimator_type,
         coadd_equiv_crosses=not args.no_coadd_equiv_crosses,
         apply_highpass_filter=not args.no_highpass_filter, n_test=args.n_test,
         previous_seed_file=args.previous_seed_file, data_mean_file=args.data_mean, data_std_file=args.data_std,
         previous_data_obs_file=args.data_obs, previous_data_file=args.previous_data,
         previous_params_file=args.previous_params)
    
