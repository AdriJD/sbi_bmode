import os
import multiprocessing
import pickle
from time import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from pixell import curvedsky
from optweight import map_utils
import torch
from torch.distributions import Normal, HalfNormal
import sbi
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import optuna

from sbi_bmode import spectra_utils, sim_utils, so_utils

opj = os.path.join
comm = MPI.COMM_WORLD

class CMBSimulator():
    '''
    Generate CMB data vectors.

    Paramaters
    ----------
    specdir : str
        Path to data directory containing power spectrum files.
    '''
    
    def __init__(self, specdir):

        self.lmax = 200
        self.lmin = 30
        self.nside = 128

        delta_ell = 10
        self.bins = np.arange(self.lmin, self.lmax, delta_ell)

        self.cov_scalar_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_nobb.dat'), self.lmax)
        self.cov_tensor_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_r1.dat'), self.lmax)

        self.minfo = map_utils.MapInfo.map_info_healpix(self.nside)
        self.ainfo = curvedsky.alm_info(self.lmax)

        self.nsplit = 2
        freq_strings = ['f030', 'f040', 'f090', 'f150', 'f230', 'f290']
        self.freqs = [so_utils.sat_central_freqs[fstr] for fstr in freq_strings]
        nfreq = len(self.freqs)

        self.size_data = sim_utils.get_ntri(self.nsplit, nfreq) * (self.bins.size - 1)

        sensitivity_mode = 'goal'
        lknee_mode = 'optimistic'
        self.noise_cov_ell = np.ones((nfreq, 2, 2, self.lmax + 1))
        fsky = 0.1
        for fidx, fstr in enumerate(freq_strings):
            self.noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
                fstr, sensitivity_mode, lknee_mode, fsky, self.lmax)

        # Fixed parameters.
        self.freq_pivot_dust = 353
        self.temp_dust = 19.6

    def gen_data(self, r_tensor, A_lens, A_d_BB, alpha_d_BB, beta_dust, seed):
        '''
        Draw data realization.

        Parameters
        ----------
        r_tensor : float
            Tensor-to-scalar ratio.
        A_lens : float
            Amplitude of lensing contribution to BB.        
        A_d_BB : float
            Dust amplitude.
        alpha_d_BB : float
            Dust spatial power law index.
        beta_dust : float
            Dust frequency power law index.
        seed : int, np.random._generator.Generator object
            Seed or random number generator object.
        '''
        
        if seed == -1:
            seed = None # unpredicable
        seed = np.random.default_rng(seed=seed)

        omap = sim_utils.gen_data(
            A_d_BB, alpha_d_BB, beta_dust, self.freq_pivot_dust, self.temp_dust,
            r_tensor, A_lens, self.freqs, seed, self.nsplit, self.noise_cov_ell,
            self.cov_scalar_ell, self.cov_tensor_ell, self.minfo, self.ainfo)
        spectra = sim_utils.estimate_spectra(omap, self.minfo, self.ainfo)

        data = sim_utils.get_final_data_vector(spectra, self.bins, self.lmin, self.lmax)

        return data

def simulate_for_sbi_mpi(simulator, proposal, num_sims, ndata, seed, comm):
    '''
    Draw parameters from proposal and simulate data.
    
    Parameters
    ----------
    simulator : CMBSimulator object
        Instance of simulator class.
    proposal : any
        Proposal distribution for parameters, must have `sample` method.
    num_sims : int
        Number of simulations to produce.
    ndata : int
        Size of the data vector
    seed : int, np.random._generator.Generator object
        Seed or random number generator object.
    comm : mpi4py.MPI.Intracomm object
        MPI communicator.
    
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
        sims[idx] = simulator.gen_data(
            theta[0], theta[1], theta[2], theta[3], theta[4], seed)

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
        
def main(trial, odir, specdir, r_true, seed, n_train, n_samples, n_rounds, num_atoms, training_batch_size, learning_rate, clip_max_norm):
    '''
    Run SBI.

    Parameters
    ----------
    odir : str
        Path to output directory.
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
    '''

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

    # Set means: r_tensor=0.1, A_lens=1, A_d_BB=2, alpha_d_BB=-0.2, beta_dust=1.59.
    prior = [HalfNormal(0.1), Normal(1, 0.1), Normal(5, 2), Normal(-0.2, 0.5),
             Normal(1.59, 0.11)]
    # sbi needs the distributions to not be scalar.
    prior = [p.expand(torch.Size([1])) for p in prior]  
    prior, num_parameters, prior_returns_numpy = process_prior(prior)

    cmb_simulator = CMBSimulator(specdir)

    # Define observations. Important that all ranks agree on this.
    if comm.rank == 0:
        x_obs = cmb_simulator.gen_data(r_true, 1, 5, -0.2, 1.59, rng_sims)
    else:
        x_obs = None
    x_obs = comm.bcast(x_obs, root=0)

    inference = SNPE(prior)
    proposal = prior

    # Train the SNPE
    for _ in range(n_rounds):

        theta, x = simulate_for_sbi_mpi(
            cmb_simulator, proposal, n_train, cmb_simulator.size_data, rng_sims, comm)

        if comm.rank == 0:
            density_estimator = inference.append_simulations(
                theta, x, proposal=proposal
            ).train(num_atoms=num_atoms, training_batch_size=training_batch_size, 
                    learning_rate=learning_rate, clip_max_norm=clip_max_norm,
                    validation_fraction=0.1, stop_after_epochs=20, 
                    max_num_epochs=1000, calibration_kernel=None, 
                    resume_training=False, force_first_round_loss=False, discard_prior_samples=False, 
                    use_combined_loss=False, retrain_from_scratch=False, 
                    show_train_summary=False, dataloader_kwargs=None)
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_obs)
            best_validation_log_prob = inference.summary['best_validation_log_prob'][0]

        proposal = comm.bcast(proposal, root=0)  # broadcast for next round which uses proposal to simulate more maps

    if comm.rank == 0:
        with open(opj(odir, 'posterior_trial%d.pkl'%trial.number), "wb") as handle:
            pickle.dump(posterior, handle)
        samples = posterior.sample((n_samples,), x=x_obs)
        np.save(opj(odir, 'samples_trial%d.npy'%trial.number), samples)

    comm.Barrier()

    if comm.rank == 0:
        return  - best_validation_log_prob   # return the loss (-logp) for optuna

def optuna_main(trial, odir, specdir, r_true, seed, n_train, n_samples, n_rounds, 
                num_atoms0, num_atoms1, 
                training_batch_size0, training_batch_size1, 
                learning_rate0, learning_rate1, 
                clip_max_norm0, clip_max_norm1):
    """
    This function prepares the parmaeters form optuna to pass into main().
    Could merge this into main, but perhaps this is a little neater.
    trial: optuna trial
    """

    print('Trial number: {}'.format(trial.number))
    print(trial)
    num_atoms           = trial.suggest_int("num_atoms", num_atoms0, num_atoms1, log=False)
    training_batch_size = trial.suggest_int("training_batch_size", training_batch_size0, training_batch_size1)
    learning_rate       = trial.suggest_float("learning_rate", learning_rate0, learning_rate1, log=True)
    clip_max_norm       = trial.suggest_float("clip_max_norm", clip_max_norm0, clip_max_norm1, log=True)

    return main(trial, odir, specdir, r_true, seed, n_train, n_samples, n_rounds, num_atoms, training_batch_size, learning_rate, clip_max_norm)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--specdir')
    parser.add_argument('--r_true', type=float, default=0.1, help="True value of r.")
    parser.add_argument('--seed', type=int, default=225186655513525153114758457104258967436,
                        help="Random seed for the training data.")
    parser.add_argument('--n_train', type=int, default=1000, help="training samples for SNPE")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples of posterior")
    parser.add_argument('--n_rounds', type=int, default=1, help="number of sequential rounds")
    parser.add_argument('--n_trials', type=int, default=1, help="number of optuna trials")
    parser.add_argument('--num_atoms0', type=int, default=1, help="number of atoms for sbi (optuna lower)")
    parser.add_argument('--num_atoms1', type=int, default=100, help="number of atoms for sbi (optuna higher)")
    parser.add_argument('--training_batch_size0', type=int, default=20, help="training_batch_size for sbi (optuna lower)")
    parser.add_argument('--training_batch_size1', type=int, default=120, help="training_batch_size for sbi (optuna higher)")
    parser.add_argument('--learning_rate0', type=float, default=1e-6, help="learning_rate for sbi (optuna lower)")
    parser.add_argument('--learning_rate1', type=float, default=1, help="learning_rate for sbi (optuna higher)")
    parser.add_argument('--clip_max_norm0', type=float, default=1e-4, help="clip_max_norm for sbi (optuna lower)")
    parser.add_argument('--clip_max_norm1', type=float, default=1e2, help="clip_max_norm for sbi (optuna higher)")

    args = parser.parse_args()

    n_trials = args.n_trials

    subdirname = 'r%.2e_s%d_nt%d_ns%d_nr%d' % (args.r_true, args.seed, args.n_train,
                                               args.n_samples, args.n_rounds)
    odir = opj(args.odir, subdirname)
    if comm.rank == 0:
        os.makedirs(odir, exist_ok=True)
    comm.Barrier()

    # un sbi
    objective = lambda trial: optuna_main(trial, odir, args.specdir, args.r_true, args.seed, args.n_train,
                                          args.n_samples, args.n_rounds, 
                                          args.num_atoms0, args.num_atoms1, 
                                          args.training_batch_size0, args.training_batch_size1, 
                                          args.learning_rate0, args.learning_rate1, 
                                          args.clip_max_norm0, args.clip_max_norm1)
    
    # OPTUNA FIXME will this work with the mpi functionality?
    if comm.rank == 0:
        #os.makedirs(opj(odir, 'optuna'), exist_ok=1)
        study_name = 'bmode_%s_na%d-%d_bs%d-%d_lr%.2e-%.2e_cm%.2e-%.2e' % \
            (subdirname,
            args.num_atoms0, args.num_atoms1, 
            args.training_batch_size0, args.training_batch_size1, 
            args.learning_rate0, args.learning_rate1, 
            args.clip_max_norm0, args.clip_max_norm1)
        storage    = 'sqlite:///bmode.db'
        
        sampler = optuna.samplers.TPESampler(n_startup_trials=10)
        #if study_name in optuna.study.get_all_study_names(storage=storage):
        #    optuna.delete_study(study_name=study_name, storage=storage)
        study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, load_if_exists=1)
        study.optimize(objective, n_trials, gc_after_trial=False)

        # get the number of pruned and complete trials
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # print some verbose
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials:   ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        # parameters of the best trial
        trial = study.best_trial
        print("Best trial: number {}".format(trial.number))
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
