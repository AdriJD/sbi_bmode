import os
import yaml
import pickle
import argparse

import numpy as np
from mpi4py import MPI
from pixell import curvedsky
from optweight import map_utils
import torch
from torch.distributions import Normal, HalfNormal
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.sbiutils import seed_all_backends
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from sbi_bmode import spectra_utils, sim_utils, so_utils

opj = os.path.join
comm = MPI.COMM_WORLD

def parse_config(config):
    '''
    Split the config up into parts.
    
    Parameters
    ----------
    config : dict
        Dictionary with "data", "fixed_params" and "params" keys.

    Returns
    -------
    data_dict : dict
        Dictionary with data generations parameters.
    fixed_params_dict : dict
        Dictionary with parameters that we keep fixed.
    params_dict : dict
        Dictionary with parameters that we sample.    
    '''

    data_dict = config['data']
    fixed_params_dict = config['fixed_params']
    params_dict = config['params']    
    
    return data_dict, fixed_params_dict, params_dict

def get_prior(params_dict):
    '''
    Parse parameter dictonary and return pytorch prior distribution.

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

        Returns
        -------
        data : (ndata) array
            Data realization.
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
        
def main(odir, config, specdir, r_true, seed, n_train, n_samples, n_rounds):
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

    data_dict, fixed_params_dict, params_dict = parse_config(config)
    prior, param_names = get_prior(params_dict)
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

            #torch.set_num_threads(40)
            print('start', torch.get_num_threads())            
            density_estimator = inference.append_simulations(
                theta, x, proposal=proposal
            ).train()
            posterior = inference.build_posterior(density_estimator)
            proposal = posterior.set_default_x(x_obs)

            #torch.set_num_threads(1)
            print('end', torch.get_num_threads())

            
        proposal = comm.bcast(proposal, root=0)

    if comm.rank == 0:
        with open(opj(odir, 'posterior.pkl'), "wb") as handle:
            pickle.dump(posterior, handle)
        samples = posterior.sample((n_samples,), x=x_obs)
        np.save(opj(odir, 'samples.npy'), samples)

    comm.Barrier()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--config', help="Path to config yaml file.")
    parser.add_argument('--specdir')
    parser.add_argument('--r_true', type=float, default=0.1, help="True value of r.")
    parser.add_argument('--seed', type=int, default=225186655513525153114758457104258967436,
                        help="Random seed for the training data.")
    parser.add_argument('--n_train', type=int, default=1000, help="training samples for SNPE")
    parser.add_argument('--n_samples', type=int, default=10000, help="samples of posterior")
    parser.add_argument('--n_rounds', type=int, default=1, help="number of sequential rounds")

    args = parser.parse_args()

    subdirname = 'r%.2e_s%d_nt%d_ns%d_nr%d' % (args.r_true, args.seed, args.n_train,
                                               args.n_samples, args.n_rounds)
    odir = opj(args.odir, subdirname)
    if comm.rank == 0:
        os.makedirs(odir, exist_ok=True)
        with open(args.config, 'r') as yfile:
            config = yaml.safe_load(yfile)
    else:
        config = None
    config = comm.bcast(config, root=0)        
    print(config)
    main(odir, config, args.specdir, args.r_true, args.seed, args.n_train,
         args.n_samples, args.n_rounds)
