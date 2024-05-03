#from generate_spectra import get_data_spectra
#from sbi_posterior import main as sbi_posterior
import numpy as np
from time import time
import torch
import matplotlib.pyplot as plt
import argparse
from utils import bin_spectrum
import os
import pickle
opj = os.path.join

from torch.distributions import Normal, HalfNormal

import sbi
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from sbi_bmode import data

outdir = '/mnt/home/abayer/cca_project/output/'

parser = argparse.ArgumentParser()
parser.add_argument('--r_true', type=float, default=0.1, help="True value of r.")
parser.add_argument('--seed', type=int, default=-1, help="Random seed for the training data. -1 means random phases for each training data. seed+1 is used for observation. \
                                                          FIXME is there a way to tell sbi to use random, but predictable phases?")
parser.add_argument('--n_train', type=int, default=1000, help="training samples for SNPE")
parser.add_argument('--n_samples', type=int, default=10000, help="samples of posterior")
parser.add_argument('--n_rounds', type=int, default=1, help="number of sequential rounds")

args = parser.parse_args()
r_true = args.r_true
seed = args.seed
n_train = args.n_train
n_samples = args.n_samples
n_rounds = args.n_rounds

subdirname = 'r%.2e_s%d_nt%d_ns%d_nr%d' % (r_true, seed, n_train, n_samples, n_rounds)
print(subdirname)
os.makedirs(opj(outdir,subdirname), exist_ok=1)

x_obs = data.simulator(r_tensor=r_true, A_lens=1, A_d_BB=5, alpha_d_BB=-0.2, beta_dust=1.59, seed=seed+1) # define observations

# means: r_tensor=0.1, A_lens=1, A_d_BB=5, alpha_d_BB=-0.2, beta_dust=1.59
prior = [HalfNormal(0.1), Normal(1, 0.1), Normal(5, 5), Normal(-0.2, 0.5), Normal(1.59, 0.11)]
prior = [p.expand(torch.Size([1])) for p in prior]   # sbi needs the distributions to not be scalar
prior, num_parameters, prior_returns_numpy = process_prior(prior)

def simulator_wrapper(theta):
    """
    Need to convert to function of one parameter. Also torch<->numpy issues.
    Summarizes the output of the simulation and converts it to `torch.Tensor`.
    FIXME there may be a better way to do this.
    FIXME also want a way to vary the seeds in different simulations
    """
    theta = theta.numpy()
    return torch.as_tensor(data.simulator(r_tensor=theta[0], A_lens=theta[1], A_d_BB=theta[2], alpha_d_BB=theta[3], beta_dust=theta[4], seed=seed))

simulator = process_simulator(simulator_wrapper, prior, prior_returns_numpy)

check_sbi_inputs(simulator, prior)

inference = SNPE(prior)

posteriors = []
proposal = prior

# train the SNPE
for _ in range(n_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=n_train)

    # In `SNLE` and `SNRE`, you should not pass the `proposal` to
    # `.append_simulations()`
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal
    ).train()
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_obs)

# save the posterior (FIXME not tested yet)
with open(opj(outdir,subdirname,'posterior.pkl'), "wb") as handle:
    pickle.dump(posterior, handle)
# save the samples
samples = posterior.sample((10000,), x=x_obs)   # consider removing this if youre saving the posterior anyway. For SNPE still ened x_obs tho
np.save(opj(outdir,subdirname,'samples.npy'), samples)

# num_rounds_seq, (r_true), r_prior_min, r_prior_max, r_prior_str(log/lin), n_trian, n_test (default 10,000), density_estimator (use maf only fo rnow and ask what they tested). lots of snpe options like learning rate and clipping etc. can play with these but not yet.
# prioririze varying r_true and n_trian. half normal.
# sigr = ""
