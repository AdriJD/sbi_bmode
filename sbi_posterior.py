import torch
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
import time

import argparse as ap
import os
opj = os.path.join

def main(r_prior, r_vector, data_tensor):
    prior = torch.distributions.half_normal.HalfNormal(torch.tensor([0.1]))
    inference = SNPE(prior=r_prior)
    data_tensor = torch.tensor(data_tensor, dtype=torch.float32)
    r_vector = torch.tensor(np.transpose([r_vector]), dtype=torch.float32)
    inference = inference.append_simulations(r_vector, data_tensor)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    return posterior

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--r_vector_file", type="str", action="store",
        help="Numpy array of r values for training sbi")
    parser.add_argument("--data_tensor_file", type="str", action="store",
        help="Numpy array of spectral data for training sbi")

if __name__ == '__main__':
    parser = get_parser(parser=None)
    args = parser.parse_args()
    main(**vars(args)) 
