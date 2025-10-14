10import os
import argparse

import numpy as np
from tarp import get_tarp_coverage
from sbi_bmode import compress_utils

opj = os.path.join

def main(odir, samples_file, true_params_file, data_file=None, emoped_data_file=None,
         emoped_params_file=None, seed=None):
    '''
    Run TARP coverage tests.

    Parameters
    ----------
    odir : str
        Path to output directory.
    samples_file : str
        Path to .npy file containing posterior samples (nsim, nsamp, nparam).
    true_params_file : str
        Path to .npy file containing true parameter values for each sim: (nsim, nparam).
    data_file : str, optional
        Path to .npy file with data vector corresponding to samples and true params (nsim, ndata)
    emoped_data_file : str, optional
        Path to .npy file with data for e-moped compression (nsim', ndata).
    emoped_params_file : str, optional
        Path to .npy file with parameters for e-moped compression (nsim', nparam).
    seed : int, optional
        Integer for numpy random.seed.
    '''

    # Needed because otherwise the TARP resampling all uses the same seed......
    seed = None
    
    samples = np.load(samples_file)
    true_params = np.load(true_params_file)

    assert samples.ndim == 3
    samples = samples.transpose(1, 0, 2)

    num_params = samples.shape[-1]
    num_sims = samples.shape[1]
    num_bootstrap = 100
    num_alpha_bins = num_sims // 10
    
    # First run joint coverage test.
    ecp, alpha = get_tarp_coverage(samples, true_params, references='random', seed=seed,
                                   bootstrap=False, num_alpha_bins=num_alpha_bins)

    ecp_boot, _ = get_tarp_coverage(samples, true_params, references='random', seed=seed,
                                    bootstrap=True, num_alpha_bins=num_alpha_bins,
                                    num_bootstrap=num_bootstrap)
    
    np.save(opj(odir, 'tarp_alpha'), alpha)
    np.save(opj(odir, 'tarp_ecp'), ecp)
    np.save(opj(odir, 'tarp_ecp_boot'), ecp_boot)

    # Then run coverage tests for the marginals.
    ecp_marginals = np.zeros((num_params, num_alpha_bins+1))
    ecp_marginals_boot = np.zeros((num_params, num_bootstrap, num_alpha_bins+1))
    
    for pidx in range(samples.shape[-1]):
        ecp, _ = get_tarp_coverage(samples[:,:,pidx:pidx+1], true_params[:,pidx:pidx+1],
                                   references='random', seed=seed,
                                   bootstrap=False, num_alpha_bins=num_alpha_bins)
        ecp_marginals[pidx] = ecp
        ecp, _ = get_tarp_coverage(samples[:,:,pidx:pidx+1], true_params[:,pidx:pidx+1],
                                   references='random', seed=seed,
                                   bootstrap=True, num_alpha_bins=num_alpha_bins, num_bootstrap=num_bootstrap)
        ecp_marginals_boot[pidx] = ecp
    
    np.save(opj(odir, 'tarp_ecp_marg'), ecp_marginals)
    np.save(opj(odir, 'tarp_ecp_marg_boot'), ecp_marginals_boot)    

    # Redo the joint TARP with reference distribution that given by P(theta | d) = C(d) + n,
    # where C(d) is the e-moped compression of the data and n is Gaussian with 10% of parameter width.
    if data_file is not None:
        data = np.load(data_file)        
        emoped_data = np.load(emoped_data_file)
        emoped_params = np.load(emoped_params_file)        

        compress_mat = compress_utils.get_e_moped_matrix(emoped_data, emoped_params)

        data_compressed = np.einsum('ij, aj -> ai', compress_mat, data)

        min_params = np.min(true_params, axis=0, keepdims=True)
        max_params = np.max(true_params, axis=0, keepdims=True)

        min_data = np.min(data_compressed, axis=0, keepdims=True)
        max_data = np.max(data_compressed, axis=0, keepdims=True)

        # Normalize to to unit hypercube.
        data_compressed = (data_compressed - min_data) / (max_data - min_data)
        # Rescale to parameter hypercube.
        data_compressed =  (max_params - min_params) * (data_compressed + min_params)

        # Add 10% noise.
        #noise = np.random.randn(num_sims, num_params) * 0.1 * (max_params - min_params)
        noise = np.random.randn(num_sims, num_params) * 1. * (max_params - min_params)        
        data_compressed += noise

        # Clip to parameter range.
        for pidx in range(num_params):
            data_compressed[:,pidx] = np.clip(data_compressed[:,pidx],
                                              a_min=min_params[0,pidx], a_max=max_params[0,pidx])

        # Note, need to set norm to False, because my reference points are not normalized.
        ecp, alpha = get_tarp_coverage(samples, true_params, references=data_compressed, seed=seed,
                                       bootstrap=False, num_alpha_bins=num_alpha_bins, norm=False)

        ecp_boot, _ = get_tarp_coverage(samples, true_params, references=data_compressed, seed=seed,
                                        bootstrap=True, num_alpha_bins=num_alpha_bins,
                                        num_bootstrap=num_bootstrap, norm=False)
    
        np.save(opj(odir, 'tarp_ecp_emoped'), ecp)
        np.save(opj(odir, 'tarp_ecp_emoped_boot'), ecp_boot)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir')
    parser.add_argument('--samples', help='Path to posterior samples .npy file')
    parser.add_argument('--test-params', help='Path to .npy file containing test set parameters')
    parser.add_argument('--test-data', help='Path to .npy file containing test set data')    
    parser.add_argument('--emoped-data', help='Path to .npy file with test set data for emoped')
    parser.add_argument('--emoped-params', help='Path to .npy file with test set aparameters for emoped')
    parser.add_argument('--seed', default=0, type=int, help='RNG seed')

    args = parser.parse_args()
    
    os.makedirs(args.odir, exist_ok=True)

    main(args.odir, args.samples, args.test_params, data_file=args.test_data,
         emoped_data_file=args.emoped_data, emoped_params_file=args.emoped_params, seed=args.seed)
