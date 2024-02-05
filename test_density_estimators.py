from generate_spectra import get_data_spectra
from sbi_posterior import main as sbi_posterior
import numpy as np
from time import time
import torch
from torch.distributions import half_normal
import matplotlib.pyplot as plt
import scipy.stats as stats

def bin_spectrum(cls, lmin=2, lmax=383, binwidth=10, return_error=False):
    """
    Bin a 1d array of power spectra using the standard binning.

    Arguments
    ---------
    cls : array_like
        Cls
    lmin : scalar, optional
        Minimum ell of the first bin.  Default: 2.
    lmax : scalar, optional
        ell of the last bin for a single spectrum. Default: 383
    binwidth : scalar, optional
        Width of each bin.  Default: 10.

    Returns
    -------
    ellb : array_like
        Bin centers
    clsb : array_like
        Binned power spectra
    clse : array_like
        Error bars on each bin (if return_error is True)
    """
    ncl = len(cls)
    ns = int(ncl/(lmax+1))
    starts = np.arange(0, ncl, lmax+1)
    ell = np.arange(ncl)
    internal_bins = np.arange(lmin, lmax + 1, binwidth)
    ibib, stst = np.meshgrid(internal_bins, starts)
    bins = (ibib+stst).flatten()
    bins = np.append(bins, [ncl])
    #ellb = stats.binned_statistic(ell, ell, "mean", bins=bins).statistic
    clsb = stats.binned_statistic(ell, cls, "mean", bins=bins).statistic
    return clsb


freqs=np.array([27., 39., 93., 145., 225., 280.])
nside=128
nsplits = 4
nCl = (len(freqs)**2)*(2*2)*(3*nside)*nsplits
#Per-row binning cooking
binwidth = 10
nspec = (len(freqs)**2)*(2*2)*nsplits #Spectra in a row
nbins = len(np.arange(2, 3*nside, binwidth))*nspec #Bins per spectrum

log_r_vector = np.log(np.load("r_vector_splits_masked.npy"))
data_tensor = np.load("data_tensor_splits_masked.npy")
nsamples = data_tensor.shape[0]
binned_data_tensor = np.empty((nsamples, nbins))
r_test_vals = np.logspace(-3, -0.7, num=10)
sample_save_de = np.zeros((5, 10, 10000))
t1 = time()
for i in range(nsamples):
    binned_data_tensor[i] = bin_spectrum(data_tensor[i])
print(f"Binned {nsamples} samples (binwidth {binwidth}) in {time()-t1:.3f}s")

density_estimators = ["mdn", "made", "maf", "maf_rqs", "nsf"]

#simulation parameters
mean_params = dict()
mean_params['A_lens'] = 1
mean_params['unit_beams'] = True

for i, de in enumerate(density_estimators):
    t1 = time()
    posterior = sbi_posterior(None, log_r_vector, binned_data_tensor, density_estimator=de)
    print(f" Trained in {time()-t1:.2f}s with density estimator {de}")
    for j, r in enumerate(r_test_vals):
        mean_params['r_tensor'] = r
        obs = get_data_spectra(freqs=freqs, 
                               nside=nside, 
                               outdir="./output", 
                               mean_params=mean_params, 
                               seed=np.random.randint(low=0, high=1e10), 
                               add_mask=True).flatten()
        binned_obs = bin_spectrum(obs)
        sample_save_de[i,j] = posterior.sample((10000,), x=binned_obs)[:,0]
stats_de = np.percentile(sample_save_de, [2.5, 50, 97.5], axis=2)
np.save("samples_density_estimators_binned10.npy", sample_save_de)
np.save("stats_density_estimators_binned10.npy", stats_de)

