import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import binned_statistic

from sbi_bmode import spectra_utils, sim_utils

class Normal():
    '''
    One-dimensional normal distributions.

    Parameters
    ----------
    mean : float
        Mean of distribution.
    sigma : float
        Standard deviation of distribution.
    '''
    
    def __init__(self, mean, sigma):

        self.mean = mean
        self.sigma = sigma

        if halfnormal:
            assert self.mean == 0.
        
    def log_prob(self, param):                
        '''
        Evaluate unnormalized Gaussian log prior.

        Parameters
        ----------
        param : float or array
            Return the log probability for this parameter value.
    
        Returns
        -------
        logprob : float
            The log probability.
        '''
    
        return -0.5 * jnp.sum(((param - self.mean) / self.sigma) ** 2)

    def sample(self, key):
        '''
        Draw sample from distribution.

        Returns
        -------
        '''

        draw = self.sigma * jax.random.normal(key)
        draw = draw + self.mean

        return draw

class HalfNormal():
    '''
    One-dimensional halfnormal distributions 

    Parameters
    ----------
    sigma : float
        Standard deviation of distribution.
    '''
    
    def __init__(self, sigma, halfnormal=False):

        self.mean = mean

    @property
    def sigma(self):
        return self.sigma * sqrt(2 / np.pi)
            
    def log_prob(self, param):                
        '''
        Evaluate unnormalized Gaussian log prior.

        Parameters
        ----------
        param : float or array
            Return the log probability for this parameter value.
    
        Returns
        -------
        logprob : float
            The log probability.
        '''
    
        return -0.5 * jnp.sum((param / self.sigma) ** 2)

    def sample(self, key):
        '''
        Draw sample from distribution.

        Returns
        -------
        '''

        draw = self.sigma * jax.random.normal(key)
        draw = jnp.abs(draw)

        return draw
    
class MultipleIndependent():
    '''
    Wrap sequence of one-dimensional distributions into a
    single distribution of independent distributions.

    Parameters
    ----------
    distributions : list of distributions
        Distributions to be wrapped in order.    
    '''

    def __init__(self, distributions):

        self.distributions = distributions
        
    def log_prob(self, params):
        '''
        Evalute multivariate distribution.

        Parameters
        ----------
        params : array
            Array of parameters.

        Returns
        -------
        logprob : float
            Logprobability        
        '''

        logprob = 0.
        for distr, param in zip(self.distributions, params):
            logprob += distr.log_prob(param)

        return logprob

    def get_mean(self):
        '''
        Return mean of distribution.
        
        Returns
        -------
        mean : (nparam) array
        
        '''

        mean = jnp.zeros(len(self.distributions))
        for idx, distr in enumerate(self.distributions):
            mean = mean.at[idx].set(distr.mean)

        return mean    

    def sample(self, key):
        '''
        Draw from the multivariate distribution.

        Parameters
        ----------
        key

        Returns
        -------
        draw : (nparam) array
        '''
        
        draw = jnp.zeros(len(self.distributions))        
        for idx, distr in enumerate(self.distributions):

            key, rng_subkey = jax.random.split(key)            
            draw = draw.at[idx].set(distr.sample(rng_subkey))

        return draw
            
def delta(*args):
    '''
    Return 1 if all arguments are equal, else 0.

    Parameters
    ----------
    *args :

    Returns
    -------
    equal : int
    '''
    
    return int(len(set(args)) == 1)
    
def get_cov_element(tri_idx, tri_jdx, signal_spectra, noise_spectra):
    '''

    Parameters
    ----------
    tri_idx : tuple
        sidx1, fidx1, sidx2, fidx2 indices
    tri_jdx : tuple
        sjdx1, fjdx1, sjdx2, fjdx2 indices
    signal_spectra : (nfreq, nfreq, nbin)
        Binned signal frequency cross spectra.
    noise_spectra : (nfreq, nfreq, nbin) array
        Binned noise frequency cross spectra.
    '''

    sidx1, fidx1, sidx2, fidx2 = tri_idx
    sjdx1, fjdx1, sjdx2, fjdx2 = tri_jdx

    # First term.
    s_cov1 = signal_spectra[fidx1,fjdx1]
    s_cov2 = signal_spectra[fidx2,fjdx2]

    n_cov1 = noise_spectra[fidx1,fjdx1]
    n_cov2 = noise_spectra[fidx2,fjdx2]
    
    term_a = (s_cov1 + n_cov1 * delta(sidx1, sjdx1))
    term_a *= (s_cov2 + n_cov2 * delta(sidx2, sjdx2))
    term_a *= (1 - delta(sidx1, sidx2)) * (1 - delta(sjdx1, sjdx2))

    # Second term.
    s_cov1 = signal_spectra[fidx1,fjdx2]
    s_cov2 = signal_spectra[fidx2,fjdx1]

    n_cov1 = noise_spectra[fidx1,fjdx2]
    n_cov2 = noise_spectra[fidx2,fjdx1]
    
    term_b = (s_cov1 + n_cov1 * delta(sidx1, sjdx2))
    term_b *= (s_cov2 + n_cov2 * delta(sidx2, sjdx1))
    term_b *= (1 - delta(sidx1, sidx2)) * (1 - delta(sjdx1, sjdx2))

    return term_a + term_b

def get_cov_prefactor(bins, lmin, lmax):
    '''
    '''

    ells = np.arange(lmax+1)
    num_modes = (2 * ells + 1)
    #num_modes_per_bin = spectra_utils.bin_spectrum(num_modes, ells, bins, lmin, lmax)

    num_modes_per_bin = binned_statistic(
        ells, num_modes, bins=bins, range=(lmin, lmax+1), statistic='sum')[0]
    
    return 1 / num_modes_per_bin
    
def get_cov(signal_spectra, noise_spectra, bins, lmin, lmax, nsplit, nfreq):
    '''

    Parameters
    ----------
    signal_spectra : (nfreq, nfreq, nbin)
        Binned signal frequency cross spectra.
    noise_spectra : (nfreq, nfreq, nbin) array
        Binned noise frequency cross spectra.
    bins : (nbin + 1) array
        Bin edges.

    Returns
    -------
    cov : (ntri, ntri, nbin) array
        Diagonal of covariance matrix.
    '''

    nbin = bins.size - 1
    assert signal_spectra.shape == (nfreq, nfreq, nbin)
    assert noise_spectra.shape == (nfreq, nfreq, nbin)

    prefactor = get_cov_prefactor(bins, lmin, lmax)
    tri_indices = sim_utils.get_tri_indices(nsplit, nfreq)
    ntri = tri_indices.shape[0]
    cov = np.zeros((ntri, ntri, nbin))
    
    for idx, tri_idx in enumerate(tri_indices):
        for jdx, tri_jdx in enumerate(tri_indices):        

            cov[idx,jdx] = get_cov_element(
                tri_idx, tri_jdx, signal_spectra, noise_spectra)
            
    cov *= prefactor

    return cov

def get_diff(data, model, tri_indices):
    '''
    Parameters
    ----------
    data : (ntri, nbin)

    model : (nfreq, nfreq, nbin)

    Returns
    -------
    diff : (ntri, nbin)
    '''

    diff = jnp.zeros_like(data)

    for idx, (sidx1, fidx1, sidx2, fidx2) in enumerate(tri_indices):

        diff = diff.at[idx].set(data[idx] - model[fidx1,fidx2])

    return diff
    
#def loglike(params, forward_model, data, icov, tri_indices):
def loglike(model, data, icov, tri_indices):    
    '''

    '''

    # IS FORWARD MODEL JUST GET_SIGNAL_SPECTRA?
    #model = forward_model(params)
    diff = get_diff(data, model, tri_indices)
    return -0.5 * jnp.einsum('al, abl, bl', diff, icov, diff)

# def logprior(params):
#     '''

#     '''

#     pass

# def logpost(params, data, icov, tri_indices):
#     '''

#     '''

#     loglike = loglike(params, data, icov)
#     logprior = logprior(params)

#     return loglike + logprior
