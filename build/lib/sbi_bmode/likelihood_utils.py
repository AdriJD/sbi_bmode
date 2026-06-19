import jax
import jax.numpy as jnp
from jax.scipy.special import logit, expit
import numpy as np
from scipy.stats import binned_statistic

from sbi_bmode import spectra_utils, sim_utils

class LogOddsTransform():
    '''
    Translated log-odds transform class to transform a parameter defined on a bounded space
    defined by the open interval (a, b), i.e. not including a or b, to an unbounded space.

    parameters
    ----------
    lower_bound : float
        Lower bound a of parameter space.
    upper_bound : float
        Upper bound b of parameter space.

    Notes
    -----
    See https://mc-stan.org/docs/reference-manual/transforms.html.
    '''
    
    def __init__(self, lower_bound, upper_bound):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def func(self, x):
        return logit((x - self.lower_bound) / (self.upper_bound - self.lower_bound))

    def inv_func(self, y):
        return self.lower_bound + (self.upper_bound - self.lower_bound) * expit(y)

    def abs_jac(self, y):
        return (self.upper_bound - self.lower_bound) * expit(y) * (1 - expit(y))

class LogTransform():
    '''
    Logarithmic transform class to transform a parameter defined on a bounded space
    defined by the open interval (a, inf), i.e. not including a, to an unbounded space.

    parameters
    ----------
    lower_bound : float
        Lower bound a of parameter space.

    Notes
    -----
    See https://mc-stan.org/docs/reference-manual/transforms.html.
    '''
    
    def __init__(self, lower_bound):

        self.lower_bound = lower_bound

    def func(self, x):
        return jnp.log(x - self.lower_bound)

    def inv_func(self, y):
        return jnp.exp(y) + self.lower_bound

    def abs_jac(self, y):
        return jnp.exp(y)

class UnityTransform():
    '''
    Unity transform class to (not) transform a parameter that is already defined
    on an unbounded space.
    '''
    
    def __init__(self):
        pass
    
    def func(self, x):
        return x

    def inv_func(self, y):
        return y

    def abs_jac(self, y):
        return 1.

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

class TruncatedNormal(Normal):

    def __init__(self, mean, sigma, lower, upper):

        self.mean = mean
        self.sigma = sigma
        self.lower = lower
        self.upper = upper

    def log_prob(self, param):

        return super().log_prob(param)
        
    def sample(self, key):


        # Scale to be appropriate for mean-0 unit-variance gaussian.
        lower = (self.lower - self.mean) / self.sigma
        upper = (self.upper - self.mean) / self.sigma

        draw = jax.random.truncated_normal(key, lower, upper)
        return draw * self.sigma + self.mean

class HalfNormal():
    '''
    One-dimensional halfnormal distributions

    Parameters
    ----------
    sigma : float
        Standard deviation of distribution.
    '''

    def __init__(self, sigma, halfnormal=False):

        self.sigma = sigma

    @property
    def mean(self):
        return self.sigma * jnp.sqrt(2 / jnp.pi)


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
    # NOTE, these lmin, llmax parameters are not doing anything because
    # `bins` are provided. Ideally remove.
    num_modes_per_bin = binned_statistic(
        ells, num_modes, bins=bins, range=(lmin, lmax+1), statistic='sum')[0]

    return 1 / num_modes_per_bin

def get_coadd_transform_matrix(sels_to_coadd, ntri):
    '''

    Parameters
    ----------
    sels_to_coadd : (n_unique) list of index arrays.
        List of index arrays containing elements in data vector to coadd.
    ntri : 

    Returns
    -------
    mat : (n_unique, ntri) array

    '''

    n_unique = len(sels_to_coadd)

    mat = np.zeros((n_unique, ntri))

    for idx, selections in enumerate(sels_to_coadd):
        mat[idx,selections] = 1 / len(selections)

    return mat

def get_cov(signal_spectra, noise_spectra, bins, lmin, lmax, nsplit, nfreq,
            coadd_matrix=None):
    '''

    Parameters
    ----------
    signal_spectra : (nfreq, nfreq, nbin)
        Binned signal frequency cross spectra.
    noise_spectra : (nfreq, nfreq, nbin) array
        Binned noise frequency cross spectra.
    bins : (nbin + 1) array
        Bin edges.
    lmin : int

    lmax : int

    coadd_matrix : (n_unique, ntri)

    Returns
    -------
    cov : (ntri, ntri, nbin) array or (n_unique, n_unique, nbin) array
        Covariance matrix.
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

    if coadd_matrix is not None:
        cov = np.einsum('ai, ijk, jb -> abk', coadd_matrix, cov, coadd_matrix.T)
            
    cov *= prefactor

    return cov

def get_diff(data, model, tri_indices, coadd_matrix=None):
    '''


    Parameters
    ----------
    data : (ntri, nbin) or (n_unique, nbin)

    model : (nfreq, nfreq, nbin)

    tri_indices

    Returns
    -------
    diff : (ntri, nbin) or (n_uniqe, nbin) array
    '''

    if coadd_matrix is not None:

        model_data = jnp.zeros((len(tri_indices), data.shape[-1]))        
        for idx, (sidx1, fidx1, sidx2, fidx2) in enumerate(tri_indices):

            model_data = model_data.at[idx].set(model[fidx1,fidx2])

        model_data = jnp.einsum('ij, jk -> ik', coadd_matrix, model_data)
        diff = data - model_data
    else:

        diff = jnp.zeros_like(data)        
        for idx, (sidx1, fidx1, sidx2, fidx2) in enumerate(tri_indices):

            diff = diff.at[idx].set(data[idx] - model[fidx1,fidx2])

    return diff

def loglike(model, data, icov, tri_indices, coadd_matrix=None):
    '''

    '''

    diff = get_diff(data, model, tri_indices, coadd_matrix=coadd_matrix)

    return -0.5 * jnp.einsum('al, abl, bl', diff, icov, diff)
