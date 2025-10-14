import os
import errno

import numpy as np
import torch
from torch.distributions import Normal, HalfNormal
from mpi4py.util import dtlib

from sbi_bmode import custom_distributions

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
        Dictionary with data generation parameters.
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
        elif prior_dict['prior_type'].lower() == 'truncatednormal':
            prior.append(custom_distributions.TruncatedNormal(*prior_dict['prior_params']))
        else:
            raise ValueError(f"{prior_dict['prior_type']=} not understood")
        param_names.append(param)

    # sbi needs the distributions to not be scalar.
    prior_list = [p.expand(torch.Size([1])) for p in prior]
    
    return prior_list, param_names

def get_param_limits(prior_list, param_names):
    '''
    Extract prior limits.

    Parameters
    ----------
    prior_list : list of torch.distribition instances
        Priors.
    param_names : list of str
        Parameter names.

    Returns
    -------
    param_limits : dict
        Dictionary with (lower, upper) tuple per parameter.
    '''

    param_limits = {}
    for dist, name in zip(prior_list, param_names):
        
        support = getattr(dist, 'support', None)
        
        if support is not None:
            lower = getattr(support, 'lower_bound', None)
            upper = getattr(support, 'upper_bound', None)

            if isinstance(lower, torch.Tensor):
                lower = float(lower.item())
            if isinstance(upper, torch.Tensor):
                upper = float(upper.item())

            param_limits[name] = (lower, upper)
        else:
            param_limits[name] = (None, None)

    return param_limits

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

def symlink_force(target, link_name):
    '''
    Create a symlink, overwrite existing link if present.

    Parameters
    ----------
    target : str
        Path to target.
    link_name : str
        Path to symlink.    
    '''
    
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def gatherv_array(array_on_rank, comm, root=0):
    '''
    Gather unequal-sized numpy arrays on root rank using MPI.

    Parameters
    ----------
    array_on_rank : (num_on_rank, ...) array
        Array on each rank to be gathererd to root.
    comm : mpi4py.MPI.Comm object
        MPI communicator.
    root : int, optional
        The index of the root rank.

    Returns
    -------
    array_full : (num_total, ...) array, None
        Full array on root rank, None on others.
    '''

    num_per_rank = comm.allgather(array_on_rank.shape[0])
    num_per_rank = np.asarray(num_per_rank, dtype=np.int64)

    dtype_per_rank = comm.allgather(array_on_rank.dtype)
    if comm.rank == root:
        assert all(x == dtype_per_rank[0] for x in dtype_per_rank)
    dtype = dtype_per_rank[0]

    postshape_per_rank = comm.allgather(array_on_rank.shape[1:])
    if comm.rank == root:        
        assert all(x == postshape_per_rank[0] for x in postshape_per_rank)
    postshape_size = np.prod(np.asarray(postshape_per_rank[0], dtype=np.int64))
    
    if comm.rank == root:
        num_total = np.sum(num_per_rank)
        array_full = np.zeros(num_total * postshape_size, dtype=dtype)
    else:
        array_full = None
        
    offsets = np.zeros(comm.size)    
    offsets[1:] = np.cumsum(num_per_rank * postshape_size)[:-1]
    comm.Gatherv(
        sendbuf=array_on_rank,
        recvbuf=(array_full, np.array(num_per_rank * postshape_size, dtype=int),
                 np.array(offsets, dtype=int), dtlib.from_numpy_dtype(dtype)),
        root=root)

    if comm.rank == root:
        array_full = array_full.reshape(num_total, *postshape_per_rank[0])

    return array_full

def preprocess_n_train(n_train, n_rounds):
    '''
    Return a list with number of training samples for each round.

    Parameters
    ----------
    n_train : int, array-like
        Number of simulations to create, potentially specified for each round.
    n_rounds : int
        Number of SNPE rounds.

    Returns
    -------
    n_train : (n_rounds) list of ints
        Number of simulations specified for each round.
    '''

    if np.isscalar(n_train):
        n_train = [n_train] * n_rounds
    else:
        if n_rounds > 0:
            assert len(n_train) == n_rounds
        n_train = list(n_train)

    return n_train
    
def str_to_slice(string):
    '''
    Convert a string to a 1D slice, e.g. ':10' into slice(None,10,None).

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    sel : slice
        Output slice.
    '''
    
    string = string.strip()
    
    if string.isdigit() or (string.startswith('-') and string[1:].isdigit()):        
        # Simple integer case.
        return [int(string)]
    
    elif ':' in string:
        # Slice case.
        parts = [p.strip() or None for p in string.split(':')]
        parts = [int(p) if p is not None else None for p in parts]
        return slice(*parts)
    
    else:
        raise ValueError(f"Invalid slice string: '{s}'")
