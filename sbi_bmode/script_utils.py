import os
import errno

import torch
from torch.distributions import Normal, HalfNormal

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
