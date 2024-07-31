
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
