import os

import numpy as np

def get_e_moped_matrix(data, params):
    '''
    Compute a compression matrix from a set of simulated datasets and
    corresponding parameters using the e-MOPED scheme from 2409.02102.
    
    Parameters
    ----------
    data : (nsims, ndata)
        Data vectors.
    params : (nsims, ntheta)
        Parameter vectors.
    
    Returns
    -------
    b_mat : (ntheta, ndata)
        Compression matrix
    '''

    nsims, ndata = data.shape
    ntheta = params.shape[-1]

    assert nsims == params.shape[0]
    
    # Get full covariance matrix.
    cov = np.cov(data.T, params.T)

    cov_data = cov[:ndata,:ndata]
    cov_data_params = cov[:ndata,ndata:]
    cov_params = cov[ndata:,ndata:]    

    params_mean = np.mean(params, axis=0)
    
    a_mat = np.dot(cov_data_params, np.linalg.inv(cov_params))
    b_vec = np.mean(data, axis=0) - np.dot(a_mat, params_mean)

    data_shifted = data - np.einsum('ij, kj -> ki', a_mat, (params - params_mean))    
    cov_data_shifted = np.cov(data_shifted.T)
    
    return get_moped_vectors(data_shifted, cov_data_shifted, a_mat)

def get_moped_vectors(data, cov_data, data_deriv):
    '''
    Compute a compression matrix from a set of simulated datasets and
    corresponding parameters using the MOPED scheme from 1707.06529.
        
    Parameters
    ----------
    data : (nsims, ndata)
        Data vectors.
    cov_data : (ndata, ndata)
        Covariance matrix of data vectors.
    data_deriv : (ndata, ntheta)
        Array with partial derivatives of data at fiducial parameters.

    Returns
    -------
    b_mat : (ntheta, ndata)
        Compression matrix.
    '''

    ndata, ntheta = data_deriv.shape
    b_mat = np.zeros((ntheta, ndata))
    
    icov_deriv = np.linalg.solve(cov_data, data_deriv[:,0])
    
    # Note that there is a typo in eq. 23 of 2409.02102, see eq. 3, 4 in 1707.06529.
    b_mat[0] = icov_deriv / np.sqrt(np.dot(icov_deriv, data_deriv[:,0]))
    
    for idx in range(1, ntheta):
        
        icov_deriv = np.linalg.solve(cov_data, data_deriv[:,idx])
        coefs = np.dot(b_mat[:idx], data_deriv[:,idx])
        
        b_mat[idx] = (icov_deriv - np.dot(b_mat[:idx,:].T, coefs))
        b_mat[idx] /= np.sqrt(np.dot(icov_deriv, data_deriv[:,idx]) - np.sum(coefs ** 2))

    return b_mat
