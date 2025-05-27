import numpy as np

# From https://lambda.gsfc.nasa.gov/product/wmap/dr5/index.html.
# In arcmin.
wmap_beam_fwhms = {'K' : 52.8, 'Ka' : 39.6, 'Q' : 30.6, 'V' : 21., 'W' : 13.2}

# In Hz.
wmap_central_freqs = {'K' : 23.e9, 'Ka' : 33.e9, 'Q' : 41.e9, 'V' : 61.e9, 'W' : 94.e9}

# Polarization sensitivity in uK arcmin. Taken from mean noise power spectrum in 30 < ell < 300.
wmap_noise_level = {'K' : 290.6, 'Ka' : 294.8, 'Q' : 281.2, 'V' : 337.4, 'W' : 407.3}

def get_wmap_noise(fstr, lmax):
    '''
    Generate white noise curves in polarization for the planck satellite.

    Parameters
    ----------
    fstr : str
        Pick from 'p30', 'p44', 'p70', 'p100', 'p143', 'p217', 'p353'.
    lmax : int
        Maximum ell used for computation.

    Returns
    -------
    cov_noise_ell : (nell)
        Output noise spectra.
    '''

    n_ell = np.ones(lmax + 1)
    n_ell *= np.radians(planck_noise_level[fstr] / 60) ** 2

    return n_ell    
