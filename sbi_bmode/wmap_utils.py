import numpy as np

# From https://lambda.gsfc.nasa.gov/product/wmap/dr5/index.html.
# In arcmin.
# NOTE, table actually gives sqrt(Omega) = sqrt(2 * pi * sigma^2) = 2.506 sigma, so not quite FHWM..
wmap_beam_fwhms = {'wK' : 52.8, 'wKa' : 39.6, 'wQ' : 30.6, 'wV' : 21., 'wW' : 13.2}

# In Hz.
#wmap_central_freqs = {'wK' : 23.e9, 'wKa' : 33.e9, 'wQ' : 41.e9, 'wV' : 61.e9, 'wW' : 94.e9}
# Changed K band to avoid divide by zero in pyilc.
wmap_central_freqs = {'wK' : 25.e9, 'wKa' : 33.e9, 'wQ' : 41.e9, 'wV' : 61.e9, 'wW' : 94.e9} 

# Polarization sensitivity in uK arcmin. Taken from mean noise power spectrum in 30 < ell < 300.
wmap_noise_level = {'wK' : 290.6, 'wKa' : 294.8, 'wQ' : 281.2, 'wV' : 337.4, 'wW' : 407.3}

def get_wmap_noise(fstr, lmax):
    '''
    Generate white noise curves in polarization for the WMAP satellite.

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
    n_ell *= np.radians(wmap_noise_level[fstr] / 60) ** 2

    return n_ell    
