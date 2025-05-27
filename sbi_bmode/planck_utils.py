import numpy as np

# From table 4 in https://www.cosmos.esa.int/documents/387566/387653/Planck_2018_results_L01.pdf.

# In arcmin.
planck_beam_fwhms = {'p30' : 32.29, 'p44' : 27.94, 'p70' : 13.08, 'p100' : 9.66, 'p143' : 7.22,
                     'p217' : 4.90, 'p353' : 4.92}

# In Hz.
planck_central_freqs = {'p30' : 28.4e9, 'p44' : 44.1e9, 'p70' : 70.4e9, 'p100' : 100e9, 'p143' : 143e9,
                        'p217' : 217e9, 'p353' : 353e9}

# Polarization sensitivity in uK degree.
planck_noise_level = {'p30' : 3.5, 'p44' : 4.0, 'p70' : 5.0, 'p100' : 1.96, 'p143' : 1.17,
                      'p217' : 1.75, 'p353' : 7.31}

def get_planck_noise(fstr, lmax):
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
    n_ell *= np.radians(planck_noise_level[fstr]) ** 2

    return n_ell    
