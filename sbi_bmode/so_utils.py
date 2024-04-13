import numpy as np

sat_beam_fwhms = {'f030' : 91., 'f040' : 63., 'f090' : 30., 'f150' : 17., 'f230' : 11., f'290' : 9.}

sat_central_freqs = {'f030' : 27., 'f040' : 39., 'f090' : 93., 'f150' : 145., 'f230' : 225., 'f290' : 280.}

sat_noise_level = {'f030' : {'baseline' : 21, 'goal' : 15},
                   'f040' : {'baseline' : 13, 'goal' : 10},
                   'f090' : {'baseline' : 3.4, 'goal' : 2.4},
                   'f150' : {'baseline' : 4.3, 'goal' : 2.7},
                   'f230' : {'baseline' : 8.6, 'goal' : 5.7},
                   'f290' : {'baseline' : 22, 'goal' : 14}}

sat_lknee = {'f030' : {'pessimistic' : 30, 'optimistic' : 15},
             'f040' : {'pessimistic' : 30, 'optimistic' : 15},
             'f090' : {'pessimistic' : 50, 'optimistic' : 25},
             'f150' : {'pessimistic' : 50, 'optimistic' : 25},
             'f230' : {'pessimistic' : 70, 'optimistic' : 35},
             'f290' : {'pessimistic' : 100, 'optimistic' : 40}}
 
sat_noise_alpha = {'f030' : -2.4,
                   'f040' : -2.4,
                   'f090' : -2.5,
                   'f150' : -3,
                   'f230' : -3,
                   'f290' : -3}

def get_ntube(fstr, nyear_lf=1):
    '''
    Return the "ntube" parameter for a given frequency band.

    Parameters
    ----------
    fstr : str
        Pick from 'f030', 'f040', 'f090', 'f150', 'f230' : 'f290'.
    nyear_lf : float
        Number of years where an lF is deployed.

    Returns
    -------
    ntube : float
        Effective number of tubes. Used to scale noise amplitudes.
    '''

    if fstr in ('f230', 'f290'):
        return 1.
    elif fstr in ('f090', 'f150'):
        return (2 - nyear_lf / 5) / 2
    else:
        return nyear_lf / 5

def get_sat_noise(fstr, sensitivity_mode, lknee_mode, fsky, lmax, nyear_lf=1,
                  include_kludge=True, nyear=5):
    '''
    Generate noise curves in polarization for the SO small aperture telescopes

    Parameters
    ----------
    fstr : str
        Pick from 'f030', 'f040', 'f090', 'f150', 'f230' : 'f290'.
    sensitivity_mode : str
         Pick from "baseline" or "goal".
    lknee_mode : str
         Pick from "pessimistic" or "optimistic".
    fsky : float
        Fraction of sky, 0 <= fsky <= 1.
    lmax : int
        Maximum ell used for computation.
    nyear_lf : int
         Number of years where an LF is deployed on SAT.    
    include_kludge : bool, optional
        Include a fudge factor that models the noise-uniformity at edge of map.
    nyear : float, optional
        Number of years of observations.

    Returns
    -------
    cov_noise_ell : (nell)
        Output noise spectra.
    '''
    
    ntube = get_ntube(fstr, nyear_lf)
    noise_level = sat_noise_level[fstr][sensitivity_mode] / np.sqrt(ntube)
    lknee = sat_lknee[fstr][lknee_mode]
    alpha = sat_noise_alpha[fstr]

    tobs = nyear * 365 * 24 * 3600
    # Retention after observing efficiency and cuts + kludge for the noise non-uniformity
    # of the map edges
    tobs *= 0.2  
    if include_kludge:
        tobs *= 0.85 
    obs_area = 4 * np.pi * fsky
    
    tot_noise_level = noise_level / np.sqrt(tobs)
    
    ells = np.arange(lmax + 1)
    n_ell = np.zeros(ells.size)
    
    n_ell[2:] = (ells[2:] / lknee) ** alpha + 1.
    n_ell[2:] *= (tot_noise_level * np.sqrt(2)) ** 2 * obs_area

    return n_ell
