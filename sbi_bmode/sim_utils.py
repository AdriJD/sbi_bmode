'''
Utils for simulating data using a simple Gaussian foreground model.
'''

import numpy as np
from scipy.stats import binned_statistic
from pixell import curvedsky
from optweight import alm_utils, sht

from sbi_bmode import spectra_utils

def gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
             r_tensor, A_lens, freqs, seed, nsplit, cov_noise_ell,
             cov_scalar_ell, cov_tensor_ell, minfo, ainfo):
    '''
    Generate simulated maps.
    
    Parameters
    ----------
    A_d_BB : float
    
    alpha_d_BB : float

    beta_dust : float

    freq_pivot_dust : float

    temp_dust : float

    r_tensor : float

    A_lens : float

    freqs : array-like

    seed : 

    nsplit : int

    cov_noise_ell : (nfreq, npol, npol, nell) array

    cov_scalar_ell : (npol, nell) array

    cov_tensor_ell : (npol, nell) array

    minfo : optweight.map_utils.MapInfo object

    ainfo : pixell.curvedsky.alm_info object
    
    Returns
    -------
    data : (nsplit, nfreq, npol, npix)
        Simulated data.
    '''

    nfreq = len(freqs)
    out = np.zeros((nsplit, nfreq, 2, minfo.npix))

    # Spawn rng for dust and noise.
    rngs = seed.spawn(2 + nsplit)
    rng_cmb = rngs[1]        
    rng_dust = rngs[1]    
    rngs_noise = rngs[2:]
    
    # Generate the CMB spectra.
    cov_ell = spectra_utils.get_combined_cmb_spectrum(
        r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell)
    cov_dust_ell = np.zeros_like(cov_ell)
    lmax = cov_ell.shape[-1] - 1
    assert ainfo.lmax == lmax

    # Generate frequency-independent signal, scale with freq later.
    cov_dust_ell[1,1] = spectra_utils.get_ell_shape(lmax, alpha_d_BB, ell_pivot=80)

    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    dust_alm = alm_utils.rand_alm(cov_dust_ell, ainfo, rng_dust, dtype=np.complex128)
    
    for fidx, freq in enumerate(freqs):
        dust_factor = spectra_utils.get_sed_dust(
            freq, beta_dust, temp_dust, freq_pivot_dust)
        g_factor = spectra_utils.get_g_fact(freq, temp_dust)
        
        signal_alm = cmb_alm.copy()
        signal_alm += dust_alm *  np.sqrt(dust_factor) * g_factor * A_d_BB
                        
        for sidx in range(nsplit):
        
            data_alm = signal_alm + alm_utils.rand_alm(
                cov_noise_ell, ainfo, rngs_noise[sidx], dtype=np.complex128)

            data_alm = np.asarray(data_alm, dtype=np.complex128)

            sht.alm2map(data_alm, out[sidx,fidx], ainfo, minfo, 2)
            
    return out

def estimate_spectra(imap, minfo, ainfo):
    '''

    Parameters
    ----------
    imap : (nsplit, nfreq, 2, npix)

    Returns
    -------
    out : (nsplit, nfreq, 2, lmax + 1)
    '''

    nsplits = imap.shape[0]
    nfreq = imap.shape[1]

    alm = np.zeros((nsplits, nfreq, 2, ainfo.nelem), dtype=np.complex128)

    out = np.zeros((nsplits, nfreq, nfreq, 1, ainfo.lmax + 1))

    sht.map2alm(imap, alm, minfo, ainfo, 2)    
        
    for sidx in range(nsplits):
        for fidx1 in range(nfreq):
            #for fidx2 in range(fidx1, nfreq):
            for fidx2 in range(0, nfreq):                
                out[sidx,fidx1,fidx2] = ainfo.alm2cl(
                    alm[sidx,fidx1,:,None,:], alm[sidx,fidx2,None,:,:])[1,1]
    return out

def bin_spectrum(spec, ells, bins, lmin, lmax):
    '''
    Bin input spectrum.
    
    Parameters
    ----------
    spec : (lmax + 1)

    bins : (nbin + 1) array
    
    Returns
    -------
    spec_binned : (nbin) array
    
    '''
    return binned_statistic(ells, spec, bins=bins, range=(lmin, lmax+1))[0]

def get_final_data_vector(spectra, bins, lmin, lmax):
    '''
    Create data vector by binning and flattening spectra.

    Parameters
    ----------
    spectra :

    Returns
    -------
    
    
    '''
    preshape = spectra.shape[:-1]
    ells = np.arange(spectra.shape[-1])
    out = np.zeros(preshape + (bins.size - 1,))

    for idxs in np.ndindex(preshape):

        out[idxs] = bin_spectrum(spectra[idxs], ells, bins, lmin, lmax)

    return out.reshape(-1)
