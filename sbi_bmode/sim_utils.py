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
        Dust amplitude.
    alpha_d_BB : float
        Dust spatial power law index.
    beta_dust : float
        Dust frequency power law index.
    freq_pivot_dust : float
        Pivot frequency for the frequency power law.
    temp_dust : float
        Dust temperature for the blackbody part of the model.
    r_tensor : float
        Tensor-to-scalar ratio.
    A_lens : float
        Amplitude of lensing contribution to BB.
    freqs : array-like
        Passband centers for the frquency channels of the instrument.
    seed : numpy.random._generator.Generator object or int
        Random number generator or seed for new random number generator.
    nsplit : int
        Number of splits of the data that have independent noise.
    cov_noise_ell : (nfreq, npol, npol, nell) array
        Noise covariance matrix.
    cov_scalar_ell : (npol, nell) array
        Signal covariance matrix with the EE and BB spectra from scalar perturbations.
    cov_tensor_ell : (npol, nell) array
        Signal covariance matrix with the EE and BB spectra from tensor perturbations.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    
    Returns
    -------
    data : (nsplit, nfreq, npol, npix)
        Simulated data.
    '''

    nfreq = len(freqs)
    out = np.zeros((nsplit, nfreq, 2, minfo.npix))

    # Spawn rng for dust and noise.
    seed = np.random.default_rng(seed)
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

    # Generate frequency-independent signal, scale with frequency later.
    cov_dust_ell[1,1] = spectra_utils.get_ell_shape(lmax, alpha_d_BB, ell_pivot=80)

    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    dust_alm = alm_utils.rand_alm(cov_dust_ell, ainfo, rng_dust, dtype=np.complex128)
    
    for fidx, freq in enumerate(freqs):
        dust_factor = spectra_utils.get_sed_dust(
            freq, beta_dust, temp_dust, freq_pivot_dust)
        g_factor = spectra_utils.get_g_fact(freq, temp_dust)
        
        signal_alm = cmb_alm.copy()
        signal_alm += dust_alm * np.sqrt(dust_factor) * g_factor * A_d_BB
                        
        for sidx in range(nsplit):
        
            data_alm = signal_alm + alm_utils.rand_alm(
                cov_noise_ell[fidx], ainfo, rngs_noise[sidx], dtype=np.complex128)
            data_alm = np.asarray(data_alm, dtype=np.complex128)
            sht.alm2map(data_alm, out[sidx,fidx], ainfo, minfo, 2)
            
    return out

def estimate_spectra(imap, minfo, ainfo):
    '''
    Compute all the auto and cross-spectra between splits and
    and frequency bands. NOTE Right now EE, EB are discarded.
    
    Parameters
    ----------
    imap : (nsplit, nfreq, 2, npix)
        Input maps.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    
    Returns
    -------
    out : (ntri, 2, lmax + 1)
        Output BB spectra. Only the elements of the upper-triangular part
        (+ the diagonal) of the (nsplits * nfreq) x (nsplits * nfreq) matrix
        are included.
    '''

    nsplit = imap.shape[0]
    nfreq = imap.shape[1]

    ntot = nsplit * nfreq
    # Number of elements in the upper triangle of the ntot x ntot matrix.
    ntri = ntot * (ntot + 1) // 2
    out = np.zeros((ntri, 1, ainfo.lmax + 1))    
    
    alm = np.zeros((nsplit, nfreq, 2, ainfo.nelem), dtype=np.complex128)
    sht.map2alm(imap, alm, minfo, ainfo, 2)    

    idxs = []
    for sidx in range(nsplit):
        for fidx in range(nfreq):
            idxs.append((sidx, fidx))
    
    idx = 0    
    for idx1 in range(ntot):
        for idx2 in range(idx1, ntot):

            sidx1, fidx1 = idxs[idx1]
            sidx2, fidx2 = idxs[idx2]            
            
            out[idx] = ainfo.alm2cl(
                alm[sidx1,fidx1,:,None,:], alm[sidx2,fidx2,None,:,:])[1,1]                    
            idx += 1
                        
    return out

def bin_spectrum(spec, ells, bins, lmin, lmax):
    '''
    Bin input spectra.
    
    Parameters
    ----------
    spec : (..., lmax + 1)
        Input spectra.
    ells :
        Multipole array corresponding to the spectra.
    bins : (nbin + 1) array
        Output bins. Specify the rightmost edge.
    lmin : int
        Do not use multipoles below lmin.
    lmax : int
        Do not use multipoles below lmax.    
    
    Returns
    -------
    spec_binned : (nbin) array
       Binned output. 
    '''

    return binned_statistic(ells, spec, bins=bins, range=(lmin, lmax+1))[0]

def get_final_data_vector(spec, bins, lmin, lmax):
    '''
    Create data vector by binning and flattening spectra.

    Parameters
    ----------
    spec : (..., lmax + 1)
        Input spectra.
    bins : (nbin + 1) array
        Output bins. Specify the rightmost edge.
    lmin : int
        Do not use multipoles below lmin.
    lmax : int
        Do not use multipoles below lmax.    
    
    Returns
    -------    
    out : (prod(...) * nbin) array
        Flattened and binned output array.        
    '''
    
    preshape = spec.shape[:-1]
    ells = np.arange(spec.shape[-1])
    out = np.zeros(preshape + (bins.size - 1,))

    for idxs in np.ndindex(preshape):

        out[idxs] = bin_spectrum(spec[idxs], ells, bins, lmin, lmax)

    return out.reshape(-1)
