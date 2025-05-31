'''
Utilities to generate simple Gaussian CMB + FG spectra. Mostly jax.
'''
import os
import argparse as ap
import yaml

from jax import jit, vmap
import jax.numpy as jnp
import ducc0
import numpy as np
import healpy as hp
from scipy.stats import binned_statistic
from pixell import curvedsky
from optweight import map_utils, sht, alm_utils

import jax
jax.config.update("jax_enable_x64", True)

# CMB temperature in Kelvin.
cmb_temp = 2.726

# Planck constant in J / Hz.
hplanck = 6.62607015e-34

# Boltzmann constant in J / K.
kboltz = 1.380649e-23

# Speed of light in m / s.
clight = 299792458.

# These two below help to avoid under/overflow with 32bit.
# 2 * kb^3 1^2 / (h c) ** 2.
b1 = 1.3339095411668483e-19

# h / kB.
hk_ratio = 4.799243073366221e-11

def get_planck_law(freq, temp):
    '''
    Return the Planck law for input frequencies and temperature.

    Parameters
    ----------
    freq : (nfreq) array
        Input frequencies in Hz.
    temp : float
        Input temperature in Kelvin.

    Returns
    -------
    b_nu : (nfreq) array
        B_nu(nu, T) for each frequency.
    '''

    b0 = b1 * temp ** 2
    xx = hk_ratio * (freq / temp)

    return b0 * temp * xx ** 3 / jnp.expm1(xx)

def get_g_fact(freq, temp=cmb_temp):
    '''
    Compute the conversion factor between RJ and CMB temperature.
    See Eq 14 in Choi et al. (2007.07289).
    
    Parameters
    ----------
    freq : (nfreq) array
        Input frequencies in Hz.
    temp : float, optional
        Input temperature in Kelvin.

    Returns
    -------
    g_fact : (nfreq) array
        Conversion from RJ to CMB.
    '''

    xx = hk_ratio * (freq / temp)

    return (jnp.expm1(xx) ** 2) / (xx ** 2 * jnp.exp(xx))

def get_cmb_spectra(spectra_filepath, lmax):
    '''    
    Return the EE and BB spectra from a CAMB text file.

    Parameters
    ----------
    spectra_filepath : str
        Path to CAMB .dat file.
    lmax : int
        Truncate spectra to this lmax.
    
    Returns
    -------
    out : (npol, npol, nell) array
        Spectra in Cl.
    '''

    ells, dtt_ell, dee_ell, dbb_ell, dte_ell = np.loadtxt(
        spectra_filepath, unpack=True)
    
    lmin = int(ells[0])
    dells = ells * (ells + 1) / 2 / np.pi

    out = np.zeros((2, 2, lmax + 1))
        
    out[0,0,lmin:] = (dee_ell / dells)[:lmax+1-lmin]
    out[1,1,lmin:] = (dbb_ell / dells)[:lmax+1-lmin]

    return out

def get_combined_cmb_spectrum(r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell):
    '''
    Given r and A_lens, combine scalar and tensor constributions.

    Parameters
    ----------
    r_tensor : float
        Tensor-to-scalar ratio.
    A_lens : float
        A_lens parameter.
    cov_scalar_ell : (npol, npol, nell) array
        Scalar cls.
    cov_tensor_ell : (npol, npol, nell) array
        Tensor cls.

    Returns
    -------
    cov_tot_ell : (npol, npol, nell) array
        Combined spectrum.
    
    Notes
    -----
    This is an approximation. In reality, a nonzero r will also change the shape
    of the TT, TE, EE spectra. 
    '''

    return A_lens * cov_scalar_ell + r_tensor * cov_tensor_ell

def get_sed_dust(freq, beta, temp, freq_pivot):
    '''
    Compute the SED of the Planck modified blackbody dust model in RJ temp,
    see Eq. 15 in Choi et al. This is the square of what to apply to a map!
    
    Parameters
    ----------
    freq : float
        Effective freq of passband in Hz.    
    beta : float
        Frequency power law index.
    temp : float
        Dust temperature.
    freq_pivot : float
        Pivot frequency in Hz.

    Returns
    -------
    out : float
        SED^2 evaluated at input freq.
    '''

    b_freq = get_planck_law(freq, temp)
    b_pivot = get_planck_law(freq_pivot, temp)
    
    return ((freq / freq_pivot) ** (beta - 2) * b_freq / b_pivot) ** 2

def get_sed_sync(freq, beta, freq_pivot):
    '''
    The frequency dependent part of Eq. 16 in Choi et al.,
    without the g1 and a_sync factors. This is in RJ temperature units and
    this is the square of what to apply to a map.
    
    Parameters
    ----------
    freq : float
        Effective freq of passband in Hz.    
    beta : float
        Frequency power law index.
    freq_pivot : float
        Pivot frequency in Hz.

    Returns
    -------
    out : float
        SED^2 evaluated at input freq.
    '''

    return (freq / freq_pivot) ** (2 * beta)

def get_ell_shape(lmax, alpha, ell_pivot=80):
    '''
    Get the ell-dependent part of the template.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    alpha : float
        Power law index.
    ell_pivot : int, optional
        Pivot multipole.

    Returns
    -------
    out : (nell) array
        Power law.
    '''

    out = jnp.zeros((lmax + 1))
    ells = jnp.arange(2, lmax + 1)
    dells = ells * (ells + 1) / 2 / np.pi

    out = out.at[2:].set((ells / ell_pivot) ** (alpha) / dells)

    return out

def bin_spectrum(spec, ells, bins, lmin, lmax, use_jax=False):
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

    preshape = spec.shape[:-1]
    
    if use_jax:
        out = jnp.zeros(preshape + (bins.size - 1,))    
    else:        
        out = np.zeros(preshape + (bins.size - 1,))
        
    for idxs in np.ndindex(preshape):

        if use_jax:
            out = out.at[idxs].set(
                jnp.histogram(ells, bins=bins, range=(lmin, lmax+1), weights=spec[idxs])[0] /
                jnp.histogram(ells, bins=bins, range=(lmin, lmax+1))[0])
        else:
            out[idxs] = binned_statistic(ells, spec[idxs], bins=bins, range=(lmin, lmax+1))[0]
        
    return out

def get_dust_spectra(amp, alpha, lmax, freqs, beta, temp, freq_pivot):
    '''
    Compute dust cross-spectra in CMB temperature.
    
    Parameters
    ----------
    amp : float
        Amplitude
    alpha : float
        Multipole power law index.
    lmax : int
        Maximum multipole.
    freqs : (nfreq) array
        Effective frequencies of bands in Hz.
    beta : float
        Dust frequency spectral index.
    temp : float
        Dust temperature.
    freq_pivot : float
        Frequency pivot scale in Hz.
    
    Returns
    -------
    out : (nfreq, nfreq, lmax + 1) array
        Cross spectra.
    '''
    
    nfreq = len(freqs)
    
    out = jnp.zeros((nfreq, nfreq, lmax+1))
    out = out.at[:].set(get_ell_shape(lmax, alpha)[jnp.newaxis,:])

    g0_factor =  get_g_fact(freq_pivot)
    
    for fidx1 in range(nfreq):
        
        f1_factor = get_sed_dust(freqs[fidx1], beta, temp, freq_pivot)
        g1_factor = get_g_fact(freqs[fidx1])
        
        for fidx2 in range(nfreq):
        
            f2_factor = get_sed_dust(freqs[fidx2], beta, temp, freq_pivot)
            g2_factor = get_g_fact(freqs[fidx2])

            out = out.at[fidx1,fidx2].multiply(amp * jnp.sqrt(f1_factor * f2_factor) \
                                               * g1_factor * g2_factor / (g0_factor ** 2))
            
    return out

def apply_beam_to_freq_cov(cov_ell, beams):
    '''
    Apply per-frequency beams to a nfreq x nfreq covariance matrix.

    Parameters
    ----------
    cov_ell : (nfreq, nfreq, lmax + 1) array
        Covariance matrix.
    beams : (nfreq, lmax + 1) array
        Beams per frequency.
    '''

    nfreq = beams.shape[0]
    beam_mat = jnp.eye(nfreq)[:,:,jnp.newaxis] * beams

    return jnp.einsum('abl, bcl, cdl -> adl', beam_mat, cov_ell, beam_mat)

def get_sync_spectra(amp, alpha, lmax, freq, beta, freq_pivot):
    '''

    Parameters
    ----------
    amp : float
        Amplitude
    alpha : float
        Mulitpole power law index.
    freq : float
        Effective frequency of band.
    beta : 
    freq_pivot : float
        Frequency pivot scale.
            
    Returns
    -------
    out : (nell) array
    '''

    f_factor = get_sed_sync(freq, beta, freq_pivot)
    
    #out = jnp.zeros((lmax + 1))
    #ells = jnp.arange(2, lmax + 1)

    prefactor = amp * f_factor

    out = get_ell_shape(lmax, alpha) * prefactor
    
    return out

def get_fg_spectra(A_d_EE, alpha_d_EE, alpha_d_BB, beta_dust, nu0_dust, temp_dust,
                   A_s_EE, alpha_s_EE, alpha_s_BB, beta_sync, nu0_sync,
                   freq, lmax):
    '''
    Compute dust and synchrotron Cls. 

    Returns
    -------
    out : (npol, npol, nell)
        Sum of dust and synchrotron spectra.
    '''
    
    #out = jnp.zeros((2, 2, lmax + 1))
    out = np.zeros((2, 2, lmax + 1))    

    g_factor = get_g_fact(freq, cmb_temp)

    out = out.at[0,0].set(get_dust_spectra(A_d_EE, alpha_d_EE, freq, beta_dust,
                                           temp_dust, nu0_dust, g_fact, lmax))
    out = out.at[1,1].set(get_dust_spectra(A_d_BB, alpha_d_BB, freq, beta_dust,
                                           temp_dust, nu0_dust, g_fact, lmax))

    out = out.at[0,0].add(get_sync_spectra(A_s_EE, alpha_s_EE, freq, beta_sync,
                                           nu0_sync, g_fact, lmax))
    out = out.at[1,1].add(get_sync_spectra(A_s_BB, alpha_s_BB, freq, beta_sync,
                                           nu0_sync, g_fact, lmax))
    out = out.at[:].multiply(g_factor ** 2)
    
    return out
