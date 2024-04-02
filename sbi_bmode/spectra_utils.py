'''
Utilities to generate simple Gaussian CMB + FG spectra.
'''
import os
import argparse as ap

#from jax import jit, vmap
#import jax.numpy as jnp
import ducc0
import numpy as np
import healpy as hp
import yaml
#import pysm
#from pysm.nominal import models
from pixell import curvedsky
from optweight import map_utils, sht, alm_utils

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

    #return b0 * temp * xx ** 3 / jnp.expm1(xx)
    return b0 * temp * xx ** 3 / np.expm1(xx)

def get_g_fact(freq, temp):
    '''
    Compute the conversion factor between antenna and CMB temperature.
    See Eq 14 in Choi et al.
    
    Parameters
    ----------
    freq : (nfreq) array
        Input frequencies in Hz.
    temp : float
        Input temperature in Kelvin.

    Returns
    -------
     : (nfreq) array

    '''

    xx = hk_ratio * (freq / temp)

    #return (jnp.expm1(xx) ** 2) / (xx ** 2 * jnp.exp(xx))
    return (np.expm1(xx) ** 2) / (xx ** 2 * np.exp(xx))

def get_cmb_spectra(spectra_filepath, lmax):
    '''    
    Return the EE and BB spectra from a CAMB text file.
    
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

    '''

    return A_lens * cov_scalar_ell + r_tensor * cov_tensor_ell

def get_sed_dust(freq, beta, temp, freq_pivot):
    '''
    Compute the SED of the Planck modified blackbody dust model in antenna temp,
    see Eq. 15 in Choi et al.
    
    Parameters
    ----------
    freq : float
        Effective freq of passband in Hz.    
    beta : float
    
    temp : float

    freq_pivot : float

    Returns
    -------
    out : float
        SED evaluated at input freq.
    '''

    b_freq = get_planck_law(freq, temp)
    b_pivot = get_planck_law(freq_pivot, temp)
    
    return ((freq / freq_pivot) ** (beta - 2) * b_freq / b_pivot) ** 2

def get_ell_shape(lmax, alpha, ell_pivot=80):
    '''
    Get the ell-dependent part of the template.
    '''

    #out = jnp.zeros((lmax + 1))
    #ells = jnp.arange(2, lmax + 1)
    
    #out = out.at[2:].set((ells / ell_pivot) ** (alpha + 2))

    out = np.zeros((lmax + 1))
    ells = np.arange(2, lmax + 1)
    
    out[2:] = (ells / ell_pivot) ** (alpha + 2)

    return out
    
def get_sed_sync(freq, beta, freq_pivot):
    '''
    The frequency dependent part of Eq. 16 in Choi et al.,
    without the g1 and a_sync factors.
    
    Parameters
    ----------
    freq : float

    beta : float

    freq_pivot : float
    '''

    return (freq / freq_pivot) ** (2 * beta)

def get_dust_spectra(amp, alpha, lmax, freq, beta, temp, freq_pivot):
    '''
    Compute dust spectrum in antenna temperature.
    
    Parameters
    ----------
    amp : float
        Amplitude
    alpha : float
        Multipole power law index.
    lmax : int
        Max multipole.
    freq : float
        Effective frequency of band.
    beta : float
    temp : float
        Dust temperature
    freq_pivot : float
        Frequency pivot scale.
    
    Returns
    -------
    out : (nell) array
    '''
    
    f_factor = get_sed_dust(freq, beta, temp, freq_pivot)

    #out = jnp.zeros((lmax + 1))
    #ells = jnp.arange(2, lmax + 1)

    prefactor = amp * f_factor

    #ell_pivot = 80    
    #out = out.at[2:].set(prefactor * (ells / ell_pivot) ** (alpha + 2))
    out = get_ell_shape(lmax, alpha) * prefactor
    
    return out

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

    #ell_pivot = 80
    #out = out.at[2:].set(prefactor * (ells / ell_pivot) ** (alpha + 2))
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

def get_combined_spectrum(params, cov_scalar_ell, cov_tensor_ell):
    '''
    
    '''

    lmax = cov_scalar_ell.shape[-1] - 1

    #c_ell = get_combined_

    pass
