import os
import argparse as ap

from jax import jit, vmap
import jax.numpy as jnp
import ducc0
import numpy as np
import healpy as hp
import yaml
import pysm
from pysm.nominal import models
from pixell import curvedsky
from optweight import map_utils, sht, alm_utils

opj = os.path.join

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

    b0 = cs.b1 * temp ** 2
    xx = cs.hk_ratio * (freq / temp)

    return b0 * temp * xx ** 3 / jnp.expm1(xx)

def get_g_fact(freq, temp):
    '''

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

    xx = cs.hk_ratio * (freq / temp)

    return xx ** 2 * jnp.exp(xx) / (jnp.expm1(xx) ** 2)

def get_cmb_spectra(spectra_filepath, lmax):
    '''

    Returns
    -------
    (npol, npol, nell)
    '''

    ells, dtt_ell, dee_ell, dbb_ell, dte_ell = np.loadtxt(spectra_filepath, unpack=True)

    dells = ells * (ells + 1) / 2 / np.pi

    out = np.zeros((2, 2, lmax + 1))
    out[0,0,1:] = dee_ell / dells
    out[1,1,1:] = dbb_ell / dells

    return out

def get_combined_cmb_spectrum(r_tensor, cov_scalar_ell, cov_tensor_ell):
    '''

    '''

    return cov_scalar_ell + r_tensor * cov_tensor_ell

def get_dust_conv_factor(freq, beta_dust, temp_dust, freq_pivot):
    '''
    Eq. 15 in Choi et al.
    
    Parameters
    ----------
    freq : float

    beta_dust : float

    temp_dust : float

    freq_pivot : float
    '''

    b_freq = get_planck_law(freq, temp_dust)
    b_ref = get_planck_law(freq_pivot, temp_dust)
    
    return ((freq / freq_pivot) ** (beta_dust - 2) * b_freq / b_pivot) ** 2

def get_sync_conv_factor(freq, beta_sync, freq_pivot):
    '''
    Eq. 16 in Choi et al. without g1 and a_sync factor.
    
    Parameters
    ----------
    freq : float

    beta_dust : float

    freq_pivot : float
    '''

    return (freq / freq_pivot) ** (2 * beta_sync)

#def get_dust_spectra(amp, alpha, freq, beta, temp, freq_pivot, g_fact, lmax):
def get_dust_spectra(amp, alpha, lmax):
    '''

    Parameters
    ----------
    amp : float
        Amplitude
    alpha : float
        Mulitpole power law index.
    freq : float
        Effective frequency of band.
    temp : float
        Dust temperature
    freq_pivot : float
        Frequency pivot scale.
    g_fact : float
        
    
    Returns
    -------
    out : (nell) array
    '''
    
    #f_factor = get_dust_conv_factor(freq, beta, temp, freq_pivot)

    out = jnp.zeros((lmax + 1))
    ells = jnp.arange(2, lmax + 1)

    prefactor = amp # * f_factor * g_factor ** 2

    ell_pivot = 80    
    out = out.at[2:].set(prefactor * (ells / ell_pivot) ** alpha)
    
    return out

#def get_sync_spectra(amp, alpha, freq, beta, freq_pivot, g_fact, lmax):
def get_sync_spectra(amp, alpha, lmax):
    '''

    Parameters
    ----------
    amp : float
        Amplitude
    alpha : float
        Mulitpole power law index.
    freq : float
        Effective frequency of band.
    temp : float
        Dust temperature
    freq_pivot : float
        Frequency pivot scale.
    g_fact : float
        
    
    Returns
    -------
    out : (nell) array
    '''
    
    out = jnp.zeros((lmax + 1))
    ells = jnp.arange(2, lmax + 1)

    prefactor = amp # * f_factor * g_factor ** 2

    ell_pivot = 80
    out = out.at[2:].set(prefactor * (ells / ell_pivot) ** alpha)
    
    return out

def get_fg_spectra(A_d_EE, A_d_EE, alpha_d_EE, alpha_d_BB, beta_dust, nu0_dust, temp_dust,
                   A_s_EE, A_s_EE, alpha_s_EE, alpha_s_BB, beta_sync, nu0_sync,
                   freq, lmax):
    '''
    Compute dust and synchrotron Cls. Without frequency scaling.

    Returns
    -------
    out : (npol, npol, nell)
        Sum of dust and synchrotron spectra.
    '''
    
    out = jnp.zeros((2, 2, lmax + 1))
    ells = jnp.arange(2, lmax + 1)

    g_factor = get_g_fact(freq, cmb_temp)

    out = out.at[0,0].set(get_dust_spectra(A_d_EE, alpha_d_EE, freq, beta_dust,
                                           temp_dust, nu0_dust, g_fact, lmax))
    out = out.at[1,1].set(get_dust_spectra(A_d_BB, alpha_d_BB, freq, beta_dust,
                                           temp_dust, nu0_dust, g_fact, lmax))

    out = out.at[0,0].add(get_sync_spectra(A_s_EE, alpha_s_EE, freq, beta_sync,
                                           nu0_sync, g_fact, lmax))
    out = out.at[1,1].add(get_sync_spectra(A_s_BB, alpha_s_BB, freq, beta_sync,
                                           nu0_sync, g_fact, lmax))
    
    return out

def get_combined_spectrum(params, cov_scalar_ell, cov_tensor_ell):
    '''
    
    '''

    lmax = cov_scalar_ell.shape[-1] - 1

    #c_ell = get_combined_

    pass

def get_noise_cov_ell():
    '''

    Returns
    -------
    cov_noise_ell : (nfreq, npol, npol, nell)
    '''
    
    pass

def draw_alm():
    pass

def draw_map():
    pass

def gen_data(A_d_EE, A_d_EE, alpha_d_EE, alpha_d_BB, beta_dust, nu0_dust, temp_dust,
             A_s_EE, A_s_EE, alpha_s_EE, alpha_s_BB, beta_sync, nu0_sync,
             r_tensor,
             freqs,
             seed,
             nsplit,
             noise_cov_ell, minfo):
    '''

    Parameters
    ----------
    noise_cov_ell : (nfreq, npol, npol, nell)

    
    Returns
    -------
    data : (nsplit, nfreq, npol, npix)
    '''

    nfreqs = freq.shape
    out = np.zeros((nsplit, nfreq, 2, minfo.npix))

    rngs = seed.spawn(2 + nspits)
    rng_dust, rng_sync = rngs[:2]
    rngs_noise = rngs[2:]
    
    # Generate the CMB spectra.
    cov_ell = get_combined_cmb_spectrum(r_tensor, cov_scalar_ell, cov_tensor_ell)
    lmax = cov_ell.shape[-1]
    ainfo = curvedsky.alm_info(lmax)

    # Generate frequency-independent signal, scale with freq later.
    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, seed, dtype=np.complex128)
    dust_alm = alm_utils.rand_alm(cov_dust_ell, ainfo, rng_dust, dtype=np.complex128)
    sync_alm = alm_utils.rand_alm(cov_sync_ell, ainfo, rng_sync, dtype=np.complex128)    
    
    # Loop over freqs
    for fidx, freq in enumerate(freqs):
            
        dust_factor = get_dust_conv_factor(freq, beta_dust, temp_dust, nu0_dust)
        sync_factor = get_dust_conv_factor(freq, beta_dust, temp_dust, nu0_dust)        
        
        signal_alm = cmb_alm.copy()
        signal_alm += dust_alm *  np.sqrt(dust_factor) * g_factor
        signal_alm += sync_alm *  np.sqrt(sync_factor) * g_factor         
                        
        for sidx in range(nsplit):
        
            # Add noise.
            data_alm = signal_alm + alm_utils.rand_alm(cov_dust_ell, ainfo, rng_dust, dtype=np.complex128)
            sht.alm2map(data_alm, out[sidx, fidx], ainfo, minfo, 2)

    return out

def estimate_spectra(data):
    '''

    '''

    # loop over splits

    # loop over freqs

    # map2alm into (nsplit, nfreq, nelem) array

    # Outer loop over nsplit, nfreq, nelem

    # 
    
    pass










def get_mean_spectra(lmax, mean_params, foregrounds):
    """ Computes amplitude power spectra for all components
    """
    ells = np.arange(lmax+1)
    dl2cl = np.ones(len(ells))
    dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
    cl2dl = (ells*(ells+1.))/(2*np.pi)

    # CMB amplitude
    # Lensing
    l, dtt, dee, dbb, dte=np.loadtxt("data/camb_lens_nobb.dat",unpack=True)
    l = l.astype(int)
    msk = l <= lmax
    l = l[msk]
    dltt = np.zeros(len(ells)); dltt[l]=dtt[msk]
    dlee = np.zeros(len(ells)); dlee[l]=dee[msk]
    dlbb = np.zeros(len(ells)); dlbb[l]=dbb[msk]
    dlte = np.zeros(len(ells)); dlte[l]=dte[msk]

    cl_cmb_bb_lens=dlbb * dl2cl
    cl_cmb_ee_lens=dlee * dl2cl

    # Lensing + r=1
    l,dtt,dee,dbb,dte=np.loadtxt("data/camb_lens_r1.dat",unpack=True)
    l = l.sastype(int)
    msk = l <= lmax
    l = l[msk]
    dltt = np.zeros(len(ells)); dltt[l]=dtt[msk]
    dlee = np.zeros(len(ells)); dlee[l]=dee[msk]
    dlbb = np.zeros(len(ells)); dlbb[l]=dbb[msk]
    dlte = np.zeros(len(ells)); dlte[l]=dte[msk]

    cl_cmb_bb_r1=dlbb * dl2cl
    cl_cmb_ee_r1=dlee * dl2cl

    cl_cmb_ee = cl_cmb_ee_lens + mean_params['r_tensor'] * (cl_cmb_ee_r1-cl_cmb_ee_lens)
    cl_cmb_bb = cl_cmb_bb_lens + mean_params['r_tensor'] * (cl_cmb_bb_r1-cl_cmb_bb_lens)

    # SEPERATE FUNCTION

    if foregrounds is True:
        # Dust
        A_dust_BB = mean_params['A_d_BB'] * fcmb(mean_params['nu0_dust'])**2
        A_dust_EE = mean_params['A_d_EE'] * fcmb(mean_params['nu0_dust'])**2
        dl_dust_bb = A_dust_BB * ((ells+1E-5) / 80.)**mean_params['alpha_d_BB']
        dl_dust_ee = A_dust_EE * ((ells+1E-5) / 80.)**mean_params['alpha_d_EE']
        cl_dust_bb = dl_dust_bb * dl2cl
        cl_dust_ee = dl_dust_ee * dl2cl

        # Sync
        A_sync_BB = mean_params['A_s_BB'] * fcmb(mean_params['nu0_sync'])**2
        A_sync_EE = mean_params['A_s_EE'] * fcmb(mean_params['nu0_sync'])**2
        dl_sync_bb = A_sync_BB * ((ells+1E-5) / 80.)**mean_params['alpha_s_BB']
        dl_sync_ee = A_sync_EE * ((ells+1E-5) / 80.)**mean_params['alpha_s_EE']
        cl_sync_bb = dl_sync_bb * dl2cl
        cl_sync_ee = dl_sync_ee * dl2cl

    # RETURN A DICT

        return(ells, dl2cl, cl2dl, cl_dust_bb, cl_dust_ee, cl_sync_bb,
               cl_sync_ee, cl_cmb_bb, cl_cmb_ee)

    else:
        return(ells, dl2cl, cl2dl, cl_cmb_bb, cl_cmb_ee)

def fcmb(nu):
    """ CMB SED (in antenna temperature units).
    """
    x=0.017608676067552197*nu
    ex=np.exp(x)
    return ex*(x/(ex-1))**2



# NOTE REPLACE
#
def create_noise_splits(freqs, nside, add_mask=False, sens=1, knee=1, ylf=1,
                        fsky=0.1, nsplits=4):
    """ Generate instrumental noise realizations.
    Args:
        nside: HEALPix resolution parameter.
        seed: seed to be used (if `None`, then a random seed will
            be used).
        add_mask: return the masked splits? Default: False.
        sens: sensitivity (0, 1 or 2)
        knee: knee type (0 or 1)
        ylf: number of years for the LF tube.
        fsky: sky fraction to use for the noise realizations.
        nsplits: number of splits (i.e. independent noise realizations).
    Returns:
        A dictionary containing the noise maps.
        If `add_mask=True`, then the masked noise maps will
        be returned.
    """
    nfreq = len(freqs)
    lmax = 3*nside-1
    ells = np.arange(lmax+1)
    nells = len(ells)
    dl2cl = np.ones(len(ells))
    dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
    cl2dl = (ells*(ells+1.))/(2*np.pi)
    nell=np.zeros([nfreq,lmax+1])
    _,nell[:,2:],_=get_SO_SAT_noise(sens,knee,ylf,fsky,lmax+1,1)
    nell*=cl2dl[None,:]

    npol = 2
    nmaps = nfreq*npol
    N_ells = np.zeros([nfreq, npol, nfreq, npol, nells])
    for i,n in enumerate(freqs):
        for j in [0,1]:
            N_ells[i, j, i, j, :] = nell[i]

    # Noise maps
    npix = hp.nside2npix(nside)
    maps_noise = np.zeros([nsplits, nfreq, npol, npix])
    for s in range(nsplits):
        for i in range(nfreq):
            nell_ee = N_ells[i, 0, i, 0, :]*dl2cl * nsplits
            nell_bb = N_ells[i, 1, i, 1, :]*dl2cl * nsplits
            nell_00 = nell_ee * 0 * nsplits
            maps_noise[s, i, :, :] = hp.synfast([nell_00, nell_ee, nell_bb,
                                                 nell_00, nell_00, nell_00],
                                                nside, pol=False, new=True,
                                                verbose=False)[1:]

    if add_mask:
        nhits=hp.ud_grade(hp.read_map("./data/norm_nHits_SA_35FOV.fits", verbose=False), nside_out=nside)
        nhits/=np.amax(nhits)
        fsky_msk=np.mean(nhits)
        nhits_binary=np.zeros_like(nhits)
        inv_sqrtnhits=np.zeros_like(nhits)
        inv_sqrtnhits[nhits>1E-3]=1./np.sqrt(nhits[nhits>1E-3])
        nhits_binary[nhits>1E-3]=1
        maps_noise *= inv_sqrtnhits

    else:
        nhits_binary = np.ones((hp.nside2npix(nside)))


    dict_out = {'maps_noise': maps_noise,
                'cls_noise': N_ells,
                'mask' : nhits_binary}

    return dict_out

def healpy_cl(maps, maps2=None):

    nfreq, npol, npix = maps.shape
    nside = hp.npix2nside(npix)
    nls = 3*nside
    ells = np.arange(nls)
    cl2dl = ells*(ells+1)/(2*np.pi)

    if maps2 is None:
        maps2 = maps

    cl_out = np.zeros([nfreq, npol, nfreq, npol, nls])
    for i in range(nfreq):
        m1 = np.zeros([3, npix])
        m1[1:,:]=maps[i, :, :]
        for j in range(i,nfreq):
            m2 = np.zeros([3, npix])
            m2[1:,:]=maps2[j, :, :]

            cl = hp.anafast(m1, m2, iter=0)
            cl_out[i, 0, j, 0] = cl[1]*cl2dl
            cl_out[i, 0, j, 1] = cl[4]*cl2dl
            cl_out[i, 1, j, 1] = cl[2]*cl2dl

            if j!=i:
                cl_out[j, 0, i, 0] = cl[1]*cl2dl
                cl_out[i, 0, j, 1] = cl[4]*cl2dl
                cl_out[j, 1, i, 1] = cl[2]*cl2dl

    return cl_out

def namaster_cl(maps, maps2=None, unit_beams=True, add_mask=False, bpw_edges=False, dell=False,
                purify_b=False):
    """ Returns an array with all auto- and cross-correlations
    for a given set of Q/U frequency maps.
    Args:
        maps: set of frequency maps with shape [nfreq, 2, npix].
        maps2: set of frequency maps with shape [nfreq, 2, npix] to cross-correlate with.
        unit_beams: unit beams instead of SO beams.
        add_mask: include the SAT mask.
        bpw_edges: bin based on bpw_edges.txt file
        dell: compute dell instead cl.
        purify_b: call for B-mode purification in namaster.
    Returns:
        Set of power spectra with shape [nfreq, 2, nfreq, 2, n_ell].
    """

    nfreq, npol, npix = maps.shape
    nside = hp.npix2nside(npix)
    mask = np.ones((npix))

    import pymaster as nmt
    import urllib.request

    # Add beams
    if unit_beams is True:
        beams = [np.ones((3*nside)) for band_idx in range(nfreq)]
    else:
        band_names = ["LF1","LF2","MF1","MF2","UHF1","UHF2"]
        beams = [np.loadtxt("data/beams/beam_"+band+".txt")[:,1] for band in band_names]

    # Add mask
    if add_mask is True:
        # Use beamconv hits
        # Download SAT nhits map
        print("Download and save SAT nhits map ...")
        sat_nhits_file = "data/sat_nhits_map.fits"
        urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb"
        url = f"{urlpref}/norm_nHits_SA_35FOV_ns512.fits"
        with urllib.request.urlopen(url, timeout=30):
            urllib.request.urlretrieve(url, filename=sat_nhits_file)
        nhits_map = hp.ud_grade(hp.read_map(sat_nhits_file, field=0), nside)

        # Generate apodized map
        sat_apo_file = "data/sat_apo_map.fits"
        mask = get_apodized_mask_from_nhits(nhits_map)
        hp.write_map(sat_apo_file, mask, overwrite=True, dtype=np.int32)

        # Plot mask for debugging
        import matplotlib.pyplot as plt
        hp.mollview(mask)
        plt.savefig("data/sat_apo_mask.png")

        # Set unobserved map pixels to zero
        unobserved = mask == 0
        maps[:, :, unobserved] = 0

    # Make nmt fields
    fields = []
    for map_idx in range(nfreq):
        fields.append(nmt.NmtField(mask=mask, maps=maps[map_idx,:,:], beam=beams[map_idx], purify_b=purify_b))

    # Get bins
    if bpw_edges is True:

        # Load bandpowers
        edges = np.loadtxt("data/bpw_edges.txt").astype(int)
        bpws = np.zeros(3*nside, dtype=int)-1
        weights = np.ones(3*nside)

        for ibpw, (l0, lf) in enumerate(zip(edges[:-1], edges[1:])):
            if lf < 3*nside:
                bpws[l0:lf] = ibpw

        if edges[-1] < 3*nside:
            dell = edges[-1]-edges[-2]
            l0 = edges[-1]
            while l0+dell < 3*nside:
                ibpw += 1
                bpws[l0:l0+dell] = ibpw
                l0 += dell

        larr_all = np.arange(3*nside)
        nmt_bins = nmt.NmtBin(nside,
                              bpws=bpws,
                              ells=larr_all,
                              weights=weights,
                              is_Dell=dell)
        nls = ibpw+1
        start_idx = 0

    else:
        nmt_bins = nmt.NmtBin(nside,
                              nlb = 1,
                              is_Dell=dell)
        nls = 3*nside
        # no monopole/dipole from namaster
        start_idx = 2

    # Compute auto/cross-spectra
    cl_out = np.zeros([nfreq, npol, nfreq, npol, nls])

    for i in range(nfreq):
        for j in range(i,nfreq):
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(fields[i], fields[j], nmt_bins)
            pcl = nmt.compute_coupled_cell(fields[i], fields[j])
            cell_ij = w.decouple_cell(pcl)

            cl_out[i, 0, j, 0][start_idx:] = cell_ij[0]
            cl_out[i, 0, j, 1][start_idx:] = cell_ij[1]
            cl_out[i, 1, j, 0][start_idx:] = cell_ij[2]
            cl_out[i, 1, j, 1][start_idx:] = cell_ij[3]

            if j!=i:
                cl_out[j, 0, i, 0][start_idx:] = cell_ij[0]
                cl_out[i, 0, j, 1][start_idx:] = cell_ij[1]
                cl_out[i, 1, j, 0][start_idx:] = cell_ij[2]
                cl_out[j, 1, i, 1][start_idx:] = cell_ij[3]

    return cl_out


def get_gaussian_beta_map(nside, beta0, amp, gamma=0, l0=80, l_cutoff=2, mean_params=None):
    """
    Returns realization of the spectral index map.
    Args:
        nside: HEALPix resolution parameter.
        beta0: mean spectral index.
        amp: amplitude
        gamma: tilt
        l0: pivot scale (default: 80)
        l_cutoff: ell below which the power spectrum will be zero.
            (default: 2).
        seed: seed (if None, a random seed will be used).
        gaussian: beta map from power law spectrum (if False, a spectral
            index map obtained from the Planck data using the Commander code
            is used for dust, and ... for sync)
    Returns:
        Spectral index map
    """

    ls = np.arange(lmax)
    cls = get_delta_beta_cl(amp, gamma, ls, l0, l_cutoff)
    mp = hp.synfast(cls, nside, verbose=False)
    mp += beta0
    return mp

def get_sky_realization(nside, freqs, plaw_amps=True, gaussian_betas=True, seed=1001,
                        mean_params=None, moment_pars=None,compute_cls=False,
                        delta_ell=10, foregrounds=False):
    """ Generate a sky realization for a set of input sky parameters.
    Args:
        nside: HEALPix resolution parameter.
        seed: seed to be used (if `None`, then a random seed will
            be used).
        mean_params: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        moment_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        compute_cls: return also the power spectra? Default: False.
        delta_ell: bandpower size to use if compute_cls is True.
        gaussian_betas: gaussian spectral index maps, see 'get_beta_map'.
            Default: True.
        plaw_amps: dust and synchrotron amplitude maps modelled as power
            laws. If false, returns realistic amplitude maps in equatorial
            coordinates. Default: True.
    Returns:
        A dictionary containing the different component maps,
        spectral index maps and frequency maps.
        If `compute_cls=True`, then the dictionary will also
        contain information of the signal, noise and total
        (i.e. signal + noise) power spectra.
    """

    npix = hp.nside2npix(nside)
    lmax = 3*nside-1

    mean_spectra  = get_mean_spectra(lmax, mean_params, foregrounds)
    sky_config = dict()
    if foregrounds is True:
        ells, dl2cl, cl2dl, cl_dust_bb, cl_dust_ee, cl_sync_bb, cl_sync_ee, cl_cmb_bb, cl_cmb_ee = mean_spectra
        cl0 = 0 * cl_dust_bb

        # Dust amplitudes
        Q_dust, U_dust = hp.synfast([cl0, cl_dust_ee, cl_dust_bb, cl0, cl0, cl0],
                                    nside, new=True)[1:]
        # Sync amplitudes
        Q_sync, U_sync = hp.synfast([cl0, cl_sync_ee, cl_sync_bb, cl0, cl0, cl0],
                                    nside, new=True)[1:]

        # Dust temperature
        temp_dust = np.ones(npix) * mean_params['temp_dust']

        # Create PySM simulation
        zeromap = np.zeros(npix)

        # Dust
        d2 = models("d2", nside)
        # Set own parameters
        d2[0]['spectral_index'] = np.ones(npix)*mean_params['beta_dust']
        d2[0]['temp'] = mean_params['temp_dust']
        d2[0]['nu_0_I'] = mean_params['nu0_dust']
        d2[0]['nu_0_P'] = mean_params['nu0_dust']
        d2[0]['A_I'] = zeromap
        d2[0]['A_Q'] = Q_dust
        d2[0]['A_U'] = U_dust

        # Sync
        s1 = models("s1", nside)
        # Set own parameters
        s1[0]['nu_0_I'] = mean_params['nu0_sync']
        s1[0]['nu_0_P'] = mean_params['nu0_sync']
        s1[0]['A_I'] = zeromap
        s1[0]['A_Q'] = Q_sync
        s1[0]['A_U'] = U_sync
        s1[0]['spectral_index'] = np.ones(npix)*mean_params['beta_sync']

        sky_config['dust'] = d2
        sky_config['synchrotron'] = s1


    else:

        ells, dl2cl, cl2dl, cl_cmb_bb, cl_cmb_ee = mean_spectra

    cl0 = 0 * cl_cmb_bb

    # CMB amplitude
    I_cmb, Q_cmb, U_cmb = hp.synfast([cl0, cl_cmb_ee, cl_cmb_bb, cl0, cl0, cl0],
                              nside, new=True, verbose=False)

    # CMB
    c1 = models("c1", nside)
    c1[0]['model'] = 'pre_computed' #different output maps at different seeds
    c1[0]['A_I'] = I_cmb
    c1[0]['A_Q'] = Q_cmb
    c1[0]['A_U'] = U_cmb

    sky_config['cmb'] = c1

    # Beams
    if mean_params['unit_beams']==True:
        bms_fwhm = np.ones_like(freqs)
        smooth = False
    else:
        bms_fwhm = get_SO_SAT_beams_fwhms()
        smooth = True

    sky = pysm.Sky(sky_config)
    instrument_config = {
        'nside' : nside,
        'frequencies' : freqs, #Expected in GHz
        'use_smoothing' : smooth, #Set if including beams
        'beams' : bms_fwhm, #Expected FWHM in arcmin
        'add_noise' : False,
        'use_bandpass' : False,
        'channel_names' : ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2'],
        'output_units' : 'uK_RJ',
        'output_directory' : 'none',
        'output_prefix' : 'none',
    }
    sky = pysm.Sky(sky_config)
    instrument = pysm.Instrument(instrument_config)
    maps_signal, _ = instrument.observe(sky, write_outputs=False)
    maps_signal = maps_signal[:,1:,:]
    # Change to CMB units
    maps_signal = maps_signal/fcmb(freqs)[:,None,None]

    dict_out = {'maps_cmb': np.array([I_cmb, Q_cmb, U_cmb]),
                'freq_maps': maps_signal}

    return maps_signal

def main():

    parser = ap.ArgumentParser(
            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--freqs",
        action="store",
        dest="freqs",
        help="The frequencies to simulate.",
        nargs='+',
        default=[27., 39., 93., 145., 225., 280.],
    )

    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        help="Simulation seed.",
        default=1001,
    )

    parser.add_argument(
        "--nside",
        action="store",
        dest="nside",
        help="Nside of the simulated maps.",
        default=128,
    )

    parser.add_argument(
        "--outdir",
        action="store",
        dest="outdir",
        help="The location for the spectra/maps to be stored.",
    )

    parser.add_argument(
        "--add_mask",
        action="store_true",
        dest="add_mask",
        help="If True, use mask.",
        default=False,
    )

    parser.add_argument(
        "--write_map",
        action="store_true",
        dest="write_map",
        default=False,
        help="If True, save signal maps.",
    )

    parser.add_argument(
        "--write_splits",
        action="store_true",
        dest="write_splits",
        help="If True, write noise splits.",
        default=False,
    )

    parser.add_argument(
        "--r_tensor",
        action="store",
        dest="r_tensor",
        type=float,
        help="The r tensor value.",
        default=0.,
    )

    parser.add_argument(
        "--beta_dust",
        action="store",
        dest="beta_dust",
        type=float,
        help="The spectral index for Dust",
        default=1.54,
    )

    parser.add_argument(
        "--A_d_BB",
        action="store",
        dest="A_d_BB",
        type=float,
        help="The amplitude in multipole space \
             for Dust",
        default=28.,
    )

    parser.add_argument(
        "--alpha_d_BB",
        action="store",
        dest="alpha_d_BB",
        type=float,
        help="The scaling index in multipole space \
              for Dust",
        default=-0.16,
    )

    parser.add_argument(
        "--beta_sync",
        action="store",
        dest="beta_sync",
        type=float,
        help="The spectral index for Synchrotron",
        default=-3.,
    )

    parser.add_argument(
        "--A_s_BB",
        action="store",
        dest="A_s_BB",
        type=float,
        help="The amplitude in multipole space \
             for Synchrotron",
        default=1.6,
    )

    parser.add_argument(
        "--alpha_s_BB",
        action="store",
        dest="alpha_s_BB",
        type=float,
        help="The scaling index in multipole space \
              for Synchrotron",
        default=-0.93,
    )

    parser.add_argument(
        "--unit_beams",
        action="store_true",
        dest="unit_beams",
        help="If True, use unit beams.",
        default=False,
    )

    parser.add_argument(
        "--foregrounds",
        action="store_true",
        dest="foregrounds",
        help="If True, add dust and synchrotron components.",
        default=False,
    )

    args = parser.parse_args()
    print('r_tensor=',args.r_tensor, type(args.r_tensor))
    mean_params = dict()
    mean_params = {# fixed parameters
                   'A_lens': 1,
                   'A_s_EE': 9,
                   'alpha_s_EE': 0.7,
                   'A_d_EE': 56,
                   'alpha_d_EE': -0.32,
                   'temp_dust': 20,
                   'nu0_dust': 353,
                   'nu0_sync': 23,
                   # user input parameters
                   'r_tensor': args.r_tensor,
                   'unit_beams': args.unit_beams,
                   'beta_dust': args.beta_dust,
                   'A_d_BB': args.A_d_BB,
                   'alpha_d_BB': args.alpha_d_BB,
                   'beta_sync': args.beta_sync,
                   'A_s_BB': args.A_s_BB,
                   'alpha_s_BB': args.alpha_s_BB
                   }

    get_data_spectra(freqs=np.array(args.freqs),
                     seed=args.seed,
                     nside=args.nside,
                     outdir=args.outdir,
                     mean_params=mean_params,
                     add_mask=args.add_mask,
                     write_map=args.write_map,
                     write_splits=args.write_splits,
                     foregrounds=args.foregrounds)


if __name__ == '__main__':


    main()
