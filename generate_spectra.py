import healpy as hp
import numpy as np
import os
import argparse as ap
import yaml
import pysm
from pysm.nominal import models

opj = os.path.join 

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
    l = l.astype(int)
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

        print('I am here')
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


def get_SO_SAT_beams_fwhms():
    ## returns the SAT beams in arcminutes
    beam_SAT_27 = 91.
    beam_SAT_39 = 63.
    beam_SAT_93 = 30.
    beam_SAT_145 = 17.
    beam_SAT_225 = 11.
    beam_SAT_280 = 9.
    return(np.array([beam_SAT_27,beam_SAT_39,beam_SAT_93,beam_SAT_145,beam_SAT_225,beam_SAT_280]))

# def Simons_Observatory_V3_SA_beams(ell):
#     SA_beams = Simons_Observatory_V3_SA_beam_FWHM() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
#     ## SAT beams as a sigma expressed in radians
#     return [np.exp(-0.5*ell*(ell+1)*sig**2.) for sig in SA_beams]

def get_SO_SAT_noise(sensitivity_mode,one_over_f_mode,SAT_yrs_LF,f_sky,ell_max,delta_ell,
                                   whitenoi_ONLY=True, include_kludge=True, include_beam=False):
    ## returns noise curves in polarization only, including the impact of the beam, for the SO small aperture telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     1: baseline, 
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    # SAT_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAT
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERTURE
    # ensure valid parameter choices
    assert( sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( one_over_f_mode == 0 or one_over_f_mode == 1)
    assert( SAT_yrs_LF <= 5) #N.B. SAT_yrs_LF can be negative
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # configuration
    if (SAT_yrs_LF > 0):
        NTubes_LF  = SAT_yrs_LF/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 - SAT_yrs_LF/5.
    else:
        NTubes_LF  = np.fabs(SAT_yrs_LF)/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 
    NTubes_UHF = 1.
    # sensitivity
    # N.B. divide-by-zero will occur if NTubes = 0
    # handle with assert() since it's highly unlikely we want any configurations without >= 1 of each tube type
    assert( NTubes_LF > 0. )
    assert( NTubes_MF > 0. )
    assert( NTubes_UHF > 0.)
    S_SA_27  = np.array([1.e9,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([1.e9,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([1.e9,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([1.e9,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([1.e9,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([1.e9,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f polarization noise
    # see Sec. 2.2 of the SO science goals paper
    f_knee_pol_SA_27  = np.array([30.,15.])
    f_knee_pol_SA_39  = np.array([30.,15.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.])
    f_knee_pol_SA_145 = np.array([50.,25.])  ## from ABS, improvement possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.])
    f_knee_pol_SA_280 = np.array([100.,40.])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])
    
    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    if include_kludge:
        t = t* 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4 * np.pi * f_sky  ## sky area in steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")
    #print("Note that this code includes a factor of 1/0.85 increase in the noise power, corresponding to assumed mode loss due to map depth non-uniformity.")
    #print("If you have your own N_hits map that already includes such non-uniformity, you should increase the total integration time by a factor of 1/0.85 when generating noise realizations from the power spectra produced by this code, so that this factor is not mistakenly introduced twice.")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2,ell_max,delta_ell)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_SA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_SA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_SA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)
    
    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels = np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise levels (T): ",Map_white_noise_levels ,"[uK-arcmin]")
    
    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    ## see Sec. 2.2 of the SO science goals paper
    AN_P_27  = (ell / f_knee_pol_SA_27[one_over_f_mode] )**alpha_pol[0] + 1.  
    AN_P_39  = (ell / f_knee_pol_SA_39[one_over_f_mode] )**alpha_pol[1] + 1. 
    AN_P_93  = (ell / f_knee_pol_SA_93[one_over_f_mode] )**alpha_pol[2] + 1.   
    AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.   
    AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.   
    AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.

    if whitenoi_ONLY is True:
        AN_P_27 = np.ones_like(AN_P_27)
        AN_P_39 = np.ones_like(AN_P_39)
        AN_P_93 = np.ones_like(AN_P_93)
        AN_P_145 = np.ones_like(AN_P_145)
        AN_P_225 = np.ones_like(AN_P_225)
        AN_P_280 = np.ones_like(AN_P_280)

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280

    if include_beam:
        ## include the impact of the beam
        SA_beams = Simons_Observatory_V3_SA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
        ## SAT beams as a sigma expressed in radians
        N_ell_P_27  *= np.exp( ell*(ell+1)* SA_beams[0]**2. )
        N_ell_P_39  *= np.exp( ell*(ell+1)* SA_beams[1]**2. )
        N_ell_P_93  *= np.exp( ell*(ell+1)* SA_beams[2]**2. )
        N_ell_P_145 *= np.exp( ell*(ell+1)* SA_beams[3]**2. )
        N_ell_P_225 *= np.exp( ell*(ell+1)* SA_beams[4]**2. )
        N_ell_P_280 *= np.exp( ell*(ell+1)* SA_beams[5]**2. )
    
    ## make an array of noise curves for P
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])
    
    ####################################################################
    return(ell,N_ell_P_SA,Map_white_noise_levels)


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


def get_sacc(freqs, cls, l_unbinned, params):
    import sacc

    nfreq = len(freqs)

    s = sacc.Sacc()

    for inu, nu in enumerate(nus):
        nu_s = np.array([nu-1, nu, nu+1])
        bnu_s = np.array([0.0, 1.0, 0.0])
        s.add_tracer('NuMap', 'band%d' % (inu+1),
                     quantity='cmb_polarization',
                     spin=2,
                     nu=nu_s,
                     bandpass=bnu_s,
                     ell=l_unbinned,
                     beam=np.ones_like(l_unbinned),
                     nu_unit='GHz',
                     map_unit='uK_CMB')

    pdict = ['e', 'b']

    for inu1, ipol1, i1, inu2, ipol2, i2, ix in iter_cls(nfreq):
        n1 = f'band{inu1+1}'
        n2 = f'band{inu2+1}'
        p1 = pdict[ipol1]
        p2 = pdict[ipol2]
        cl_type = f'cl_{p1}{p2}'
        s.add_ell_cl(cl_type, n1, n2, l_unbinned, cls[ix])


    return s


def get_apodized_mask_from_nhits(nhits_map,
                                 zero_threshold=1e-3,
                                 apodization_scale=10,
                                 apodization_type="C1"):
    import pymaster as nmt

    # Smooth nhits map
    nhits_smoothed = hp.smoothing(nhits_map, fwhm=np.pi/180, verbose=False)
    nhits_smoothed[nhits_smoothed < 0] = 0

    # Normalize maps
    nhits_map /= np.amax(nhits_map)
    nhits_smoothed /= np.amax(nhits_smoothed)

    # Threshold smoothed nhits map
    nhits_smoothed_thresholded = np.zeros_like(nhits_smoothed)
    nhits_smoothed_thresholded[nhits_smoothed > zero_threshold] = 1

    # Apodize the non-smoothed and smoothed binary mask
    nhits_smoothed_thresholded_apo = nmt.mask_apodization(
        nhits_smoothed_thresholded, apodization_scale, 
        apotype=apodization_type
    )

    return nhits_map * nhits_smoothed_thresholded_apo


def get_data_spectra(freqs, seed, nside, outdir, mean_params=None,
                     add_mask=False, write_map=False, write_splits=False,
                     foregrounds=False):

    # Make signal maps
    maps_signal= get_sky_realization(nside=nside, freqs=freqs, seed=seed, mean_params=mean_params,
                                     foregrounds=foregrounds)
    
    # Make noise splits
    noise = create_noise_splits(freqs=freqs,nside=nside)
    maps_noise = noise['maps_noise']

    if write_map:
        # Save sky maps
        nfreq = len(freqs)
        npol = 2
        nmaps = nfreq*npol
        npix = hp.nside2npix(nside)
        hp.write_map(outdir+"/maps_sky_signal.fits", maps_signal.reshape([nmaps,npix]),
        overwrite=True)

    # Add signal and noise splits
    nsplits = len(maps_noise)
    for s in range(nsplits):
        maps_signoi = maps_signal[:,:,:]+maps_noise[s,:,:,:]
        if add_mask:
            maps_signoi *= noise['mask']
        if write_splits:
            hp.write_map(outdir+"/obs_split%dof%d.fits.gz" % (s+1, nsplits),
                         (maps_signoi).reshape([nmaps,npix]),
                         overwrite=True)

        cls_unbinned = healpy_cl(maps_signoi)
        filename = 'spectra_r'+str(mean_params['r_tensor'])+'_'+str(seed)+'nsplits'+str(s)
        np.save(opj(outdir,filename), cls_unbinned)

    return



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
    



