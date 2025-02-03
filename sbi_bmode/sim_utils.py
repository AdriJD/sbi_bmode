'''
Utils for simulating data using a simple Gaussian foreground model.
'''

import os

import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import alm_utils, sht, map_utils, mat_utils
import healpy as hp
from jax import grad
import jax.numpy as jnp

from sbi_bmode import spectra_utils, so_utils, nilc_utils, likelihood_utils

opj = os.path.join

class CMBSimulator():
    '''
    Generate CMB data vectors and power spectra.

    Parameters
    ----------
    specdir : str
        Path to data directory containing power spectrum files.
    data_dict:
    fixed_params_dict:
    pyilcdir: str
        Path to pyilc respository. Setting to None means NILC is not used.
    use_dbeta_map: Bool
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of first moment w.r.t. beta and include it in
        auto- and cross-spectra in the data vector
    deproj_dust: Bool
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject dust in CMB NILC map.
    deproj_dbeta: Bool
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject first moment of dust w.r.t. beta in CMB NILC map.
    odir: str
        Path to output directory
    norm_params: dict, optional
        Parameters of fiducial model used to whiten the (multifrequency) data.
    score_params: dict, optional
        Parameters of fiducial model where the score is evaluted for score
        compression.
    '''
    
    def __init__(self, specdir, data_dict, fixed_params_dict, pyilcdir=None, use_dbeta_map=False,
                deproj_dust=False, deproj_dbeta=False, odir=None, norm_params=None, score_params=None):
        
        self.lmax = data_dict['lmax']
        self.lmin = data_dict['lmin']
        self.nside = data_dict['nside']
        self.nsplit = data_dict['nsplit']        
        self.delta_ell = data_dict['delta_ell']
        self.pyilcdir = pyilcdir
        self.use_dbeta_map = use_dbeta_map
        self.deproj_dust = deproj_dust
        self.deproj_dbeta = deproj_dbeta
        self.odir = odir
        self.bins = np.arange(self.lmin, self.lmax, self.delta_ell)

        self.cov_scalar_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_nobb.dat'), self.lmax)
        self.cov_tensor_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_r1.dat'), self.lmax)

        self.minfo = map_utils.MapInfo.map_info_healpix(self.nside)
        self.ainfo = curvedsky.alm_info(self.lmax)

        self.freq_strings = data_dict['freq_strings']
        self.freqs = [so_utils.sat_central_freqs[fstr] for fstr in self.freq_strings]
        self.nfreq = len(self.freqs)
        if pyilcdir:
            self.ncomp = 2 if not self.use_dbeta_map else 3
            self.size_data = get_ntri(self.nsplit, self.ncomp) * (self.bins.size - 1)
        else:
            self.size_data = get_ntri(self.nsplit, self.nfreq) * (self.bins.size - 1)

        self.sensitivity_mode = data_dict['sensitivity_mode']
        self.lknee_mode = data_dict['lknee_mode']
        self.noise_cov_ell = np.ones((self.nfreq, 2, 2, self.lmax + 1))
        self.fsky = data_dict['fsky']
        
        for fidx, fstr in enumerate(self.freq_strings):

            # We scale the noise with the number of splits.
            self.noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * so_utils.get_sat_noise(
                fstr, self.sensitivity_mode, self.lknee_mode, self.fsky, self.lmax) \
                * self.nsplit

        # Fixed parameters.
        self.freq_pivot_dust = fixed_params_dict['freq_pivot_dust']
        self.temp_dust = fixed_params_dict['temp_dust']

        self.norm_params = norm_params
        if self.norm_params and pyilcdir is None:
            
            self.norm_model = np.asarray(self.get_signal_spectra(
                norm_params['r_tensor'], norm_params['A_lens'], norm_params['A_d_BB'],
                norm_params['alpha_d_BB'], norm_params['beta_dust']))
            noise_spectra = self.get_noise_spectra()
            norm_cov = likelihood_utils.get_cov(
                self.norm_model, noise_spectra, self.bins, self.lmin,
                self.lmax, self.nsplit, self.nfreq)
            self.sqrt_norm_cov = mat_utils.matpow(norm_cov, 0.5, return_diag=True)
            self.isqrt_norm_cov = mat_utils.matpow(norm_cov, -0.5, return_diag=True)  
            
        self.score_params = score_params
        if self.score_params and pyilcdir is None:
            
            if self.score_params and self.norm_params:
                raise ValueError('Cannot have both norm_params and score_params')
            
            self.score_model = np.asarray(self.get_signal_spectra(
                score_params['r_tensor'], score_params['A_lens'], score_params['A_d_BB'],
                score_params['alpha_d_BB'], score_params['beta_dust']))
            noise_spectra = self.get_noise_spectra()
            cov = likelihood_utils.get_cov(
                self.score_model, noise_spectra, self.bins, self.lmin,
                self.lmax, self.nsplit, self.nfreq)
            tri_indices = get_tri_indices(self.nsplit, self.nfreq)
            icov = mat_utils.matpow(cov, -1, return_diag=True)
            
            score_params_arr = jnp.asarray(
                [score_params['r_tensor'], score_params['A_lens'], score_params['A_d_BB'],
		 score_params['alpha_d_BB'], score_params['beta_dust']])
            
            def get_loglike(params, data):

                data = data.reshape(tri_indices.shape[0], -1)
                model = self.get_signal_spectra(*params)
                loglike = likelihood_utils.loglike(model, data, icov, tri_indices)

                return loglike
                
            self.grad_logdens = grad(get_loglike, argnums=0)
            self.score_compress = lambda x: self.grad_logdens(score_params_arr, x)
            
    def get_signal_spectra(self, r_tensor, A_lens, A_d_BB, alpha_d_BB, beta_dust):
        '''
        Generate binned signal frequency cross spectra.

        Parameters
        ----------
        r_tensor : float
            Tensor-to-scalar ratio.
        A_lens : float
            A_lens parameter.
        A_d_BB : float
        
        Returns
        -------
        cov_bin : (nfreq, nfreq, nbin) array
            Signal frequency cross spectra.
        '''

        cov_ell = spectra_utils.get_dust_spectra(
            A_d_BB, alpha_d_BB, self.lmax, self.freqs, beta_dust, self.temp_dust,
            self.freq_pivot_dust)

        # Only adding the BB part because `get_dust_spectra` only produces BB.
        cov_ell = cov_ell.at[:].add(spectra_utils.get_combined_cmb_spectrum(
            r_tensor, A_lens, self.cov_scalar_ell, self.cov_tensor_ell)[1,1])

        cov_bin = spectra_utils.bin_spectrum(
            cov_ell, np.arange(self.lmax+1), self.bins, self.lmin, self.lmax,
            use_jax=True)

        return cov_bin

    def get_noise_spectra(self, use_jax=False):
        '''

        Returns
        -------
        cov_bin : (nfreq, nfreq, nbin) array
            Noise frequency cross spectra.        
        '''

        out = np.zeros((self.nfreq, self.nfreq, self.lmax+1))                
        out[:] = np.eye(self.nfreq)[:,:,np.newaxis] * self.noise_cov_ell[:,1,1]

        cov_bin = spectra_utils.bin_spectrum(
            out, np.arange(self.lmax+1), self.bins, self.lmin, self.lmax, use_jax=use_jax)

        return cov_bin
        
    def draw_data(self, r_tensor, A_lens, A_d_BB, alpha_d_BB, beta_dust, seed):
        '''
        Draw data realization.

        Parameters
        ----------
        r_tensor : float
            Tensor-to-scalar ratio.
        A_lens : float
            Amplitude of lensing contribution to BB.        
        A_d_BB : float
            Dust amplitude.
        alpha_d_BB : float
            Dust spatial power law index.
        beta_dust : float
            Dust frequency power law index.
        seed : int, np.random._generator.Generator object
            Seed or random number generator object.

        Returns
        -------
        data : (ndata) array
            Data realization.
        '''
        
        if seed == -1:
            seed = None
        seed = np.random.default_rng(seed=seed)

        omap = gen_data(
            A_d_BB, alpha_d_BB, beta_dust, self.freq_pivot_dust, self.temp_dust,
            r_tensor, A_lens, self.freqs, seed, self.nsplit, self.noise_cov_ell,
            self.cov_scalar_ell, self.cov_tensor_ell, self.minfo, self.ainfo)

        if self.pyilcdir:
            # build NILC B-mode maps with shape (nsplit, ncomp=2, npix)
            nfreq = len(self.freqs)
            B_maps = np.zeros((self.nsplit, nfreq, self.minfo.npix))
            for split in range(self.nsplit):
                for f, freq_str in enumerate(self.freq_strings):
                    Q, U = omap[split, f]
                    # alm_T is just a placeholder                    
                    alm_T, alm_E, alm_B = hp.map2alm([np.zeros_like(Q), Q, U], pol=True) 
                    B_maps[split, f] = 10**(-6)*hp.alm2map(alm_B, self.nside)
            map_tmpdir = nilc_utils.write_maps(B_maps, output_dir=self.odir)
            nilc_maps = nilc_utils.get_nilc_maps(self.pyilcdir, map_tmpdir, self.nsplit, self.nside, 
                                                 beta_dust, self.temp_dust, self.freq_pivot_dust, 
                                                 so_utils.sat_central_freqs, so_utils.sat_beam_fwhms,
                                                 use_dbeta_map=self.use_dbeta_map, deproj_dust=self.deproj_dust,
                                                 deproj_dbeta=self.deproj_dbeta, output_dir=self.odir,
                                                 remove_files=True, debug=False)
            spectra = estimate_spectra_nilc(nilc_maps, self.minfo, self.ainfo)
        
        else:
            spectra = estimate_spectra(omap, self.minfo, self.ainfo)
            
        data = get_final_data_vector(spectra, self.bins, self.lmin, self.lmax)

        if self.norm_params:
            data = self.get_norm_data(data)
            
        return data

    def get_norm_data(self, data):
        '''
        Subtract mean and multiply by inverse sqrt of covariance.

        Parameters
        ----------
        data : (ndata) array
            Input data.
        
        Returns
        -------
        data_norm : (ndata) array
            Normalized data.
        '''
        
        ntri = get_ntri(self.nsplit, self.nfreq)
        tri_indices = get_tri_indices(self.nsplit, self.nfreq)
        data = likelihood_utils.get_diff(
            data.reshape(ntri, -1), self.norm_model, tri_indices)
        data = np.einsum('ijk, jk -> ik', self.isqrt_norm_cov, data)
        data_norm = data.reshape(-1)
        
        return data_norm

    def get_unnorm_data(self, data_norm):
        '''
        Subtract mean and multiply by inverse sqrt of covariance.

        Parameters
        ----------
        data_norm : (ndata) array
            Nomalized data.
        
        Returns
        -------
        data : (ndata) array
            Unnormalized data.
        '''

        ntri = get_ntri(self.nsplit, self.nfreq)
        tri_indices = get_tri_indices(self.nsplit, self.nfreq)
        data = np.einsum(
            'ijk, jk -> ik', self.sqrt_norm_cov, data_norm.reshape(ntri, -1))
        data = likelihood_utils.get_diff(
            data, -self.norm_model, tri_indices)
        data = data.reshape(-1)

        return data
    
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
    rng_cmb = rngs[0]
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
        g_factor = spectra_utils.get_g_fact(freq)

        signal_alm = cmb_alm.copy()
        signal_alm += dust_alm * np.sqrt(dust_factor * np.abs(A_d_BB)) * g_factor * np.sign(A_d_BB)

        for sidx in range(nsplit):

            data_alm = signal_alm + alm_utils.rand_alm(
                cov_noise_ell[fidx], ainfo, rngs_noise[sidx], dtype=np.complex128)
            data_alm = np.asarray(data_alm, dtype=np.complex128)
            sht.alm2map(data_alm, out[sidx,fidx], ainfo, minfo, 2)

    return out

def apply_obsmatrix(imap, obs_matrix):
    '''
    Transform a set of maps by applying an observation matrix.

    Parameters
    ----------
    imap : (nsplit, nfreq, npol, npix) array
        A set of maps as input
    obsmatrix: (npol*npix, npol*npix) sparse array object
      A square matrix that simulates observation effects

    Returns
    -------
    omap : (nsplit, nfreq, npol, npix) array
        Filtered output maps.
    '''
    reobs_imap = np.empty_like(imap)
    nsplit = imap.shape[0]
    nfreq = imap.shape[1]
    for i in range(nsplit):
        for j in range(nfreq):
            nest_imap = hp.reorder(imap[i,j], r2n=True)
            reobs_imap[i,j] = hp.reorder(
                obs_matrix.dot(nest_imap.ravel()).reshape([3, -1]), n2r=True)

    return reobs_imap

def get_ntri(nsplit, nfreq):
    '''
    Get the number of elements in the upper triangle of the
    (nsplit x nfreq) x (nsplit x nfreq) matrix.

    Parameters
    ----------
    nsplit : int
        Number of splits.
    nfreq : int
        Number of frequencies.

    Returns
    -------
    ntri : int
        Number of elements in upper triangle.
    '''

    return nfreq * nfreq * (nsplit * (nsplit - 1) // 2)

def get_tri_indices(nsplit, nfreq):
    '''
    Get indices into upper-triangular part of the
    (nsplits * nfreq) x (nsplits * nfreq) cross-spectrum matrix.

    Parameters
    ----------
    nsplit : int

    nfreq : int

    Returns
    -------
    tri_indices : (ntri, 4) array
        The sidx1, fidx1, sidx2, fidx2 indices into the split and freq
        axes for each element.
    '''

    idxs = []
    for sidx in range(nsplit):
        for fidx in range(nfreq):
            idxs.append((sidx, fidx))

    ntot = nsplit * nfreq
    ntri = get_ntri(nsplit, nfreq)
    tri_indices = np.zeros((ntri, 4), dtype=int)

    idx = 0
    for idx1 in range(ntot):
        for idx2 in range(idx1, ntot):

            sidx1, fidx1 = idxs[idx1]
            sidx2, fidx2 = idxs[idx2]
            # Exclude all elements that contain equal splits.
            if sidx1 != sidx2:
                tri_indices[idx] = [sidx1, fidx1, sidx2, fidx2]
                idx += 1

    return tri_indices

def estimate_spectra(imap, minfo, ainfo):
    '''
    Compute all the cross-spectra between splits and
    and frequency bands. NOTE Right now EE, EB are discarded and
    all spectra that involve two maps of the same splits are also
    discarded.

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
    out : (ntri, 1, lmax + 1)
        Output BB spectra. Only the elements of the upper-triangular part
        of the (nsplits * nfreq) x (nsplits * nfreq) matrix are included.
    '''

    nsplit = imap.shape[0]
    nfreq = imap.shape[1]

    # Number of elements in the upper triangle of the ntot x ntot matrix.
    ntri = get_ntri(nsplit, nfreq)
    out = np.zeros((ntri, 1, ainfo.lmax + 1))

    alm = np.zeros((nsplit, nfreq, 2, ainfo.nelem), dtype=np.complex128)
    sht.map2alm(imap, alm, minfo, ainfo, 2)

    tri_indices = get_tri_indices(nsplit, nfreq)
    for idx, (sidx1, fidx1, sidx2, fidx2) in enumerate(tri_indices):
        out[idx] = ainfo.alm2cl(
            alm[sidx1,fidx1,:,None,:], alm[sidx2,fidx2,None,:,:])[1,1]

    return out

def estimate_spectra_nilc(imap, minfo, ainfo):
    '''
    Compute all the auto and cross-spectra between splits and
    and frequency bands.

    Parameters
    ----------
    imap : (nsplit, ncomp=2, npix)
        Input maps.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.

    Returns
    -------
    out : (ntri, 1, lmax + 1)
        Output BB spectra. Only the elements of the upper-triangular part
        (+ the diagonal) of the (nsplits * nfreq) x (nsplits * nfreq) matrix
        are included.
    '''

    nsplit = imap.shape[0]
    ncomp = imap.shape[1]

    # Number of elements in the upper triangle of the ntot x ntot matrix.
    ntri = get_ntri(nsplit, ncomp)
    out = np.zeros((ntri, 1, ainfo.lmax + 1))

    tri_indices = get_tri_indices(nsplit, ncomp)
    for idx, (sidx1, fidx1, sidx2, fidx2) in enumerate(tri_indices):
        out[idx, 0] = hp.anafast(imap[sidx1, fidx1], imap[sidx2, fidx2], lmax=ainfo.lmax)

    return out

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

        out[idxs] = spectra_utils.bin_spectrum(spec[idxs], ells, bins, lmin, lmax)

    return out.reshape(-1)
