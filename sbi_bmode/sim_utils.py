'''
Utils for simulating data using a simple Gaussian foreground model.
'''

import os

import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import alm_utils, sht, map_utils, mat_utils, alm_c_utils
import healpy as hp
from jax import grad
import jax.numpy as jnp

from sbi_bmode import (spectra_utils, so_utils, nilc_utils, likelihood_utils,
                       planck_utils, wmap_utils)

opj = os.path.join

class CMBSimulator():
    '''
    Generate CMB data vectors and power spectra.

    Parameters
    ----------
    specdir : str
        Path to data directory containing power spectrum files.
    data_dict : dict
        Dictionary with data generation parameters.
    fixed_params_dict : dict
        Dictionary with parameter names and values that we keep fixed.
    pyilcdir: str
        Path to pyilc respository. Setting to None means NILC is not used.
    use_dust_map: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of dust and include it in auto- and cross-spectra in
        the data vector
    use_dbeta_map: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of first moment w.r.t. beta and include it in
        auto- and cross-spectra in the data vector
    use_sync_map: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of synchrotron and include it in auto- and cross-spectra in
        the data vector
    use_dbeta_sync_map: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether
        to build map of first moment w.r.t. beta_synchrotron and include it in
        auto- and cross-spectra in the data vector
    deproj_dust: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject dust in CMB NILC map.
    deproj_dbeta: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject first moment of dust w.r.t. beta in CMB NILC map.
    deproj_sync: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject synchrotron in CMB NILC map.
    deproj_dbeta_sync: bool, optional
        Only relevant if using nilc (pyilc dir is not None). Whether to
        deproject first moment of synchrotron w.r.t. beta in CMB NILC map.
    fiducial_beta: float, optional
        Use this value for beta when building nilc maps.
    fiducial_T_dust: float, optional
        Use this value for T_dust when building nilc maps.
    fiducial_beta_sync: float, optional
        Use this value for beta_synchrotron when building nilc maps.
    odir: str
        Path to output directory
    norm_params: dict, optional
        Parameters of fiducial model used to whiten the (multifrequency) data.
    score_params: dict, optional
        Parameters of fiducial model where the score is evaluted for score
        compression.
    coadd_equiv_crosses: bool, optional
        Whether to use the mean of e.g. comp1 x comp2 and comp2 x comp1 spectra.
    apply_highpass_filter: bool, optional
        Filter out signal modes below lmin in the simulations.
    '''

    def __init__(self, specdir, data_dict, fixed_params_dict, pyilcdir=None, use_dust_map=True,
                 use_dbeta_map=False, use_sync_map=False, use_dbeta_sync_map=False,
                 deproj_dust=False, deproj_dbeta=False, deproj_sync=False, deproj_dbeta_sync=False,
                 fiducial_beta=None, fiducial_T_dust=None, fiducial_beta_sync=None,
                 odir=None, norm_params=None, score_params=None,
                 coadd_equiv_crosses=True, apply_highpass_filter=True):

        self.lmax = data_dict['lmax']
        self.lmin = data_dict['lmin']
        self.nside = data_dict['nside']
        self.nsplit = data_dict['nsplit']
        self.delta_ell = data_dict['delta_ell']
        self.pyilcdir = pyilcdir
        self.use_dust_map = use_dust_map
        self.use_dbeta_map = use_dbeta_map
        self.use_sync_map = use_sync_map
        self.use_dbeta_sync_map = use_dbeta_sync_map        
        self.deproj_dust = deproj_dust
        self.deproj_dbeta = deproj_dbeta
        self.deproj_sync = deproj_sync
        self.deproj_dbeta_sync = deproj_dbeta_sync        
        self.fiducial_beta = fiducial_beta
        self.fiducial_T_dust = fiducial_T_dust
        self.fiducial_beta_sync = fiducial_beta_sync
        self.odir = odir
        self.bins = np.arange(self.lmin, self.lmax, self.delta_ell)
        self.coadd_equiv_crosses = coadd_equiv_crosses

        self.cov_scalar_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_nobb.dat'), self.lmax)
        self.cov_tensor_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_r1.dat'), self.lmax)

        self.minfo = map_utils.MapInfo.map_info_healpix(self.nside)
        self.ainfo = curvedsky.alm_info(self.lmax)

        freq_strings = data_dict['freq_strings']

        beam_fwhms = [self.get_beam_fwhms(fstr) for fstr in freq_strings]
        # We re-order the freq_strings based on FWHM because pyilc requires that.
        freq_strings_ordered = np.asarray(freq_strings)[np.argsort(np.asarray(beam_fwhms))][::-1]
        self.freq_strings = [str(fstr) for fstr in freq_strings_ordered]
        self.beam_fwhms = [self.get_beam_fwhms(fstr) for fstr in self.freq_strings]
        self.freqs = [self.get_freqs(fstr) for fstr in self.freq_strings]        
        assert np.all(np.asarray(self.freqs) > 1e9), 'Frequencies have to be in Ghz.'
        self.nfreq = len(self.freqs)
        
        self.b_ells = self.get_gaussian_beams(self.beam_fwhms, self.lmax)
        if apply_highpass_filter:
            self.highpass_filter = get_highpass_filter(
                self.lmin, self.lmax, data_dict['highpass_delta_ell'])
        else:
            self.highpass_filter = None

        if pyilcdir:
            self.ncomp = 1
            if self.use_dust_map: self.ncomp += 1
            if self.use_dbeta_map: self.ncomp += 1
            if self.use_sync_map: self.ncomp += 1
            if self.use_dbeta_sync_map: self.ncomp += 1
            
            self.sels_to_coadd = get_coadd_sels(self.nsplit, self.ncomp)
            self.size_data = len(self.sels_to_coadd) * (self.bins.size - 1)
            if self.use_dust_map or self.deproj_dust or self.deproj_dbeta:
                assert self.fiducial_beta is not None
                assert self.fiducial_T_dust is not None
            if self.use_sync_map or self.deproj_sync or self.deproj_dbeta_sync:
                assert self.fiducial_beta_sync is not None
        else:
            self.sels_to_coadd = get_coadd_sels(self.nsplit, self.nfreq)
            self.size_data = len(self.sels_to_coadd) * (self.bins.size - 1)
            
        self.sensitivity_mode = data_dict['sensitivity_mode']
        self.lknee_mode = data_dict['lknee_mode']
        self.noise_cov_ell = np.ones((self.nfreq, 2, 2, self.lmax + 1))

        for fidx, fstr in enumerate(self.freq_strings):

            # We scale the noise with the number of splits.
            self.noise_cov_ell[fidx] = np.eye(2)[:,:,np.newaxis] * self.get_noise_ps(fstr) \
                * self.nsplit

        # Fixed parameters.
        self.freq_pivot_dust = fixed_params_dict['freq_pivot_dust']
        self.freq_pivot_sync = fixed_params_dict.get('freq_pivot_sync')
        assert self.freq_pivot_dust > 1e9, "Freq pivot dust has to be in GHz."
        if self.freq_pivot_sync is not None:
            assert self.freq_pivot_sync > 1e9, "Freq pivot sync has to be in GHz."
        self.temp_dust = fixed_params_dict['temp_dust']

        self.norm_params = norm_params
        if self.norm_params and pyilcdir is None:

            # UPDATE WITH SYNC.
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

            # UPDATE WITH SYNC.
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

    def get_noise_ps(self, fstr):
        '''
        Return the BB noise power spectrum for a given band.

        Parameters
        ----------
        fstr : str
            Frequency band identifier.

        Return
        ------
        n_ell : (nell) array
            BB noise power spectrum.
        '''

        if fstr.startswith('f'):            
            return so_utils.get_sat_noise(
                fstr, self.sensitivity_mode, self.lknee_mode, self.lmax)
        if fstr.startswith('p'):
            return planck_utils.get_planck_noise(fstr, self.lmax)
        if fstr.startswith('w'):
            return wmap_utils.get_wmap_noise(fstr, self.lmax)        
        else:
            raise ValueError(f'{fstr=} not recognized')
            
    @staticmethod
    def get_freqs(fstr):
        '''
        Return central frequency.

        Parameters
        ----------
        fstr : str
            Frequency band identifier.

        Return
        ------
        central_freq : float
            Central frequency in Hz.
        '''

        if fstr.startswith('f'):
            return so_utils.sat_central_freqs[fstr]
        elif fstr.startswith('p'):
            return planck_utils.planck_central_freqs[fstr]
        elif fstr.startswith('w'):
            return wmap_utils.wmap_central_freqs[fstr]
        else:
            raise ValueError(f'{fstr=} not recognized')

    @staticmethod
    def get_beam_fwhms(fstr):
        '''
        Return beam FWHM.

        Parameters
        ----------
        fstr : str
            Frequency band identifier.

        Return
        ------
        fwhm : float
            FWHM in arcmin.
        '''

        if fstr.startswith('f'):
            return so_utils.sat_beam_fwhms[fstr]
        elif fstr.startswith('p'):
            return planck_utils.planck_beam_fwhms[fstr]
        elif fstr.startswith('w'):
            return wmap_utils.wmap_beam_fwhms[fstr]
        else:
            raise ValueError(f'{fstr=} not recognized')
        
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

        cov_ell = spectra_utils.apply_beam_to_freq_cov(cov_ell, self.b_ells)

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

    def draw_data(self, r_tensor, A_lens, A_d_BB, alpha_d_BB, beta_dust,
                  seed, amp_beta_dust=None, gamma_beta_dust=None, A_s_BB=None,
                  alpha_s_BB=None, beta_sync=None,
                  amp_beta_sync=None, gamma_beta_sync=None, rho_ds=None):
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
        amp_beta_dust : float, optional
            Amplitude of dust beta power spectrum at pivot multipole.
        gamma_beta_dust : float, optional
            Tilt of dust beta power spectrum.
        A_s_BB : float
            Synchrotron amplitude.
        alpha_s_BB : float
            Synchrotron spatial power law index.
        beta_sync : float
            Synchrotron frequency power law index.
        amp_beta_sync : float, optional
            Amplitude of synchrotron beta power spectrum at pivot multipole.
        gamma_beta_sync : float, optional
            Tilt of synchrotron beta power spectrum.
        rho_ds : float, optional
            Correlation coefficient between dust and synchroton amplitudes.

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
            self.cov_scalar_ell, self.cov_tensor_ell, self.b_ells, self.minfo, self.ainfo,
            amp_beta_dust=amp_beta_dust, gamma_beta_dust=gamma_beta_dust,
            A_s_BB=A_s_BB, alpha_s_BB=alpha_s_BB, beta_sync=beta_sync,
            freq_pivot_sync=self.freq_pivot_sync, amp_beta_sync=amp_beta_sync,
            gamma_beta_sync=gamma_beta_sync, rho_ds=rho_ds,
            signal_filter=self.highpass_filter)

        if self.pyilcdir:
            # build NILC B-mode maps.
            B_maps = np.zeros((self.nsplit, self.nfreq, self.minfo.npix))
            tmp_alm = np.zeros((2, self.ainfo.nelem), dtype=np.complex128) # E, B.
            for split in range(self.nsplit):
                for f, freq_str in enumerate(self.freq_strings):
                    sht.map2alm(omap[split,f], tmp_alm, self.minfo, self.ainfo, 2)
                    sht.alm2map(tmp_alm[1], B_maps[split,f], self.ainfo, self.minfo, 0)

            B_maps *= 1e-6 # Convert to K because pyilc assumes that input is in K.

            map_tmpdir = nilc_utils.write_maps(B_maps, output_dir=self.odir)
            nilc_maps = nilc_utils.get_nilc_maps(
                self.pyilcdir, map_tmpdir, self.nsplit, self.nside, self.fiducial_beta,
                self.fiducial_T_dust, self.freq_pivot_dust, self.freqs,
                self.beam_fwhms, use_dust_map=self.use_dust_map, use_dbeta_map=self.use_dbeta_map,
                use_sync_map=self.use_sync_map, use_dbeta_sync_map=self.use_dbeta_sync_map,
                deproj_dust=self.deproj_dust, deproj_dbeta=self.deproj_dbeta,
                deproj_sync=self.deproj_sync, deproj_dbeta_sync=self.deproj_dbeta_sync,
                fiducial_beta_sync=self.fiducial_beta_sync, freq_pivot_sync=self.freq_pivot_sync,
                output_dir=self.odir, remove_files=True, debug=False)

            spectra = estimate_spectra_nilc(nilc_maps, self.minfo, self.ainfo)

        else:
            spectra = estimate_spectra(omap, self.minfo, self.ainfo)

        if self.coadd_equiv_crosses:
            # coadd spectra
            ncomps = self.nfreq if self.pyilcdir is None else self.ncomp
            spectra = coadd(spectra, self.sels_to_coadd)

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

    @staticmethod
    def get_gaussian_beams(fwhms, lmax):
        '''
        Return Gaussian harmonic beam functions.

        Parameters
        ----------
        fwhms : (nfreq,) array-like
            List of FWHM values in arcmin.
        lmax : int
            Max multipole of output.

        Returns
        -------
        b_ells : (nfreq, lmax + 1) array
            Beam functions.
        '''

        fwhms = np.atleast_1d(fwhms)
        nfreq = len(fwhms)

        out = np.zeros((nfreq, lmax + 1))

        for fidx, fwhm in enumerate(fwhms):
            out[fidx] = hp.gauss_beam(np.radians(fwhm / 60), lmax=lmax)

        return out

def get_delta_beta_cl(amp, gamma, lmax, ell_0=1, ell_cutoff=1):
    '''
    Returns power spectrum for spectral index fluctuations.

    Parameters
    ----------
    amp : float
        Amplitude at pivot multipole.
    gamma : float
        Tilt of power spectrum.
    lmax : int
        Maximum multipole
    ell_0 : int, optional
        Pivot multipole.
    ell_cutoff : int, optional
        Multipole below which the power spectrum will be zero.

    Returns
    -------
    c_ell_beta : (nell) array
        Beta power spectrum
    '''

    ells = np.arange(lmax + 1)
    ind_above = np.where(ells >= ell_cutoff)[0]
    cls = np.zeros(len(ells))
    cls[ind_above] = amp * (ells[ind_above] / ell_0) ** gamma

    return cls

def get_beta_map(minfo, ainfo, beta0, amp, gamma, seed, ell_0=1, ell_cutoff=1):
    '''
    Returns realization of the spectral index map.

    Parameters
    ----------
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    beta0 : float
        Monopole of the beta map.
    amp : float
        Amplitude at pivot multipole.
    gamma : float
        Tilt of power spectrum.
    seed : numpy.random._generator.Generator object or int
        Random number generator or seed for new random number generator.
    ell_0 : int, optional
        Pivot multipole.
    ell_cutoff : int, optional
        Multipole below which the power spectrum will be zero.

    Returns
    -------
    beta_map : (npix) array
        Beta map, including monopole of beta.
    '''

    seed = np.random.default_rng(seed=seed)

    cls = get_delta_beta_cl(amp, gamma, ainfo.lmax, ell_0, ell_cutoff)
    alm_beta = alm_utils.rand_alm(cls, ainfo, seed, dtype=np.complex128)

    map_beta = np.zeros((minfo.npix))
    sht.alm2map(alm_beta, map_beta, ainfo, minfo, 0)

    return map_beta + beta0

def gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
             r_tensor, A_lens, freqs, seed, nsplit, cov_noise_ell,
             cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
             amp_beta_dust=None, gamma_beta_dust=None, A_s_BB=None,
             alpha_s_BB=None, beta_sync=None, freq_pivot_sync=None,
             amp_beta_sync=None, gamma_beta_sync=None, rho_ds=None,
             signal_filter=None):
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
    b_ells : (nfreq, nell) array
        Beam for each frequency.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    amp_beta_dust : float, optional
        Amplitude of dust beta power spectrum at pivot multipole.
    gamma_beta_dust : float, optional
        Tilt of dust beta power spectrum.
    A_s_BB : float
        Synchrotron amplitude.
    alpha_s_BB : float
        Synchrotron spatial power law index.
    beta_sync : float
        Synchrotron frequency power law index.
    freq_pivot_sync: float
        Pivot frequency for the synchrotron frequency power law.
    amp_beta_sync : float, optional
        Amplitude of synchrotron beta power spectrum at pivot multipole.
    gamma_beta_sync : float, optional
        Tilt of synchrotron beta power spectrum.
    rho_ds : float, optional
        Correlation coefficient between dust and synchroton amplitudes.
    signal_filter : (nell) array. optional
        Harmonic filter that is applied to the signal (similar to beam).

    Returns
    -------
    data : (nsplit, nfreq, npol, npix)
        Simulated data.
    '''

    nfreq = len(freqs)
    out = np.zeros((nsplit, nfreq, 2, minfo.npix))

    # Spawn rng for dust and noise.
    seed = np.random.default_rng(seed)
    rngs = seed.spawn(3 + nsplit)
    rng_cmb = rngs[0]
    rng_dust = rngs[1]
    rng_beta = rngs[2]
    rngs_noise = rngs[3:]

    # Generate the CMB spectra.
    cov_ell = spectra_utils.get_combined_cmb_spectrum(
        r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell)
    lmax = cov_ell.shape[-1] - 1
    assert ainfo.lmax == lmax
    
    #cov_dust_ell = np.zeros_like(cov_ell)
    if A_s_BB is not None:
        ncomp_fg = 2
    else:
        ncomp_fg = 1        
    cov_fg_ell = np.zeros((ncomp_fg, ncomp_fg, lmax + 1))

    # Generate frequency-independent signal, scale with frequency later.
    cov_fg_ell[0,0] = spectra_utils.get_ell_shape(lmax, alpha_d_BB, ell_pivot=80)
    cov_fg_ell[0,0] *= A_d_BB

    if A_s_BB is not None:
        cov_fg_ell[1,1] = spectra_utils.get_ell_shape(lmax, alpha_s_BB, ell_pivot=80)
        cov_fg_ell[1,1] *= A_s_BB

        if rho_ds is not None:
            cov_fg_ell[0,1] = rho_ds * np.sqrt(cov_fg_ell[0,0] * cov_fg_ell[1,1])
            cov_fg_ell[1,0] = cov_fg_ell[0,1]
        
    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    fg_alm = alm_utils.rand_alm(cov_fg_ell, ainfo, rng_dust, dtype=np.complex128)    

    if A_s_BB is not None:
        if (gamma_beta_dust != gamma_beta_sync) and None in (gamma_beta_dust, gamma_beta_sync):
            # Raises error only if one of two is None.
            raise ValueError('We only support either both dust and sync gammas or none.')
    
    if gamma_beta_dust is not None:
        assert amp_beta_dust is not None

        # Create real-space dust map.
        alm_tmp = np.zeros((2, ainfo.nelem), dtype=np.complex128)
        alm_tmp[1] = fg_alm[0]
        dust_map = np.zeros((2, minfo.npix))
        sht.alm2map(alm_tmp, dust_map, ainfo, minfo, 2)

        # Generate the dust beta map.
        beta_dust = get_beta_map(minfo, ainfo, beta_dust, amp_beta_dust, gamma_beta_dust, rng_beta)
        
        if A_s_BB is not None:
            alm_tmp[1] = fg_alm[1]
            sync_map = np.zeros((2, minfo.npix))
            sht.alm2map(alm_tmp, sync_map, ainfo, minfo, 2)            
            beta_sync = get_beta_map(minfo, ainfo, beta_sync, amp_beta_sync, gamma_beta_sync, rng_beta)
        else:
            sync_map, beta_sync = None, None

        gen_data_per_freq = lambda freq, cov_noise_ell, b_ell: _gen_data_per_freq_gamma(
            freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
            cmb_alm, dust_map, nsplit, rngs_noise, ainfo, minfo, b_ell, sync_map=sync_map,
            beta_sync=beta_sync, freq_pivot_sync=freq_pivot_sync)

    else:
        gen_data_per_freq = lambda freq, cov_noise_ell, b_ell: _gen_data_per_freq_simple(
            freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
            cmb_alm, fg_alm, nsplit, rngs_noise, ainfo, minfo, b_ell, beta_sync=beta_sync,
            freq_pivot_sync=freq_pivot_sync)

    for fidx, freq in enumerate(freqs):
        
        b_ell = b_ells[fidx]
        if signal_filter is not None:
            b_ell = b_ell * signal_filter
        out[:,fidx,:,:] = gen_data_per_freq(freq, cov_noise_ell[fidx], b_ell)

    return out

def _gen_data_per_freq_simple(freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
                              cmb_alm, fg_alm, nsplit, rngs_noise, ainfo, minfo, b_ell,
                              beta_sync=None, freq_pivot_sync=None):
    '''
    Generate data for a given frequency, using a data model with constant beta.

    Parameters
    ----------
    freq : float
        Effective freq of passband in Hz.
    cov_noise_ell : (npol, npol, nell) array
        Noise covariance matrix.
    beta_dust : float
        Dust frequency power law index.
    temp_dust : float
        Dust temperature for the blackbody part of the model.
    freq_pivot_dust : float
        Pivot frequency for the frequency power law in Hz.
    cmb_alm : (2, nelem) complex array
        CMB E- and B-mode alms.
    fg_alm : (1, nelem) or (2, nelem) complex array
        Dust (and possibly synchrotron) B-mode amplitude alms.
    nsplit : int
        Number of splits of the data that have independent noise.
    rngs_noise : array-like of numpy.random._generator.Generator object
        Random number generators for per-split noise.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    b_ell : (lmax + 1) array
        Beam for this frequency.
    beta_sync : float, optional
        Synchrotron frequency power law index.
    freq_pivot_sync : float, optional
        Pivot frequency for the synchrotron frequency power law in Hz.

    Returns
    -------
    out : (nsplit, 2, npix) array
        Stokes Q and U maps for each split.
    '''

    out = np.zeros((nsplit, 2, minfo.npix))

    dust_factor = np.sqrt(spectra_utils.get_sed_dust(
        freq, beta_dust, temp_dust, freq_pivot_dust))
    dust_factor *= spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_dust)
    
    signal_alm = cmb_alm.copy()
    signal_alm[1] += fg_alm[0] * dust_factor

    ncomp_fg = fg_alm.shape[0]
    if ncomp_fg == 2:
        assert not None in (beta_sync, freq_pivot_sync)
        sync_factor = np.sqrt(spectra_utils.get_sed_sync(
            freq, beta_sync, freq_pivot_sync))
        sync_factor *= spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_sync)                    
        signal_alm[1] += fg_alm[1] * sync_factor
    
    # Apply beam.
    signal_alm = alm_c_utils.lmul(signal_alm, b_ell, ainfo, inplace=False)

    for sidx in range(nsplit):

        data_alm = signal_alm + alm_utils.rand_alm(
            cov_noise_ell, ainfo, rngs_noise[sidx], dtype=np.complex128)
        data_alm = np.asarray(data_alm, dtype=np.complex128)
        sht.alm2map(data_alm, out[sidx], ainfo, minfo, 2)

    return out

def _gen_data_per_freq_gamma(freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
                             cmb_alm, dust_map, nsplit, rngs_noise, ainfo, minfo, b_ell,
                             sync_map=None, beta_sync=None, freq_pivot_sync=None):
    '''
    Generate data for a given frequency, using a data model with varying beta.

    Parameters
    ----------
    freq : float
        Effective freq of passband in Hz.
    cov_noise_ell : (npol, npol, nell) array
        Noise covariance matrix.
    beta_dust : (npix) array
        Beta map, including monopole of beta.
    temp_dust : float
        Dust temperature for the blackbody part of the model.
    freq_pivot_dust : float
        Pivot frequency for the frequency power law.
    cmb_alm : (2, nelem) complex array
        CMB E- and B-mode alms.
    dust_map : (2, nelem) array
        Dust amplitude Stokes Q and U maps.
    nsplit : int
        Number of splits of the data that have independent noise.
    rngs_noise : array-like of numpy.random._generator.Generator object
        Random number generators for per-split noise.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    b_ell : (lmax + 1) array
        Beam for this frequency.
    sync_map : (2, nelem) array, optional
        Synchrotron amplitude Stokes Q and U maps.
    beta_sync : (npix) array
        Beta synchrotron map, including monopole of beta.
    freq_pivot_sync : float
        Pivot frequency for the synchrotron frequency power law.

    Returns
    -------
    out : (nsplit, 2, npix) array
        Stokes Q and U maps for each split.
    '''

    out = np.zeros((nsplit, 2, minfo.npix))

    # Apply spatially varying SED scaling in real space.
    sed_map = spectra_utils.get_sed_dust(freq, beta_dust, temp_dust, freq_pivot_dust)
    scaled_dust_map = dust_map * np.sqrt(sed_map)
    scaled_dust_map *= spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_dust)

    fg_map = scaled_dust_map
    
    if sync_map is not None:
        sed_sync_map = spectra_utils.get_sed_sync(freq, beta_sync, freq_pivot_sync)
        scaled_sync_map = sync_map * np.sqrt(sed_sync_map)
        scaled_sync_map *= spectra_utils.get_g_fact(freq) / spectra_utils.get_g_fact(freq_pivot_sync)
        fg_map += scaled_sync_map
        
    # Apply beam.
    fg_alm = np.zeros(cmb_alm.shape, dtype=np.complex128)
    sht.map2alm(fg_map, fg_alm, minfo, ainfo, 2)
    signal_alm = cmb_alm + fg_alm
    signal_alm = alm_c_utils.lmul(signal_alm, b_ell, ainfo, inplace=False)

    for sidx in range(nsplit):

        data_alm = signal_alm + alm_utils.rand_alm(
            cov_noise_ell, ainfo, rngs_noise[sidx], dtype=np.complex128)
        data_alm = np.asarray(data_alm, dtype=np.complex128)
        sht.alm2map(data_alm, out[sidx], ainfo, minfo, 2)

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
    (nsplits * nfreq) x (nsplits * nfreq) cross-spectrum matrix,
    while excluding combinations that share split indices.

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
        Output BB spectra. See `get_tri_indices`.
    '''

    nsplit = imap.shape[0]
    nfreq = imap.shape[1]

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
    and components, while excluding combinations with the same split indices.

    Parameters
    ----------
    imap : (nsplit, ncomp, npix)
        Input B-mode maps.
    minfo : optweight.map_utils.MapInfo object
        Geometry of output map.
    ainfo : pixell.curvedsky.alm_info object
        Layout of spherical harmonic coefficients.

    Returns
    -------
    out : (ntri, 1, lmax + 1)
        Output BB spectra. See `get_tri_indices`.
    '''

    nsplit = imap.shape[0]
    ncomp = imap.shape[1]

    ntri = get_ntri(nsplit, ncomp)
    out = np.zeros((ntri, 1, ainfo.lmax + 1))

    alm = np.zeros((nsplit, ncomp, ainfo.nelem), dtype=np.complex128)
    sht.map2alm(imap.astype(np.float64, copy=False), alm, minfo, ainfo, 0)

    tri_indices = get_tri_indices(nsplit, ncomp)
    for idx, (sidx1, cidx1, sidx2, cidx2) in enumerate(tri_indices):
        out[idx,0] = ainfo.alm2cl(alm[sidx1,cidx1], alm2=alm[sidx2,cidx2])

    return out

def get_coadd_sels(nsplits, ncomps):
    '''
    Find list of index lists that will coadd equivalent cross-spectra in
    the datavector, i.e. comp1 x comp2 and comp2 x comp1.

    Parameters
    ----------
    nsplits: int
        Number of splits.
    ncomps: int
        Number of frequencies or number of components.

    Returns
    -------
    sels_to_coadd : (n_unique) list of index arrays.
        List of index arrays containing elements in data vector to coadd.
    '''

    sidx1, cidx1, sidx2, cidx2 = get_tri_indices(nsplits, ncomps).T
    ntri = get_ntri(nsplits, ncomps)
    assert sidx1.size == ntri

    # Extract the unique cidx1, cidx2 combinations and put them in unique_combs.
    pairs = [tuple(sorted((cidx1[i], cidx2[i]))) for i in range(ntri)]
    unique_combs = sorted(set(pairs))

    # List of length len(unique_combs) where each elements is another list of indices.
    sels_to_coadd = []
    for comb in unique_combs:
        sel = [i for i in range(ntri) if tuple(sorted((cidx1[i], cidx2[i]))) == comb]
        sels_to_coadd.append(sel)

    return sels_to_coadd

def coadd(spec, sels_to_coadd):
    '''
    Coadd cross-spectra from different splits

    Parameters
    ----------
    spec: (ntri, 1, lmax + 1) array
        Input spectra to coadd.
    sels_to_coadd : (n_unique) list of index arrays.
        List of index arrays containing elements in data vector to coadd.
        See `get_coadd_sels`.

    Returns
    -------
    final_spectra: (len(sels_to_coadd), 1, ellmax + 1) array
        Coadded spectra.
    '''

    nell = spec.shape[-1]
    final_spectra = np.zeros((len(sels_to_coadd), 1, nell))

    for idx, selections in enumerate(sels_to_coadd):
        final_spectra[idx,0,:] = spec[selections,0,:].mean(axis=0)

    return final_spectra

def get_final_data_vector(spec, bins, lmin, lmax):
    '''
    Create data vector by binning and flattening spectra.

    Parameters
    ----------
    spec : (len(sels_to_coadd), 1, ellmax + 1)
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

def get_highpass_filter(lmin, lmax, delta_ell):
    '''
    Return a filter that smoothly transitions from 0 below lmin - delta_ell
    to 1 above lmin.

    Parameters
    ----------
    lmin : int
        Multipole above which the filter is 1.
    lmax : int
        Maximum multipole.
    delta_ell : int
        Wdith of filter below lmin.

    Returns
    -------
    f_ell : (lmax + 1) array
        Filter.
    '''

    f_ell = np.ones(lmax + 1)
    assert delta_ell > 0
    assert (lmin - delta_ell) >= 0

    ells = np.arange(delta_ell)    
    transition = 0.5 * (1 + np.cos(ells * np.pi / delta_ell))
    f_ell[lmin-delta_ell+1:lmin+1] = transition[::-1]
    f_ell[:lmin-delta_ell+1] = 0

    return f_ell
