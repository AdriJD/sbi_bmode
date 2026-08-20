'''
Utils for simulating data using a fidutial lensing field.
'''
import os

import numpy as np
import healpy as hp
from pixell import curvedsky
from optweight import alm_utils, sht, map_utils, mat_utils, alm_c_utils
import healpy as hp
from jax import grad
import jax.numpy as jnp

import lenspyx
from lenspyx import lensing

from sbi_bmode import (spectra_utils, so_utils, lenshILC_utils, likelihood_utils,
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
    wavelet_type : str, optonal
        Type of wavelets. For this pipeline specifically, it's called "hILC_lens".        
    fiducial_beta: float, optional
        Use this value for beta when building mixing matrix A for delensing operator N.
    fiducial_T_dust: float, optional
        Use this value for T_dust when building mixing matrix A for delensing operator N.
    fiducial_beta_sync: float, optional
        Use this value for beta_synchrotron when building mixing matrix A for delensing operator N.
    odir: str
        Path to output directory
    score_params: dict, optional
        Parameters of fiducial model where the score is evaluted for score
        compression.
    apply_highpass_filter: bool, optional
        Filter out signal modes below lmin in the simulations.
    mask_file : str
        Path to .fits file containing mask in HEALPix format.
    fg_template_files : dict, optional
        Dictionary such that d['dust']['f090'] returns a path to set of B-mode spherical
        harmonic coefficients that represent a foreground template. Possible outer keys
        are 'dust' and 'sync'. Inner keys have to match the band names in the config.
    '''

    def __init__(self, specdir, data_dict, fixed_params_dict,
                 wavelet_type='hILC_lens',
                 fiducial_beta=None, fiducial_T_dust=None, fiducial_beta_sync=None,
                 odir=None, score_params=None,
                 apply_highpass_filter=True, mask_file=None,
                 fg_template_files=None):

        self.lmax = data_dict['lmax']
        self.lmin = data_dict['lmin']
        self.nside = data_dict['nside']
        self.nsplit = data_dict['nsplit']
        self.delta_ell = data_dict['delta_ell']

        self.wavelet_type = wavelet_type
        self.fiducial_beta = fiducial_beta
        self.fiducial_T_dust = fiducial_T_dust
        self.fiducial_beta_sync = fiducial_beta_sync
        self.odir = odir

        self.bins = np.arange(self.lmin, self.lmax, self.delta_ell)

        self.cov_scalar_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_nobb.dat'), self.lmax)
        self.cov_tensor_ell = spectra_utils.get_cmb_spectra(
            opj(specdir, 'camb_lens_r1.dat'), self.lmax)

        self.minfo = map_utils.MapInfo.map_info_healpix(self.nside)
        self.ainfo = curvedsky.alm_info(self.lmax)

        freq_strings = data_dict['freq_strings']
        beam_fwhms = [self.get_beam_fwhms(fstr) for fstr in freq_strings]
        
        self.freq_strings = list(data_dict['freq_strings'])
        self.beam_fwhms = [self.get_beam_fwhms(fstr) for fstr in self.freq_strings]
        self.freqs = [self.get_freqs(fstr) for fstr in self.freq_strings]        
        assert np.all(np.asarray(self.freqs) > 1e9), 'Frequencies have to be in Ghz.'
        self.nfreq = len(self.freqs)

        if fg_template_files is not None:
            self.fg_templates = {}
            for fstr in self.freq_strings:
                self.fg_templates[fstr] = hp.read_alm(
                    fg_template_files['dust'][fstr]).astype(np.complex128)
                self.fg_templates[fstr] += hp.read_alm(
                    fg_template_files['sync'][fstr]).astype(np.complex128)
        
        self.b_ells = self.get_gaussian_beams(self.beam_fwhms, self.lmax)
        if apply_highpass_filter:
            self.highpass_filter = get_highpass_filter(
                self.lmin, self.lmax, data_dict['highpass_delta_ell'])
        else:
            self.highpass_filter = None

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

        self.score_params = score_params
        if self.score_params:

            self.score_model = np.asarray(self.get_signal_spectra(
                score_params['r_tensor'], score_params['A_d_BB'],
                score_params['alpha_d_BB'], score_params['beta_dust']))
            noise_spectra = self.get_noise_spectra()
            cov = likelihood_utils.get_cov(
                self.score_model, noise_spectra, self.bins, self.lmin,
                self.lmax, self.nsplit, self.nfreq)
            tri_indices = get_tri_indices(self.nsplit, self.nfreq)
            icov = mat_utils.matpow(cov, -1, return_diag=True)

            score_params_arr = jnp.asarray(
                [score_params['r_tensor'], score_params['A_d_BB'],
		 score_params['alpha_d_BB'], score_params['beta_dust']])

            def get_loglike(params, data):

                data = data.reshape(tri_indices.shape[0], -1)
                model = self.get_signal_spectra(*params)
                loglike = likelihood_utils.loglike(model, data, icov, tri_indices)

                return loglike

            self.grad_logdens = grad(get_loglike, argnums=0)
            self.score_compress = lambda x: self.grad_logdens(score_params_arr, x)

        if mask_file:
            self.mask = hp.read_map(mask_file).astype(np.float64)
        else:
            self.mask = None

        #initiate mixing matrix A
        self.A = lenshILC_utils.build_A(
            freqs=self.freqs,
            fiducial_beta_dust=self.fiducial_beta,
            fiducial_T_dust=self.fiducial_T_dust,
            freq_pivot_dust=self.freq_pivot_dust,
            fiducial_beta_sync=self.fiducial_beta_sync,
            freq_pivot_sync=self.freq_pivot_sync,
            default_T_dust=self.temp_dust,
            include_dust_beta1=False,
            include_sync_beta1=False,
        )
        
        if self.A.shape[0] != self.nfreq:
            raise ValueError(
                "lenshILC mixing matrix and simulated data have inconsistent "
                "frequency channels: "
                f"A.shape={self.A.shape}, self.nfreq={self.nfreq}."
            )
    
            
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
        
    def get_signal_spectra(self, r_tensor, A_d_BB, alpha_d_BB, beta_dust,
                           A_s_BB=None, alpha_s_BB=None, beta_sync=None, rho_ds=None):
        '''
        Generate binned signal frequency cross spectra.

        Parameters
        ----------
        r_tensor : float
            Tensor-to-scalar ratio.
        A_d_BB : float
            Amplitude of dust power spectrum.
        alpha_d_BB : float
            Power law index of dust power spectrum.
        beta_dust : float
            Power law index of dust SED.
        A_s_BB : float, optional
            Amplitude of synchrotron power spectrum.
        alpha_s_BB : float, optional
            Power law index of synchrotron power spectrum.
        beta_dust : float, optonal
            Power law index of synchrtron SED.
        rho_ds : float, optional
            Cross correlation coefficient of dust and synchrotron angular power spectra.

        Returns
        -------
        cov_bin : (nfreq, nfreq, nbin) array
            Signal frequency cross spectra.
        '''
        
        cov_ell = spectra_utils.get_dust_spectra(
            A_d_BB, alpha_d_BB, self.lmax, self.freqs, beta_dust, self.temp_dust,
            self.freq_pivot_dust)

        if A_s_BB is not None:
            cov_ell = cov_ell.at[:].add(spectra_utils.get_sync_spectra(
                A_s_BB, alpha_s_BB, self.lmax, self.freqs, beta_sync, self.freq_pivot_sync))

        if rho_ds is not None:
            cov_ell = cov_ell.at[:].add(spectra_utils.get_dust_sync_cross_spectra(
                rho_ds, A_d_BB, alpha_d_BB, A_s_BB, alpha_s_BB, self.lmax, self.freqs,
                beta_dust, self.temp_dust, beta_sync, self.freq_pivot_dust,
                self.freq_pivot_sync))
        
        # Only adding the BB part because `get_dust_spectra` only produces BB.
        cov_ell = cov_ell.at[:].add(spectra_utils.get_combined_cmb_spectrum(
            r_tensor, self.cov_scalar_ell, self.cov_tensor_ell)[1,1])

        cov_ell = spectra_utils.apply_beam_to_freq_cov(cov_ell, self.b_ells)

        cov_bin = spectra_utils.bin_spectrum(
            cov_ell, np.arange(self.lmax+1), self.bins, use_jax=True)

        return cov_bin

    def get_noise_spectra(self, use_jax=False):
        '''
        Generate binned noise frequency cross spectra.
        
        Parameters
        ----------
        use_jax : bool, optional
            If set, use JAX backend.

        Returns
        -------
        cov_bin : (nfreq, nfreq, nbin) array
            Noise frequency cross spectra.
        '''

        out = np.zeros((self.nfreq, self.nfreq, self.lmax+1))
        out[:] = np.eye(self.nfreq)[:,:,np.newaxis] * self.noise_cov_ell[:,1,1]

        cov_bin = spectra_utils.bin_spectrum(
            out, np.arange(self.lmax+1), self.bins, use_jax=use_jax)

        return cov_bin

    def draw_data(self, r_tensor, A_d_BB, alpha_d_BB, beta_dust, 
                # Fiducial SED parameters used by reconstruction.
                  seed, amp_beta_dust=None, gamma_beta_dust=None, 
                  A_s_BB=None, alpha_s_BB=None, beta_sync=None,
                  amp_beta_sync=None, gamma_beta_sync=None,
                  rho_ds=None, draw_from_fg_template=False,

                  return_maps=False,
                  return_unbinned_spectra=False,
                  use_lensing_operator=False,

                  phi_planck_alm=None,
                  phi_signal_cl=None,
                  phi_noise_cl=None,
                  
                  lens_seed=None,
                  lens_dlmax=500,
                  lens_epsilon=1e-6,
                  lens_components=(0,),
                  cg_bin_size=20,
                  cg_maxiter=500,
                  cg_tol=1e-12):
        '''
        Draw data realization. Added cg criteria.

        Parameters
        ----------
        r_tensor : float
            Tensor-to-scalar ratio.
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
        draw_from_fg_template : bool, optional
            If set, draw foregrounds from provided templates.
            
        return_maps : bool, optional
            If True, keep the simulated Q/U maps in the output dictionary under
            out_dict["maps"]. The shape is (nsplit, nfreq, 2, npix).
            This is mainly useful for debugging and tests. The default is False.

        return_unbinned_spectra : bool, optional
            If True, return the CG reconstructed cmb BB spectra to 
            out_dict["recovered_cmb_cl_bb"]. The default is False. For testing.

        use_lensing_operator : bool, optional
            If True, construct a lensing operator using
            lenshILC_utils.generate_defl and
            lenshILC_utils.LensingOperator and use it inside Lmix.
            If False, use an identity lensing operator. The identity option is
            useful for fast smoke tests of the mixing, covariance, and CG
            reconstruction without lenspyx remapping. The default is False.

        lens_seed : int, optional
            Random seed used to generate the lensing deflection field when
            use_lensing_operator is True. If None, a seed is drawn from the
            main random number generator. Ignored when use_lensing_operator
            is False.

        lens_dlmax : int, optional
            Extra multipole range used when generating the lensing deflection
            field. Passed to lenshILC_utils.generate_defl as dlmax.
            Ignored when use_lensing_operator is False. The default is 500.

        lens_epsilon : float, optional
            Accuracy parameter passed to the lenspyx deflection object through
            lenshILC_utils.generate_defl. Ignored when use_lensing_operator is False. The default is 1e-6.

        lens_components : tuple of int, optional
            Indices of components to which the lensing operator is applied
            inside lenshILC_utils.Lmix. For example, (0,) means lens
            only the first component, typically CMB. Use () to apply no
            lensing, which is appropriate when use_lensing_operator is
            False. The default is (0,).

        cg_bin_size : int, optional
            Multipole bin size used when estimating and regularizing the
            empirical inverse covariance in
            lenshILC_utils.compute_cinv_pol. Passed to
            lenshILC_utils.CGComponentReconstructor. The default is 20.

        cg_maxiter : int, optional
            Maximum number of conjugate-gradient iterations used by
            lenshILC_utils.CGComponentReconstructor.solve_components.
            The default is 500.

        cg_tol : float, optional
            Relative convergence tolerance for the conjugate-gradient solve.
            Passed to lenshILC_utils.CGComponentReconstructor. The default
            is 1e-12.

        Returns
        -------
        out_dict : dict
            Output dictionary with following key-value pairs:
                data : (ndata) array
                    Data realization.
                gamma_dust_ell : (lmax + 1) array, optional
                    Realization of the gamma_dust power spectrum
                gamma_sync_ell : (lmax + 1) array, optional
                    Realization of the gamma_sync power spectrum
        '''

        if seed == -1:
            seed = None
        seed = np.random.default_rng(seed=seed)
        
        A = self.A #Reading in predefined mixing matrix.
        if use_lensing_operator:
            if phi_planck_alm is None:
                raise ValueError("phi_planck_alm must be provided when use_lensing_operator=True.")
            if phi_signal_cl is None or phi_noise_cl is None:
                raise ValueError("phi_signal_cl and phi_noise_cl must be provided.")
            if lens_seed is None:
                lens_seed = int(seed.integers(0, 2**32 - 1))

            lmax_phi = self.lmax + lens_dlmax
            phi_cr_alm, phi_wf_alm, phi_residual_alm = lenshILC_utils.draw_constrained_phi(phi_planck_alm, phi_signal_cl, phi_noise_cl, lmax=lmax_phi, seed=lens_seed)

            defl_true = lenshILC_utils.phi_alm_to_defl(phi_cr_alm, nside=self.nside, lmax=lmax_phi, epsilon=lens_epsilon)
            defl_recon = lenshILC_utils.phi_alm_to_defl(phi_planck_alm, nside=self.nside, lmax=lmax_phi, epsilon=lens_epsilon)
            
            L_true = lenshILC_utils.LensingOperator(defl_true, self.lmax)
            L_recon = lenshILC_utils.LensingOperator(defl_recon, self.lmax)

        else:
            class IdentityLensing:
                def forward(self, x):
                    return np.asarray(x, dtype=np.complex128).copy()

                def adjoint(self, x):
                    return np.asarray(x, dtype=np.complex128).copy()

            L_true = IdentityLensing()
            L_recon = IdentityLensing()

        if draw_from_fg_template:
            out_dict = gen_data_fg_template(
                self.fg_templates, r_tensor, self.freq_strings,
                seed, self.nsplit, self.noise_cov_ell, self.cov_scalar_ell,
                self.cov_tensor_ell, self.b_ells, self.minfo, self.ainfo,
                signal_filter=self.highpass_filter, no_cmb_ee=(self.mask is not None))

        else:
            out_dict = gen_data(
                A_d_BB, alpha_d_BB, beta_dust, self.freq_pivot_dust, self.temp_dust,
                r_tensor, self.freqs, seed, self.nsplit, self.noise_cov_ell,
                self.cov_scalar_ell, self.cov_tensor_ell, self.b_ells, self.minfo, self.ainfo,
                amp_beta_dust=amp_beta_dust, gamma_beta_dust=gamma_beta_dust,
                A_s_BB=A_s_BB, alpha_s_BB=alpha_s_BB, beta_sync=beta_sync,
                freq_pivot_sync=self.freq_pivot_sync, amp_beta_sync=amp_beta_sync,
                gamma_beta_sync=gamma_beta_sync, rho_ds=rho_ds,
                signal_filter=self.highpass_filter, no_cmb_ee=(self.mask is not None),
                lensing_operator=L_true,
            ) #This is generated lensed - data.
            
        omap = out_dict["data"]

        if self.mask is not None:
            omap *= self.mask[None, None, None, :]

        if return_maps:
            out_dict["maps"] = omap.copy() # omap shape: (nsplit, nfreq, 2, npix)

      
        Lx = lenshILC_utils.Lmix(L=L_recon, A=A, lens_components=list(lens_components))
        
        d_alm_obs = np.zeros((self.nsplit, self.nfreq, 2, self.ainfo.nelem), dtype=np.complex128)
        
        sht.map2alm(omap.astype(np.float64, copy=False), d_alm_obs, self.minfo, self.ainfo, 2)
        
        cg_recons = lenshILC_utils.CGComponentReconstructor(bin_size=cg_bin_size, cg_maxiter=cg_maxiter, cg_tol=cg_tol)
        
        cg_recons.set_operator(A, Lx)
        
        component_alms = cg_recons.solve_components(d_alm_obs, self.ainfo)
        
        if component_alms.shape[:2] != (self.nsplit, A.shape[1]):
            raise RuntimeError(f"Unexpected reconstructed component shape: {component_alms.shape}.")
        
        out_dict["component_alms"] = component_alms
        out_dict["mixing_matrix"] = A
        
        spectra_cg = estimate_spectra_cg(component_alms, self.ainfo, component_idx=0)
        data_cg_bb = get_final_data_vector(spectra_cg[:, 1:2, :], self.bins)
        
        out_dict["data"] = data_cg_bb
        
        if return_unbinned_spectra:
            out_dict["recovered_cmb_cl_bb"] = spectra_cg[:, 1:2, :]
        
        return out_dict


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
    beta_cl : (lmax + 1) array
        Realization of the beta power spectrum.
    '''

    seed = np.random.default_rng(seed=seed)

    cls = get_delta_beta_cl(amp, gamma, ainfo.lmax, ell_0, ell_cutoff)
    assert cls.ndim == 1
    # To make sure output is (1, nelem)
    alm_beta = alm_utils.rand_alm(cls[np.newaxis,:], ainfo, seed, dtype=np.complex128)
    assert alm_beta.ndim == 2
    alm_beta[0,0] += np.sqrt(4 * np.pi) * beta0

    beta_cl = ainfo.alm2cl(alm_beta[0])
    
    map_beta = np.zeros((minfo.npix))
    sht.alm2map(alm_beta, map_beta, ainfo, minfo, 0)

    return map_beta, beta_cl

def gen_data_fg_template(fg_templates, r_tensor, freq_strings, seed, nsplit,
                         cov_noise_ell, cov_scalar_ell, cov_tensor_ell, b_ells,
                         minfo, ainfo, signal_filter=None, no_cmb_ee=False):
    '''
    Generate test sets using foreground templates.

    Parameters
    ----------
    fg_templates : dict
        Dictionary with fstr keys containing foreground B-mode alms.
    r_tensor : float
        Tensor-to-scalar ratio.
    freq_strings : array-like
        Identifiers, e.g. f090, for the frequency channels of the instrument.
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
    signal_filter : (nell) array. optional
        Harmonic filter that is applied to the signal (similar to beam).
    no_cmb_ee : bool, optional
        If set, set EE cmb constribution to zero. Added for backwards compatibiliy.

    Returns
    -------
    out_dict : dict
        Output dictionary with following key-value pairs:
            data : (nsplit, nfreq, npol, npix)
                Simulated data.
    '''

    nfreq = len(freq_strings)
    out = np.zeros((nsplit, nfreq, 2, minfo.npix))

    # Spawn rng for noise.
    seed = np.random.default_rng(seed)
    rngs = seed.spawn(1 + nsplit)
    rng_cmb = rngs[0]
    rngs_noise = rngs[1:]

    # Generate the CMB spectra.
    cov_ell = spectra_utils.get_combined_cmb_spectrum(
        r_tensor, cov_scalar_ell, cov_tensor_ell)
    lmax = cov_ell.shape[-1] - 1
    assert ainfo.lmax == lmax
            
    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    if no_cmb_ee:
        cmb_alm[0] = 0

    for fidx, fstr in enumerate(freq_strings):
        
        b_ell = b_ells[fidx]
        if signal_filter is not None:
            b_ell = b_ell * signal_filter
        out[:,fidx,:,:] = _gen_data_per_freq_fg_template(
            fstr, cov_noise_ell[fidx], cmb_alm, nsplit, rngs_noise, ainfo, minfo, b_ell,
            fg_templates)

    out_dict = {'data' : out}    

    return out_dict
    
def gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
    r_tensor, freqs, seed, nsplit, cov_noise_ell,
    cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
    amp_beta_dust=None, gamma_beta_dust=None, A_s_BB=None,
    alpha_s_BB=None, beta_sync=None, freq_pivot_sync=None,
    amp_beta_sync=None, gamma_beta_sync=None, rho_ds=None,
    signal_filter=None, no_cmb_ee=False, lensing_operator=None):
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
    no_cmb_ee : bool, optional
        If set, set EE cmb constribution to zero. Added for backwards compatibiliy.
    lensing_operator : object, optional
    Lensing operator applied to the simulated CMB realization before
    foreground mixing, beam convolution, and noise addition.

    Returns
    -------
    out_dict : dict
        Output dictionary with following key-value pairs:
            data : (nsplit, nfreq, npol, npix)
                Simulated data.
            gamma_dust_ell : (lmax + 1) array, optional
                Realization of the gamma_dust power spectrum
            gamma_sync_ell : (lmax + 1) array, optional
                Realization of the gamma_sync power spectrum
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
        r_tensor, cov_scalar_ell, cov_tensor_ell)
    lmax = cov_ell.shape[-1] - 1
    assert ainfo.lmax == lmax
    
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
        
    cmb_unlensed_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    
    if no_cmb_ee:
        cmb_unlensed_alm[0] = 0
    
    cmb_unlensed_bb_ell = ainfo.alm2cl(cmb_unlensed_alm[1], alm2=cmb_unlensed_alm[1])
    
    if lensing_operator is not None:
        cmb_alm = lensing_operator.forward(cmb_unlensed_alm)
    else:
        cmb_alm = cmb_unlensed_alm.copy()
    
    cmb_lensed_bb_ell = ainfo.alm2cl(cmb_alm[1], alm2=cmb_alm[1])
        
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
        beta_dust, gamma_dust_ell = get_beta_map(
            minfo, ainfo, beta_dust, amp_beta_dust, gamma_beta_dust, rng_beta)
        
        if A_s_BB is not None:
            alm_tmp[1] = fg_alm[1]
            sync_map = np.zeros((2, minfo.npix))
            sht.alm2map(alm_tmp, sync_map, ainfo, minfo, 2)            
            beta_sync, gamma_sync_ell = get_beta_map(
                minfo, ainfo, beta_sync, amp_beta_sync, gamma_beta_sync, rng_beta)
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
        
        gamma_dust_ell, gamma_sync_ell = None, None
        
    for fidx, freq in enumerate(freqs):
        
        b_ell = b_ells[fidx]
        if signal_filter is not None:
            b_ell = b_ell * signal_filter
        out[:,fidx,:,:] = gen_data_per_freq(freq, cov_noise_ell[fidx], b_ell)

    out_dict = {
        "data": out,
        "input_unlensed_cmb_cl_bb": cmb_unlensed_bb_ell,
        "input_lensed_cmb_cl_bb": cmb_lensed_bb_ell,
    }
    
    if gamma_dust_ell is not None:
        out_dict['gamma_dust_ell'] = gamma_dust_ell
    if gamma_sync_ell is not None:
        out_dict['gamma_sync_ell'] = gamma_sync_ell

    return out_dict

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

def _gen_data_per_freq_fg_template(fstr, cov_noise_ell, cmb_alm, nsplit, rngs_noise,
                                   ainfo, minfo, b_ell, fg_templates):
    '''
    Generate data for a given frequency, using foreground templates.

    Parameters
    ----------
    fstr : float
        Identifier of band, e.g. f090.
    cov_noise_ell : (npol, npol, nell) array
        Noise covariance matrix.
    cmb_alm : (2, nelem) complex array
        CMB E- and B-mode alms.
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
    fg_templates : dict
        Dictionary with fstr keys containing foreground B-mode alms.

    Returns
    -------
    out : (nsplit, 2, npix) array
        Stokes Q and U maps for each split.
    '''

    out = np.zeros((nsplit, 2, minfo.npix))

    signal_alm = cmb_alm.copy()
    signal_alm[1] += fg_templates[fstr]

    # Apply beam.
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

def estimate_spectra_cg(component_alms, ainfo, component_idx=0):
    """
    Compute cross-split EE and BB spectra for one reconstructed component.

    Parameters
    ----------
    component_alms : (nsplit, ncomp, 2, nalm) complex array
        Reconstructed component E/B alms for all noise splits.

    ainfo : pixell.curvedsky.alm_info
        Harmonic coefficient layout.

    component_idx : int, optional
        Index of the reconstructed component whose spectra are estimated.
        The default is 0, normally corresponding to CMB.

    Returns
    -------
    spectra : (npair, (E,B) , lmax + 1) array
        Cross-split spectra for each unique split pair.
    """
    component_alms = np.asarray(component_alms)

    if component_alms.ndim != 4 or component_alms.shape[2] != 2:
        raise ValueError(
            "component_alms must have shape "
            f"(nsplit, ncomp, 2, nalm), got {component_alms.shape}"
        )

    nsplit = component_alms.shape[0]

    if nsplit < 2:
        raise ValueError(
            "At least two independent noise splits are required "
            "for cross-split spectra."
        )

    alms = component_alms[:, component_idx]
    # Shape: (nsplit, 2, nalm)

    split_pairs = [
        (s1, s2)
        for s1 in range(nsplit)
        for s2 in range(s1 + 1, nsplit)
    ]

    spectra = np.zeros(
        (len(split_pairs), 2, ainfo.lmax + 1),
        dtype=float,
    )

    for idx, (s1, s2) in enumerate(split_pairs):
        cls = ainfo.alm2cl(
            alms[s1, :, None, :],
            alm2=alms[s2, None, :, :],
        )

        spectra[idx, 0] = cls[0, 0]  # EE
        spectra[idx, 1] = cls[1, 1]  # BB

    return spectra

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

    # List of length len(unique_combs) where each element is another list of indices.
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

def get_final_data_vector(spec, bins):
    '''
    Create data vector by binning and flattening spectra.

    Parameters
    ----------
    spec : (len(sels_to_coadd), 1, ellmax + 1)
        Input spectra.
    bins : (nbin + 1) array
        Output bins. Specify left edges and the rightmost edge.

    Returns
    -------
    out : (prod(...) * nbin) array
        Flattened and binned output array.
    '''

    preshape = spec.shape[:-1]
    ells = np.arange(spec.shape[-1])
    out = np.zeros(preshape + (bins.size - 1,))

    for idxs in np.ndindex(preshape):

        out[idxs] = spectra_utils.bin_spectrum(spec[idxs], ells, bins)

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
