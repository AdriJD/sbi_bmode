import os

import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch

from pixell import enmap, enplot, reproject
from optweight import alm_utils, sht, alm_c_utils
from sbi_bmode import sim_utils, script_utils, spectra_utils, nilc_utils

opj = os.path.join
basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
specdir = '/u/adriaand/local/cca_project/data'
pyilcdir = '/u/adriaand/local/pyilc'

odir = opj(basedir, 'check_sync_fiducial')
imgdir = opj(odir, 'img')
os.makedirs(odir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)

def draw_from_prior(prior_list, nsamp):
    '''
    Extract prior limits.

    Parameters
    ----------
    prior_list : list of torch.distribition instances
        Priors.
    nsamp : int
        Number of prior draws.

    Returns
    -------
    draw : (nsamp, n_prior) array
        Prior draws.
    '''

    out = np.zeros((nsamp, len(prior_list)))

    for pidx, prior in enumerate(prior_list):
        out[:,pidx] = np.asarray(prior.sample((nsamp,)))[:,0]

    return out

def gen_data(A_d_BB, alpha_d_BB, beta_dust, freq_pivot_dust, temp_dust,
             r_tensor, A_lens, freqs, seed, nsplit, cov_noise_ell,
             cov_scalar_ell, cov_tensor_ell, b_ells, minfo, ainfo,
             amp_beta_dust=None, gamma_beta_dust=None, A_s_BB=None,
             alpha_s_BB=None, beta_sync=None, freq_pivot_sync=None,
             amp_beta_sync=None, gamma_beta_sync=None, rho_ds=None,
             signal_filter=None, no_cmb_ee=False):
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
    no_cmb_ee : bool, optional
        If set, set EE cmb constribution to zero. Added for backwards compatibiliy.

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
        r_tensor, A_lens, cov_scalar_ell, cov_tensor_ell)
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

    cmb_alm = alm_utils.rand_alm(cov_ell, ainfo, rng_cmb, dtype=np.complex128)
    if no_cmb_ee:
        cmb_alm[0] = 0

    cmb_map = np.zeros((2, minfo.npix))
    sht.alm2map(cmb_alm, cmb_map, ainfo, minfo, 2)
        
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
        beta_dust, gamma_dust_ell = sim_utils.get_beta_map(
            minfo, ainfo, beta_dust, amp_beta_dust, gamma_beta_dust, rng_beta)

        if A_s_BB is not None:
            alm_tmp[1] = fg_alm[1]
            sync_map = np.zeros((2, minfo.npix))
            sht.alm2map(alm_tmp, sync_map, ainfo, minfo, 2)
            beta_sync, gamma_sync_ell = sim_utils.get_beta_map(
                minfo, ainfo, beta_sync, amp_beta_sync, gamma_beta_sync, rng_beta)
        else:
            sync_map, beta_sync = None, None

        gen_data_per_freq = lambda freq, cov_noise_ell, b_ell: sim_utils._gen_data_per_freq_gamma(
            freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
            cmb_alm, dust_map, nsplit, rngs_noise, ainfo, minfo, b_ell, sync_map=sync_map,
            beta_sync=beta_sync, freq_pivot_sync=freq_pivot_sync)

    else:
        gen_data_per_freq = lambda freq, cov_noise_ell, b_ell: sim_utils._gen_data_per_freq_simple(
            freq, cov_noise_ell, beta_dust, temp_dust, freq_pivot_dust,
            cmb_alm, fg_alm, nsplit, rngs_noise, ainfo, minfo, b_ell, beta_sync=beta_sync,
            freq_pivot_sync=freq_pivot_sync)

        gamma_dust_ell, gamma_sync_ell = None, None

    for fidx, freq in enumerate(freqs):

        b_ell = b_ells[fidx]
        if signal_filter is not None:
            b_ell = b_ell * signal_filter
        out[:,fidx,:,:] = gen_data_per_freq(freq, cov_noise_ell[fidx], b_ell)

    out_dict = {'data' : out}
    out_dict['cmb_map'] = cmb_map    
    out_dict['dust_map'] = dust_map
    out_dict['sync_map'] = sync_map

    out_dict['beta_dust'] = beta_dust
    out_dict['beta_sync'] = beta_sync   
    
    if gamma_dust_ell is not None:
        out_dict['gamma_dust_ell'] = gamma_dust_ell
    if gamma_sync_ell is not None:
        out_dict['gamma_sync_ell'] = gamma_sync_ell

    return out_dict

with open(opj(basedir, 'run86', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)

data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
prior, param_names = script_utils.get_prior(params_dict)
param_truths = script_utils.get_true_params(params_dict)

torch.manual_seed(0)
nsamp = 1
thetas = draw_from_prior(prior, nsamp)

wavelet_type = 'TopHatHarmonic'
fiducial_T_dust = 19.6
fiducial_beta = 1.6
fiducial_beta_sync = -3

lmax = 200
ells = np.arange(lmax + 1)
bins = bins = np.arange(30, 210, 10)
bin_centers = bins[:-1] + 5
oversample = 4
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                         2 * np.pi / (2 * oversample * lmax + 1)],
                                    variant='fejer1')

cmb_simulator = sim_utils.CMBSimulator(
    specdir, data_dict, fixed_params_dict, pyilcdir=pyilcdir, wavelet_type=wavelet_type,
    use_dust_map=True, use_dbeta_map=True, use_sync_map=True,
    use_dbeta_sync_map=True, deproj_dust=True, deproj_dbeta=True,
    deproj_sync=True, deproj_dbeta_sync=True, fiducial_beta=fiducial_beta,
    fiducial_T_dust=fiducial_T_dust, fiducial_beta_sync=fiducial_beta_sync, odir=odir,
    coadd_equiv_crosses=True)

highpass_filter = np.ones((2, lmax+1)) * cmb_simulator.highpass_filter

print(f'{cmb_simulator.freq_pivot_sync=}')

# NOTE
#cmb_simulator.freq_pivot_sync = 30e9

for idx, theta in enumerate(thetas):
    print(f'{idx=}, {theta=}')

    # NOTE NOTE
    theta[7] = 4.
    #theta[7] = 40.
    #theta[2] = 0.001
    #theta[1] = 0.001
    #theta[0] = 0.001    

    # NOTE
    #cmb_simulator.noise_cov_ell *= 0.01
    
    theta_dict = dict(zip(param_names, theta))

    rng = np.random.default_rng(idx)

    # Create maps, save components
    odict = gen_data(
        theta_dict['A_d_BB'], theta_dict['alpha_d_BB'], theta_dict['beta_dust'],
        cmb_simulator.freq_pivot_dust, cmb_simulator.temp_dust,
        theta_dict['r_tensor'], theta_dict['A_lens'], cmb_simulator.freqs, rng,
        cmb_simulator.nsplit, cmb_simulator.noise_cov_ell,
        cmb_simulator.cov_scalar_ell, cmb_simulator.cov_tensor_ell,
        cmb_simulator.b_ells, cmb_simulator.minfo, cmb_simulator.ainfo,
        amp_beta_dust=theta_dict['amp_beta_dust'], gamma_beta_dust=theta_dict['gamma_beta_dust'],
        A_s_BB=theta_dict['A_s_BB'], alpha_s_BB=theta_dict['alpha_s_BB'], beta_sync=theta_dict['beta_sync'],
        freq_pivot_sync=cmb_simulator.freq_pivot_sync, amp_beta_sync=theta_dict['amp_beta_sync'],
        gamma_beta_sync=theta_dict['gamma_beta_sync'], rho_ds=theta_dict['rho_ds'],
        signal_filter=cmb_simulator.highpass_filter, no_cmb_ee=(cmb_simulator.mask is not None))

    #####
    #  data : (nsplit, nfreq, npol, npix)  
    omap = odict['data']
    dust_map = odict['dust_map']
    sync_map = odict['sync_map']
    cmb_map = odict['cmb_map']

    print(f'{omap.shape=}')
    print(f'{dust_map.shape=}')
    print(f'{sync_map.shape=}')

    tmp_alm = np.zeros((2, cmb_simulator.ainfo.nelem), dtype=np.complex128) # E, B.
    
    dust_b_map = np.zeros(cmb_simulator.minfo.npix)
    dust_alm = np.zeros_like(tmp_alm)

    sht.map2alm(dust_map, dust_alm, cmb_simulator.minfo, cmb_simulator.ainfo, 2)
    dust_alm = alm_c_utils.lmul(dust_alm, highpass_filter, cmb_simulator.ainfo)        
    sht.alm2map(dust_alm[1], dust_b_map, cmb_simulator.ainfo, cmb_simulator.minfo, 0)

    sync_b_map = np.zeros(cmb_simulator.minfo.npix)
    sync_alm = np.zeros_like(tmp_alm)
    
    sht.map2alm(sync_map, sync_alm, cmb_simulator.minfo, cmb_simulator.ainfo, 2)
    sync_alm = alm_c_utils.lmul(sync_alm, highpass_filter, cmb_simulator.ainfo)        
    sht.alm2map(sync_alm[1], sync_b_map, cmb_simulator.ainfo, cmb_simulator.minfo, 0)

    cmb_b_map = np.zeros(cmb_simulator.minfo.npix)
    cmb_alm = np.zeros_like(tmp_alm)
    
    sht.map2alm(cmb_map, cmb_alm, cmb_simulator.minfo, cmb_simulator.ainfo, 2)
    cmb_alm = alm_c_utils.lmul(cmb_alm, highpass_filter, cmb_simulator.ainfo)        
    sht.alm2map(cmb_alm[1], cmb_b_map, cmb_simulator.ainfo, cmb_simulator.minfo, 0)

    
    dust_enmap = reproject.healpix2map(dust_b_map, shape=shape, wcs=wcs, method='spline', order=0)
    sync_enmap = reproject.healpix2map(sync_b_map, shape=shape, wcs=wcs, method='spline', order=0)
    cmb_enmap = reproject.healpix2map(cmb_b_map, shape=shape, wcs=wcs, method='spline', order=0)            

    plot = enplot.plot(dust_enmap, colorbar=True, grid=30)
    enplot.write(opj(imgdir, f'dust_input'), plot)

    plot = enplot.plot(sync_enmap, colorbar=True, grid=30)
    enplot.write(opj(imgdir, f'sync_input'), plot)

    plot = enplot.plot(cmb_enmap, colorbar=True, grid=30)
    enplot.write(opj(imgdir, f'cmb_input'), plot)
    
    # Call NILC with tmp directories
    B_maps = np.zeros((cmb_simulator.nsplit, cmb_simulator.nfreq, cmb_simulator.minfo.npix))
    for split in range(cmb_simulator.nsplit):
        for f, freq_str in enumerate(cmb_simulator.freq_strings):
            sht.map2alm(omap[split,f], tmp_alm, cmb_simulator.minfo, cmb_simulator.ainfo, 2)
            sht.alm2map(tmp_alm[1], B_maps[split,f], cmb_simulator.ainfo, cmb_simulator.minfo, 0)

    B_maps *= 1e-6 # Convert to K because pyilc assumes that input is in K.

    cls_dict = {}

    betas = [-3, -4, -5, -6, -7]
    #betas = [-3, -4]
    #for bidx, fiducial_beta_sync in enumerate([-3, -4, -5, -6, -7]):
    for bidx, fiducial_beta_sync in enumerate(betas):

        cls_dict[fiducial_beta_sync] = {}

        
        map_tmpdir = nilc_utils.write_maps(B_maps, output_dir=cmb_simulator.odir)
        # nilc_maps = nilc_utils.get_nilc_maps(
        #     cmb_simulator.pyilcdir, map_tmpdir, cmb_simulator.nsplit, cmb_simulator.nside, cmb_simulator.fiducial_beta,
        #     cmb_simulator.fiducial_T_dust, cmb_simulator.freq_pivot_dust, cmb_simulator.freqs,
        #     cmb_simulator.beam_fwhms, wavelet_type=cmb_simulator.wavelet_type, use_dust_map=cmb_simulator.use_dust_map,
        #     use_dbeta_map=cmb_simulator.use_dbeta_map, use_sync_map=cmb_simulator.use_sync_map,
        #     use_dbeta_sync_map=cmb_simulator.use_dbeta_sync_map, deproj_dust=cmb_simulator.deproj_dust,
        #     deproj_dbeta=cmb_simulator.deproj_dbeta, deproj_sync=cmb_simulator.deproj_sync,
        #     deproj_dbeta_sync=cmb_simulator.deproj_dbeta_sync, fiducial_beta_sync=fiducial_beta_sync,
        #     freq_pivot_sync=cmb_simulator.freq_pivot_sync, output_dir=cmb_simulator.odir, remove_files=False,
        #     debug=True)

        nilc_maps = nilc_utils.get_nilc_maps(
            cmb_simulator.pyilcdir, map_tmpdir, cmb_simulator.nsplit, cmb_simulator.nside, cmb_simulator.fiducial_beta,
            cmb_simulator.fiducial_T_dust, cmb_simulator.freq_pivot_dust, cmb_simulator.freqs,
            cmb_simulator.beam_fwhms, wavelet_type=cmb_simulator.wavelet_type, use_dust_map=cmb_simulator.use_dust_map,
            use_dbeta_map=False, use_sync_map=cmb_simulator.use_sync_map,
            use_dbeta_sync_map=True, deproj_dust=cmb_simulator.deproj_dust,
            deproj_dbeta=False, deproj_sync=cmb_simulator.deproj_sync,
            deproj_dbeta_sync=True, fiducial_beta_sync=fiducial_beta_sync,
            freq_pivot_sync=cmb_simulator.freq_pivot_sync, output_dir=cmb_simulator.odir, remove_files=False,
            debug=True)


        print(f'{nilc_maps.shape=}')

        # Plot output
        #for cidx, comp in enumerate(['cmb', 'dust', 'sync', 'dbd', 'dbs']):
        for cidx, comp in enumerate(['cmb', 'dust', 'sync']):            

            for sidx in range(2):

                comp_enmap = reproject.healpix2map(
                    nilc_maps[sidx,cidx], shape=shape, wcs=wcs, method='spline', order=0)

                plot = enplot.plot(comp_enmap, colorbar=True, grid=30)
                enplot.write(opj(imgdir, f'out_mbfs{abs(fiducial_beta_sync)}_s{sidx}_{comp}'), plot)

        #for cidx, comp in enumerate(['cmb', 'dust', 'sync', 'dbd', 'dbs']):
        for cidx, comp in enumerate(['cmb', 'dust', 'sync']):            
            if not (comp in ('dust', 'sync')):
                continue
            
            cross_cls = []
            cross_cmb_cls = []
            cross_dbd_cls = []            
            in_cls = []
            bias_cls = []
            rho_cls = []
                        
            for sidx in range(2):
            
                alm_in = sync_alm if comp == 'sync' else dust_alm
                    
                sht.map2alm(nilc_maps[sidx,cidx].astype(np.float64), tmp_alm[1],
                            cmb_simulator.minfo, cmb_simulator.ainfo, 0)

                # Cross-spectrum output input
                cl_cross = cmb_simulator.ainfo.alm2cl(tmp_alm[1], alm_in[1])
                cl_cross_cmb = cmb_simulator.ainfo.alm2cl(tmp_alm[1], cmb_alm[1])
                cl_in = cmb_simulator.ainfo.alm2cl(alm_in[1], alm_in[1])
                cl_out = cmb_simulator.ainfo.alm2cl(tmp_alm[1], tmp_alm[1])                    

                bias = np.zeros(cl_in.shape)
                bias = np.divide(cl_cross, cl_in, where=ells > 30, out=bias)

                rho = np.zeros(cl_in.shape)
                rho = np.divide(cl_cross, np.sqrt(cl_in * cl_out), where=ells > 30, out=rho)

                alm_dbd_out = tmp_alm.copy()                

                sht.map2alm(nilc_maps[sidx,3].astype(np.float64), alm_dbd_out[1],
                            cmb_simulator.minfo, cmb_simulator.ainfo, 0)

                cl_cross_dbd = cmb_simulator.ainfo.alm2cl(tmp_alm[1], alm_dbd_out[1])

                cross_cls.append(cl_cross)
                cross_cmb_cls.append(cl_cross_cmb)
                cross_dbd_cls.append(cl_cross_dbd) 
                in_cls.append(cl_in)
                bias_cls.append(bias)
                rho_cls.append(rho)            
                
            print(f'{np.asarray(cross_cls).shape=}')
            cross_cls = np.mean(np.asarray(cross_cls), axis=0)
            cross_cmb_cls = np.mean(np.asarray(cross_cmb_cls), axis=0)
            cross_dbd_cls = np.mean(np.asarray(cross_dbd_cls), axis=0)            
            in_cls = np.mean(np.asarray(in_cls), axis=0)
            bias_cls = np.mean(np.asarray(bias_cls), axis=0)
            rho_cls = np.mean(np.asarray(rho_cls), axis=0)            
            
            cls_dict[fiducial_beta_sync][comp] = {}
            cls_dict[fiducial_beta_sync][comp]['cross_cls'] = cross_cls
            cls_dict[fiducial_beta_sync][comp]['cross_cmb_cls'] = cross_cmb_cls
            cls_dict[fiducial_beta_sync][comp]['cross_dbd_cls'] = cross_dbd_cls            
            cls_dict[fiducial_beta_sync][comp]['in_cls'] = in_cls
            cls_dict[fiducial_beta_sync][comp]['bias_cls'] = bias_cls
            cls_dict[fiducial_beta_sync][comp]['rho_cls'] = rho_cls            
            
    for cidx, comp in enumerate(['dust', 'sync']):
        
        fig, ax = plt.subplots(dpi=300)
        for bidx, fiducial_beta_sync in enumerate(betas):
            cross_cls = cls_dict[fiducial_beta_sync][comp]['cross_cls']
            ax.plot(bin_centers, spectra_utils.bin_spectrum(cross_cls, ells, bins),
                    label=f'beta_pyilc={fiducial_beta_sync+2}')
        ax.legend(frameon=False)
        ax.set_xlabel('Bin center')
        ax.set_ylabel(r'$C_{\ell}^{\mathrm{in, out}}$')
        fig.savefig(opj(imgdir, f'{comp}_cross'))
        plt.close(fig)
        
        fig, ax = plt.subplots(dpi=300)
        for bidx, fiducial_beta_sync in enumerate(betas):
            cross_cmb_cls = cls_dict[fiducial_beta_sync][comp]['cross_cmb_cls']
            ax.plot(bin_centers, spectra_utils.bin_spectrum(cross_cmb_cls, ells, bins),
                    label=f'beta_pyilc={fiducial_beta_sync+2}')
        ax.legend(frameon=False)
        ax.set_xlabel('Bin center')
        ax.set_ylabel(r'$C_{\ell}^{\mathrm{cmb, out}}$')        
        fig.savefig(opj(imgdir, f'{comp}_cross_cmb'))
        plt.close(fig)


        fig, ax = plt.subplots(dpi=300)
        for bidx, fiducial_beta_sync in enumerate(betas):
            cross_dbd_cls = cls_dict[fiducial_beta_sync][comp]['cross_dbd_cls']
            ax.plot(bin_centers, spectra_utils.bin_spectrum(cross_dbd_cls, ells, bins),
                    label=f'beta_pyilc={fiducial_beta_sync+2}')
        ax.legend(frameon=False)
        ax.set_xlabel('Bin center')
        ax.set_ylabel(r'$C_{\ell}^{\mathrm{dbd, out}}$')        
        fig.savefig(opj(imgdir, f'{comp}_cross_dbd'))
        plt.close(fig)

        
        fig, ax = plt.subplots(dpi=300)
        for bidx, fiducial_beta_sync in enumerate(betas):
            in_cls = cls_dict[fiducial_beta_sync][comp]['in_cls']
            ax.plot(bin_centers, spectra_utils.bin_spectrum(in_cls, ells, bins),
                    label=f'beta_pyilc={fiducial_beta_sync+2}')
        ax.legend(frameon=False)
        ax.set_xlabel('Bin center')
        ax.set_ylabel(r'$C_{\ell}^{\mathrm{in, in}}$')                
        fig.savefig(opj(imgdir, f'{comp}_in'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300)
        for bidx, fiducial_beta_sync in enumerate(betas):
            bias_cls = cls_dict[fiducial_beta_sync][comp]['bias_cls']
            ax.plot(bin_centers, spectra_utils.bin_spectrum(bias_cls, ells, bins),
                    label=f'beta_pyilc={fiducial_beta_sync+2}')
        ax.legend(frameon=False)
        ax.set_xlabel('Bin center')
        ax.set_ylabel(r'$C_{\ell}^{\mathrm{in, out}} / C_{\ell}^{\mathrm{in, in}}$')        
        fig.savefig(opj(imgdir, f'{comp}_bias'))
        plt.close(fig)

        fig, ax = plt.subplots(dpi=300)
        for bidx, fiducial_beta_sync in enumerate(betas):
            rho_cls = cls_dict[fiducial_beta_sync][comp]['rho_cls']
            ax.plot(bin_centers, spectra_utils.bin_spectrum(rho_cls, ells, bins),
                    label=f'beta_pyilc={fiducial_beta_sync+2}')
        ax.legend(frameon=False)
        ax.set_xlabel('Bin center')
        ax.set_ylabel(r'$C_{\ell}^{\mathrm{in, out}} / \sqrt{C_{\ell}^{\mathrm{in, in}} C_{\ell}^{\mathrm{out, out}}}$')        
        fig.savefig(opj(imgdir, f'{comp}_rho'))
        plt.close(fig)
        
                    
# Do this for different values of fiducial_beta_sync
# Perhaps also for different prior draws...
#
