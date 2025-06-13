import os
import shutil
import tempfile
import subprocess
import yaml

import numpy as np
import healpy as hp

def write_maps(B_maps, output_dir=None):
    '''
    Write maps to disk.

    Parameters
    ----------
    B_maps : (nsplit, nfreq, npix) ndarray
        B-mode maps (as scalar fields).
    output_dir : str
        Directory in which to make temporary directory (if set to None, the default $TMPDIR
        will be used).

    Returns
    -------
    map_tmpdir : str
        Temporary directory to which maps are written.
    '''
    
    map_tmpdir = tempfile.mkdtemp(dir=output_dir)
    nsplit, nfreq, npix = B_maps.shape
    
    for split in range(nsplit):
        for f in range(nfreq):
            map_ = B_maps[split, f]
            hp.write_map(f'{map_tmpdir}/map_split{split}_freq{f}.fits', map_, dtype=np.float32)
            
    return map_tmpdir

def get_nilc_maps(pyilc_path, map_tmpdir, nsplit, nside, fiducial_beta, fiducial_T_dust, freq_pivot_dust, 
                  central_freqs, beam_fwhms, use_dust_map=True, use_dbeta_map=False, use_sync_map=False,
                  use_dbeta_sync_map=False, deproj_dust=False, deproj_dbeta=False, deproj_sync=False,
                  deproj_dbeta_sync=False, fiducial_beta_sync=None, freq_pivot_sync=None, output_dir=None,
                  remove_files=True, debug=False):
    '''
    Run pyilc and return NILC map(s).

    Parameters
    ----------
    pyilc_path : str
        Path to pyilc repository.
    map_tmpdir : str
        Path where frequency maps have been written.
    nsplit : int
        Number of splits.
    nside : int
        Resolution parameter for maps.
    fiducial_beta : float
        Beta dust parameter.
    fiducial_T_dust : float
        Temperature of dust in K.
    freq_pivot_dust : float
        Pivot frequency of dust in Hz.
    central_freqs : array-like
        List of floats representing frequencies in Hz.
    beam_fwhms : array-like
        List of floats representing beam FWHM.
    use_dust_map : Bool, optional
        Whether to produce dust map.
    use_dbeta_map : Bool, optional
        Whether to build map of first moment w.r.t. beta.
    use_sync_map : Bool, optional
        Whether to produce sync map.
    use_dbeta_sync_map : Bool, optional
        Whether to build map of first moment w.r.t. beta synchrotron.
    deproj_dust : Bool, optional
        Whether to deproject dust in CMB NILC map.
    deproj_dbeta : Bool, optional
        Whether to deproject first moment of dust w.r.t. beta in CMB NILC map.
    deproj_sync : Bool, optional
        Whether to deproject synchrotron in CMB NILC map.
    deproj_dbeta_sync : Bool, optional
        Whether to deproject first moment of dust w.r.t. beta synchrotron in CMB NILC map.
    fiducial_beta_sync : float. optional
        Beta synchrotron parameter.
    freq_pivot_sync : float, None
        Pivot frequency of synchrotron SED in Hz.
    output_dir : str, optional
        Directory in which to make temporary directory for NILC maps (if set to None,
        the default $TMPDIR will be used).
    remove_files : Bool, optional
        Whether to remove files when they're no longer needed.
    debug : Bool, optional
        Set to True to print intermediate outputs from pyilc, False to suppress.

    Returns
    -------
    nilc_maps: (nsplit, ncomp=1, 2, 3, 4 or 5, npix) ndarray
        NILC maps for ncomp, the first index is for CMB NILC maps and the second index
        is for dust, dbeta, sync, dbeta_sync NILC maps.
    '''

    Ncomp = 1
    if use_dust_map: Ncomp += 1
    if use_dbeta_map: Ncomp += 1
    if use_sync_map: Ncomp += 1
    if use_dbeta_sync_map: Ncomp += 1

    nilc_maps = np.zeros((nsplit, Ncomp, 12*nside**2), dtype=np.float32)
    
    # Current environment, also environment in which to run subprocesses.
    env = os.environ.copy()

    # Convert to Ghz.
    central_freqs = [nu * 1e-9 for nu in central_freqs]
    
    for split in range(nsplit):

        nilc_tmpdir = tempfile.mkdtemp(dir=output_dir)

        # Basic NILC parameters.
        pyilc_input_params = {}
        pyilc_input_params['output_dir'] = nilc_tmpdir + '/'
        pyilc_input_params['output_prefix'] = f""
        pyilc_input_params['save_weights'] = "no"
        pyilc_input_params['ELLMAX'] = 3*nside-2
        pyilc_input_params['N_scales'] = 4
        pyilc_input_params['GN_FWHM_arcmin'] = [300., 120., 60.] 
        pyilc_input_params['taper_width'] = 0
        pyilc_input_params['N_freqs'] = len(central_freqs)
        pyilc_input_params['freqs_delta_ghz'] = central_freqs
        pyilc_input_params['N_side'] = nside
        pyilc_input_params['wavelet_type'] = "GaussianNeedlets"
        pyilc_input_params['bandpass_type'] = "DeltaBandpasses"
        pyilc_input_params['beam_type'] = "Gaussians"
        pyilc_input_params['beam_FWHM_arcmin'] = beam_fwhms
        pyilc_input_params['ILC_bias_tol'] = 0.01
        pyilc_input_params['N_deproj'] = 0
        pyilc_input_params['N_SED_params'] = 0
        pyilc_input_params['N_maps_xcorr'] = 0
        pyilc_input_params['freq_map_files'] = \
            [f'{map_tmpdir}/map_split{split}_freq{f}.fits' for f in range(pyilc_input_params['N_freqs'])] 
        pyilc_input_params['save_as'] = 'fits'

        # NOTE, I should keep track of these failures. They should never reach 0.9, but values of
        # 1e-2 seem to be hard to avoid when deprojecting four sky components.
        pyilc_input_params['resp_tol'] = 10 # i.e. disable.
        #pyilc_input_params['resp_tol'] = 1e-2

        # Foreground parameters.
        nu0_radio_ghz = float(freq_pivot_sync) * 1e-9 if freq_pivot_sync is not None else 22.0
        beta_radio = float(fiducial_beta_sync) if fiducial_beta_sync is not None else -3.0
        pars = {'beta_CIB': float(fiducial_beta),
                'Tdust_CIB': float(fiducial_T_dust),
                'nu0_CIB_ghz': float(freq_pivot_dust) * 1e-9,
                'kT_e_keV': 5.0,
                'nu0_radio_ghz' : nu0_radio_ghz,
                'beta_radio': beta_radio}
        dust_pars_yaml = f'{nilc_tmpdir}/dust_pars.yaml'
        with open(dust_pars_yaml, 'w') as outfile:
            yaml.dump(pars, outfile, default_flow_style=None)
        pyilc_input_params['param_dict_file'] = f'{nilc_tmpdir}/dust_pars.yaml'

        # CMB-specific and dust-specific dictionaries.
        cmb_param_dict = {'ILC_preserved_comp': 'CMB'}
        cmb_param_dict.update(pyilc_input_params)
        cmb_oname = 'needletILCmap_component_CMB'
        ilc_deproj_comps = []
        if (deproj_dust or use_dust_map):
            ilc_deproj_comps.append('CIB')
        if deproj_dbeta or use_dbeta_map:
            ilc_deproj_comps.append('CIB_dbeta')
        if (deproj_sync or use_sync_map):
            ilc_deproj_comps.append('radio')
        if deproj_dbeta_sync or use_dbeta_sync_map:
            ilc_deproj_comps.append('radio_dbeta')

        cmb_param_dict['N_deproj'] = len(ilc_deproj_comps)
        cmb_param_dict['ILC_deproj_comps'] = ilc_deproj_comps
        if len(ilc_deproj_comps) > 0:
            cmb_oname += f'_deproject_{'_'.join(ilc_deproj_comps)}'
            
        comps = ['CMB']        
        all_param_dicts = [cmb_param_dict]
        
        if use_dust_map:
            comps.append('dust')
            dust_oname = 'needletILCmap_component_CIB'
            ilc_deproj_comps_dust = ['CMB']
            if (deproj_dbeta or use_dbeta_map):
                ilc_deproj_comps_dust.append('CIB_dbeta')
            if (deproj_sync or use_sync_map):
                ilc_deproj_comps_dust.append('radio')
            if (deproj_dbeta_sync or use_dbeta_sync_map):
                ilc_deproj_comps_dust.append('radio_dbeta')

            dust_param_dict = {'ILC_preserved_comp': 'CIB'}
            dust_param_dict.update(pyilc_input_params)
            dust_param_dict['ILC_deproj_comps'] = ilc_deproj_comps_dust
            dust_param_dict['N_deproj'] = len(ilc_deproj_comps_dust)
            all_param_dicts.append(dust_param_dict)
            dust_oname += f'_deproject_{'_'.join(ilc_deproj_comps_dust)}'
            
        if use_dbeta_map:

            comps.append('dbeta')
            dbeta_oname = 'needletILCmap_component_CIB_dbeta'
            ilc_deproj_comps_dbeta = ['CMB']
            if (deproj_dust or use_dust_map):
                ilc_deproj_comps_dbeta.append('CIB')
            if (deproj_sync or use_sync_map):
                ilc_deproj_comps_dbeta.append('radio')
            if (deproj_dbeta_sync or use_dbeta_sync_map):
                ilc_deproj_comps_dbeta.append('radio_dbeta')

            dbeta_param_dict = {'ILC_preserved_comp': 'CIB_dbeta'}
            dbeta_param_dict.update(pyilc_input_params)
            dbeta_param_dict['ILC_deproj_comps'] = ilc_deproj_comps_dbeta
            dbeta_param_dict['N_deproj'] = len(ilc_deproj_comps_dbeta)
            all_param_dicts.append(dbeta_param_dict)
            dbeta_oname += f'_deproject_{'_'.join(ilc_deproj_comps_dbeta)}'

        if use_sync_map:

            comps.append('sync')
            sync_oname = 'needletILCmap_component_radio'
            ilc_deproj_comps_sync = ['CMB']
            if (deproj_dust or use_dust_map):
                ilc_deproj_comps_sync.append('CIB')
            if (deproj_dbeta or use_dbeta_map):
                ilc_deproj_comps_sync.append('CIB_dbeta')
            if (deproj_dbeta_sync or use_dbeta_sync_map):
                ilc_deproj_comps_sync.append('radio_dbeta')

            sync_param_dict = {'ILC_preserved_comp': 'radio'}
            sync_param_dict.update(pyilc_input_params)
            sync_param_dict['ILC_deproj_comps'] = ilc_deproj_comps_sync
            sync_param_dict['N_deproj'] = len(ilc_deproj_comps_sync)
            all_param_dicts.append(sync_param_dict)
            sync_oname += f'_deproject_{'_'.join(ilc_deproj_comps_sync)}'
 
        if use_dbeta_sync_map:

            comps.append('dbeta_sync')
            dbeta_sync_oname = 'needletILCmap_component_radio_dbeta'            
            ilc_deproj_comps_dbeta_sync = ['CMB']
            if (deproj_dust or use_dust_map):
                ilc_deproj_comps_dbeta_sync.append('CIB')
            if (deproj_dbeta or use_dbeta_map):
                ilc_deproj_comps_dbeta_sync.append('CIB_dbeta')
            if (deproj_sync or use_sync_map):
                ilc_deproj_comps_dbeta_sync.append('radio')

            dbeta_sync_param_dict = {'ILC_preserved_comp': 'radio_dbeta'}
            dbeta_sync_param_dict.update(pyilc_input_params)
            dbeta_sync_param_dict['ILC_deproj_comps'] = ilc_deproj_comps_dbeta_sync
            dbeta_sync_param_dict['N_deproj'] = len(ilc_deproj_comps_dbeta_sync)
            all_param_dicts.append(dbeta_sync_param_dict)           
            dbeta_sync_oname += f'_deproject_{'_'.join(ilc_deproj_comps_dbeta_sync)}'
                                    
        # Dump CMB and dust yaml files.
        all_yaml_files = [f'{nilc_tmpdir}/split{split}_{comp}_preserved.yml' for comp in comps]
        for c, comp in enumerate(comps):
            with open(all_yaml_files[c], 'w') as outfile:
                yaml.dump(all_param_dicts[c], outfile, default_flow_style=None)

        # Run pyilc for each preserved component.
        #stdout = subprocess.DEVNULL if not debug else None
        stdout = open(os.path.join(nilc_tmpdir, 'stdout.txt'), "w") if not debug else None

        for c, comp in enumerate(comps):
            subprocess.run([f"python {pyilc_path}/pyilc/main.py {all_yaml_files[c]}"],
                           shell=True, env=env, stdout=stdout, stderr=subprocess.STDOUT)
        stdout.close()
            
        # Load NILC maps, then remove nilc tmpdir.
        cmb_nilc = hp.read_map(f'{nilc_tmpdir}/{cmb_oname}.fits')
        nilc_maps[split,0] = cmb_nilc
        idx = 1
        if use_dust_map:
            dust_nilc = hp.read_map(f'{nilc_tmpdir}/{dust_oname}.fits')
            nilc_maps[split,idx] = dust_nilc
            idx += 1
        if use_dbeta_map:
            dbeta_nilc = hp.read_map(f'{nilc_tmpdir}/{dbeta_oname}.fits')
            nilc_maps[split,idx] = dbeta_nilc
            idx += 1
        if use_sync_map:
            sync_nilc = hp.read_map(f'{nilc_tmpdir}/{sync_oname}.fits')
            nilc_maps[split,idx] = sync_nilc
            idx += 1
        if use_dbeta_sync_map:
            dbeta_sync_nilc = hp.read_map(f'{nilc_tmpdir}/{dbeta_sync_oname}.fits')
            nilc_maps[split,idx] = dbeta_sync_nilc
            idx += 1
            
        if remove_files:
            shutil.rmtree(nilc_tmpdir)
    
    # Remove frequency map tmpdir.
    if remove_files:
        shutil.rmtree(map_tmpdir)
        
    return nilc_maps


    
