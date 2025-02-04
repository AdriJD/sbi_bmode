import shutil
import tempfile
import healpy as hp
import numpy as np
import subprocess
import yaml
import os

def write_maps(B_maps, output_dir=None):
    '''
    Parameters
    ----------
    B_maps: (nsplit, nfreq, npix) ndarray containing B-mode maps (as scalar fields)
    output_dir: str, directory in which to make temporary directory
                (if set to None, the default $TMPDIR will be used)

    Returns
    -------
    map_tmpdir: str, temporary directory to which maps are written
    '''
    map_tmpdir = tempfile.mkdtemp(dir=output_dir)
    nsplit, nfreq, npix = B_maps.shape
    for split in range(nsplit):
        for f in range(nfreq):
            map_ = B_maps[split, f]
            hp.write_map(f'{map_tmpdir}/map_split{split}_freq{f}.fits', map_, dtype=np.float32)
    return map_tmpdir


def get_nilc_maps(pyilc_path, map_tmpdir, nsplit, nside, beta_dust, temp_dust, freq_pivot_dust, 
                  sat_central_freqs, sat_beam_fwhms, use_dbeta_map=False, 
                  deproj_dust=False, deproj_dbeta=False, fiducial_beta=None, fiducial_T_dust=None,
                  output_dir=None, remove_files=True, debug=False):
    '''
    Parameters
    ----------
    pyilc_path: str, path to pyilc repository
    map_tmpdir: str, path where frequency maps have been written
    nsplit: int, number of splits
    nside: int, resolution parameter for maps
    beta_dust: float, beta dust parameter
    temp_dust: float, temperature of dust
    freq_pivot_dust: float, pivot frequency of dust
    sat_central_freqs: dict, maps frequency strings to floats representing frequencies
    sat_beam_fwhms: dict, maps frequency strings to floats representing beam FWHM
    use_dbeta_map: Bool, whethert o build map of first moment w.r.t. beta
    deproj_dust: Bool, Whether to deproject dust in CMB NILC map.
    deproj_dbeta: Bool, Whether to deproject first moment of dust w.r.t. beta in CMB NILC map.
    fiducial_beta: float, optional
        If not None, use this value for beta when building nilc maps. Otherwise, use a 
        separate value for each simulation.
    fiducial_T_dust: float, optional
        If not None, use this value for T_dust when building nilc maps. Otherwise, use a 
        separate value for each simulation.
    output_dir: str, directory in which to make temporary directory for NILC maps
                (if set to None, the default $TMPDIR will be used)
    remove_files: Bool, whether to remove files when they're no longer needed
    debug: Bool, set to True to print intermediate outputs from pyilc, False to suppress

    Returns
    -------
    nilc_maps: (nsplit, ncomp=2 or 3, npix) ndarray containing NILC maps
                For ncomp, the first index is for CMB NILC maps and 
                the second index is for dust NILC maps

    '''
    Ncomp = 2 if not use_dbeta_map else 3
    nilc_maps = np.zeros((nsplit, Ncomp, 12*nside**2), dtype=np.float32)
    
    # current environment, also environment in which to run subprocesses
    env = os.environ.copy()

    for split in range(nsplit):

        nilc_tmpdir = tempfile.mkdtemp(dir=output_dir)

        #Basic NILC parameters
        pyilc_input_params = {}
        pyilc_input_params['output_dir'] = nilc_tmpdir + '/'
        pyilc_input_params['output_prefix'] = f""
        pyilc_input_params['save_weights'] = "no"
        pyilc_input_params['ELLMAX'] = 3*nside-2
        pyilc_input_params['N_scales'] = 4
        pyilc_input_params['GN_FWHM_arcmin'] = [300., 120., 60.] 
        pyilc_input_params['taper_width'] = 0
        pyilc_input_params['N_freqs'] = len(sat_central_freqs)
        pyilc_input_params['freqs_delta_ghz'] = list(sat_central_freqs.values())
        pyilc_input_params['N_side'] = nside
        pyilc_input_params['wavelet_type'] = "GaussianNeedlets"
        pyilc_input_params['bandpass_type'] = "DeltaBandpasses"
        pyilc_input_params['beam_type'] = "Gaussians"
        pyilc_input_params['beam_FWHM_arcmin'] = list(sat_beam_fwhms.values())
        pyilc_input_params['ILC_bias_tol'] = 0.01
        pyilc_input_params['N_deproj'] = 0
        pyilc_input_params['N_SED_params'] = 0
        pyilc_input_params['N_maps_xcorr'] = 0
        pyilc_input_params['freq_map_files'] = [f'{map_tmpdir}/map_split{split}_freq{f}.fits' for f in range(pyilc_input_params['N_freqs'])] 
        pyilc_input_params['save_as'] = 'fits'

        #Dust parameters
        pars = {'beta_CIB': float(beta_dust), 'Tdust_CIB': float(temp_dust), 'nu0_CIB_ghz': float(freq_pivot_dust),
                'kT_e_keV':5.0, 'nu0_radio_ghz':150.0, 'beta_radio': -0.5}
        if fiducial_beta is not None:
            pars['beta_CIB'] = float(fiducial_beta)
        if fiducial_T_dust is not None:
            pars['Tdust_CIB'] = float(fiducial_T_dust)
        dust_pars_yaml = f'{nilc_tmpdir}/dust_pars.yaml'
        with open(dust_pars_yaml, 'w') as outfile:
            yaml.dump(pars, outfile, default_flow_style=None)
        pyilc_input_params['param_dict_file'] = f'{nilc_tmpdir}/dust_pars.yaml'

        #CMB-specific and dust-specific dictionaries
        cmb_param_dict = {'ILC_preserved_comp': 'CMB'}
        cmb_param_dict.update(pyilc_input_params)
        if deproj_dust and deproj_dbeta:
            cmb_param_dict['ILC_deproj_comps'] = ['CIB','CIB_dbeta']
            cmb_param_dict['N_deproj'] = 2
        elif deproj_dust:
            cmb_param_dict['ILC_deproj_comps'] = ['CIB']
            cmb_param_dict['N_deproj'] = 1
        dust_param_dict = {'ILC_preserved_comp': 'CIB'} 
        dust_param_dict.update(pyilc_input_params)
        all_param_dicts = [cmb_param_dict, dust_param_dict]
        if use_dbeta_map:
            dbeta_param_dict = {'ILC_preserved_comp': 'CIB_dbeta'}
            dbeta_param_dict.update(pyilc_input_params)
            all_param_dicts.append(dbeta_param_dict)

        #Dump CMB and dust yaml files
        comps = ['CMB', 'dust']
        if use_dbeta_map:
            comps.append('dbeta')
        all_yaml_files = [f'{nilc_tmpdir}/split{split}_{comp}_preserved.yml' for comp in comps]
        for c, comp in enumerate(comps):
            with open(all_yaml_files[c], 'w') as outfile:
                yaml.dump(all_param_dicts[c], outfile, default_flow_style=None)

        #run pyilc for each preserved component
        stdout = subprocess.DEVNULL if not debug else None
        for c, comp in enumerate(comps):
            subprocess.run([f"python {pyilc_path}/pyilc/main.py {all_yaml_files[c]}"], shell=True, env=env, stdout=stdout, stderr=stdout)
        
        #load NILC maps, then remove nilc tmpdir
        cmb_nilc = hp.read_map(f'{nilc_tmpdir}/needletILCmap_component_CMB.fits')
        #dust_nilc = 10**6*hp.read_map(f'{nilc_tmpdir}/needletILCmap_component_CIB.fits')
        dust_nilc = hp.read_map(f'{nilc_tmpdir}/needletILCmap_component_CIB.fits')        
        if use_dbeta_map:
            #dbeta_nilc = 10**6*hp.read_map(f'{nilc_tmpdir}/needletILCmap_component_CIB_dbeta.fits')
            dbeta_nilc = hp.read_map(f'{nilc_tmpdir}/needletILCmap_component_CIB_dbeta.fits')            
            nilc_maps[split] = np.array([cmb_nilc, dust_nilc, dbeta_nilc])
        else:
            nilc_maps[split] = np.array([cmb_nilc, dust_nilc])
        if remove_files:
            shutil.rmtree(nilc_tmpdir)
    
    # remove frequency map tmpdir
    if remove_files:
        shutil.rmtree(map_tmpdir)
        
    return nilc_maps


    
