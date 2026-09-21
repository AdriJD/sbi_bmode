TRANSFER FUNCTION

1. transfer_function_90.npy
    config_compute_transfer_1.yaml (90 GHz, highpass filter)
    d = OY(Bs + n)
2. transfer_function_90_no_highpass.npy 
    config_compute_transfer_2.yaml (90 GHz, no highpass filter)
    d = OY(Bs + n)
3. transfer_function_90_no_highpass_map_noise.npy
    config_compute_transfer_3.yaml (90 GHz, no highpass filter)
    d = OYBs + n
4. transfer_function_90_no_highpass_map_noise_1.npy
    config_compute_transfer_3.yaml (90 GHz, no highpass filter)
    d = OYBs + n
    use cross-spectra between 2 splits instead of auto-spectra on 1 split
5. transfer_function_90_main.py [main result]
    config_compute_transfer_4.yaml (90 GHz, no highpass filter)
    d = OYBs + n
    use cross-spectra between 2 splits instead of auto-spectra on 1 split  
    save apodization mask for later usage  
    Convert the NaN to 0 before saving

Note:
    - n is sampled from noise_cov_ell from N_ell of wolz et al. 2024
    - Transfer function = the linear interpolation from monte carlo average over 100 simulations.

MASK
1. full_mask_nside128.fits (from SO)
20% galatic mask
point source mask
SAT observation mask

2. full_mask_nside128_apod.fits
apodized C2, 2 deg

3. obsmat_planck70_apod_C2_3deg.fits [main result]
70% galatic mask from Planck
SAT observation mask
no point source mask