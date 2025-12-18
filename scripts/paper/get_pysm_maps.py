import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pysm3
import pysm3.units as up
from pixell import enmap, reproject

opj = os.path.join

maskdir_planck = '/u/adriaand/project/planck/20250825_2015_gal_masks'
odir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(odir, 'masks')
imgdir = opj(odir, 'img')
os.makedirs(odir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)

freqs = ['wK', 'f030', 'f040', 'f090', 'f150', 'f230', 'f290', 'p353']

#dust_models = ['d1', 'd4', 'd10', 'd12']
#sync_models = ['s5', 's7', ]
dust_models = ['d12']
sync_models = []

# Load maps at native res
native_res_dict = {'d0' : 512, 'd1' : 512, 'd4' : 512, 'd10' : 2048, 'd12' : 2048,
                   's5' : 2048, 's7' : 2048}
scaling_dict = {'d0' : 1, 'd1' : 1, 'd4' : 1, 'd10' : 1, 'd12' : 1,
                's5' : 1, 's7' : 1, 'd10alt' : 1.6}
central_freq_dict = {'wK' : 25.e9, 'f030' : 27e9, 'f040' : 39e9, 'f090' : 93e9,
                     'f150' : 145e9, 'f230' : 225e9, 'f290' : 280e9, 'p353' : 340e9}

def apod_mask(mask_hp, width):
    '''
    Reproject to CAR, apodize and reproject back.
    '''
    
    small_pix = mask_hp < 0.5
    mask_hp[small_pix] = 0
    mask_hp[~small_pix] = 1

    nside = hp.npix2nside(mask_hp.shape[-1])
    lmax = 2 * nside
    oversample = 1
    shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                             2 * np.pi / (2 * oversample * lmax + 1)])
    
    # Project to pixell
    mask_enmap = reproject.healpix2map(mask_hp, shape=shape, wcs=wcs, method='spline', order=0)
    small_pix = mask_enmap < 1e-4
    mask_enmap[small_pix] = 0
    mask_enmap[~small_pix] = 1

    # Apodize
    mask_enmap = enmap.apod_mask(mask_enmap, width=width, edge=False)

    return reproject.map2healpix(mask_enmap, nside=nside, method='spline', order=3)

# Mask is needed to compute average Tdust in masked region for 1.6 scaled beta_d d10 version.
# See 2508.00073.
mask = hp.read_map(opj(maskdir, 'mask.fits'))

# Create apodized masks used to remove bright pixels. 80% mask.
galmask = hp.read_map(opj(maskdir_planck, 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'), field=4)
galmask = galmask.astype(np.float64)
galmask_512 = hp.ud_grade(galmask, 512)
galmask_2048 = hp.ud_grade(galmask, 2048)

galmask_512 = apod_mask(galmask_512, np.radians(2))
galmask_2048 = apod_mask(galmask_2048, np.radians(2))

hp.mollview(galmask_512)
fig = plt.gcf()
fig.savefig(opj(imgdir, "galmask_4_512"))
plt.close(fig)

hp.mollview(galmask_2048)
fig = plt.gcf()
fig.savefig(opj(imgdir, "galmask_4_2048"))
plt.close(fig)

galmask_dict = {512 : galmask_512, 2048 : galmask_2048}

for model in (dust_models + sync_models):

    pysm_model = model[:-3] if model[-3:] == 'alt' else model
    nside = native_res_dict[pysm_model]    
    sky = pysm3.Sky(nside=nside, preset_strings=[pysm_model])

    scaling = scaling_dict[model]
    if scaling != 1:
        mask_ug = hp.ud_grade(mask, nside)
        beta_dust = sky.components[0].mbb_index
        beta_dust_mean = np.sum(beta_dust * mask_ug) / np.sum(mask_ug)
        sky.components[0].mbb_index[:] = (beta_dust - beta_dust_mean) * scaling + beta_dust_mean
        t_dust = sky.components[0].mbb_temperature
        sky.components[0].mbb_temperature[:] = np.sum(t_dust * mask_ug) / np.sum(mask_ug)
        print(sky.components[0].mbb_temperature[:])
    
    for freq in freqs:
        
        central_freq = central_freq_dict[freq]

        imap = sky.get_emission(central_freq * up.Hz)

        # Convert to CMB units.
        imap = imap.to(up.uK_CMB, equivalencies=up.cmb_equivalencies(central_freq * up.Hz))
        
        # Downgrade
        lmax = 200
        # Convert to T, E, B.
        alm = hp.map2alm(imap, 2 * nside, use_pixel_weights=True)
        imap_b = hp.alm2map(alm[2], nside)
        alm_b = hp.map2alm(imap_b * galmask_dict[nside], lmax, use_pixel_weights=True)        
        
        hp.write_alm(opj(odir, f'pysm_{model}_{freq}.fits'), alm_b, overwrite=True)

        # Also save unmasked T at nside 128.
        alm = hp.map2alm(imap[0] * galmask_dict[nside], lmax, use_pixel_weights=True)        
        imap_t = hp.alm2map(alm, 128)
        
        hp.write_map(opj(odir, f'pysm_{model}_{freq}_I_map_unmasked.fits'), imap_t, overwrite=True)
