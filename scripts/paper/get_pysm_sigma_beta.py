import os

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import healpy as hp
import pysm3
import pysm3.units as up
from pixell import enmap, reproject, curvedsky, enplot

opj = os.path.join

maskdir_planck = '/u/adriaand/project/planck/20250825_2015_gal_masks'
odir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(odir, 'masks')
imgdir = opj(odir, 'img')
os.makedirs(odir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)

def get_noise_variance(cl_beta, lmin=2, lmax=2000):
    '''
    Convert amplitude and gamma parameters to noise variance.

    Parameters
    ----------
    cl_beta : (nell) array
        Beta power spectrum.
    lmin : int, optional
        Lower limit for variance computation.
    lmax : int, optional
        Upper limit for variance computation.

    Returns
    -------
    var : float
        Variance.    
    '''
            
    ell_pivot = 1
    ells = np.arange(lmin, lmax+1)
    c_ell = np.zeros(lmax+1)
    c_ell[:cl_beta.size] = cl_beta
    #c_ell[:,lmin:lmax+1] = amp * (ells / ell_pivot) ** gamma    
    out = np.sum((2 * ells + 1) / (4 * np.pi) * c_ell[lmin:lmax+1])

    return out

def fit_power_law(cl_beta, lmin=10, lmax=100):
    '''
    Fit B (ell / 1)^{gamma} to power spectrum.

    Parameters
    ----------
    cl_beta : (nell) array
        Beta power spectrum.
    lmin : int, optional
        Minimum multipole for fit
    lmax : int, optional
        Maximum multipole for fit.

    Returns
    -------
    b_amp : float
        Amplitude of power spectrum.
    gamma : float
        Power law index.
    '''

    ells = np.arange(cl_beta.size)
    power_law = lambda ells, b, g: b * ells ** g

    popt, pcov = curve_fit(power_law, ells[lmin:lmax+1], cl_beta[lmin:lmax+1])
    
    return popt
                     
#freqs = ['wK', 'f030', 'f040', 'f090', 'f150', 'f230', 'f290', 'p353']

dust_models = ['d1', 'd10', 'd12']
sync_models = ['s5']

# Load maps at native res
native_res_dict = {'d0' : 512, 'd1' : 512, 'd4' : 512, 'd10' : 2048, 'd12' : 2048,
                   's5' : 2048, 's7' : 2048}
scaling_dict = {'d0' : 1, 'd1' : 1, 'd4' : 1, 'd10' : 1, 'd12' : 1,
                's5' : 1, 's7' : 1, 'd10alt' : 1.6}
central_freq_dict = {'wK' : 25.e9, 'f030' : 27e9, 'f040' : 39e9, 'f090' : 93e9,
                     'f150' : 145e9, 'f230' : 225e9, 'f290' : 280e9, 'p353' : 340e9}

mask = hp.read_map(opj(maskdir, 'mask.fits'))

# NOTE
#mask[:] = 1
lmax = 200
ainfo = curvedsky.alm_info(lmax)
oversample = 2
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                        2 * np.pi / (2 * oversample * lmax + 1)])
omap = enmap.zeros(shape, wcs)

fig, ax = plt.subplots(dpi=300)
fig2, ax2 = plt.subplots(dpi=300)

for midx, model in enumerate((dust_models + sync_models)):

    pysm_model = model[:-3] if model[-3:] == 'alt' else model
    nside = native_res_dict[pysm_model]    
    sky = pysm3.Sky(nside=nside, preset_strings=[pysm_model])

    # Set reference freqs
    
    mask_ug = hp.ud_grade(mask, nside)
    if model.startswith('d'):
        if model == 'd12':
            beta = sky.components[0].mbb_index[0].value # First layer.
        else:
            beta = sky.components[0].mbb_index.value
    elif model.startswith('s'):
        beta = sky.components[0].pl_index.value

    # Downgrade beta to 128
    alm_beta = hp.map2alm(beta, lmax=200)
    alm_beta_masked = hp.map2alm((beta - np.mean(beta[mask_ug > 0.1])) * mask_ug, lmax=200)    
    beta_128 = hp.alm2map(alm_beta, 128)
    mask_ug128 = hp.ud_grade(mask, 128)

    pmap = 4 * np.pi / mask_ug128.size
    w2 = np.sum((mask_ug128 ** 2) * pmap) / np.pi / 4.
    print(f'{w2=}')
    
    curvedsky.alm2map(alm_beta, omap, ainfo=ainfo)
    plot = enplot.plot(omap, colorbar=True, ticks=30, quantile=0.05)
    enplot.write(opj(imgdir, f'beta_{model}'), plot)


    curvedsky.alm2map(alm_beta_masked, omap, ainfo=ainfo)
    plot = enplot.plot(omap, colorbar=True, ticks=30, quantile=0.05)
    enplot.write(opj(imgdir, f'beta_masked_{model}'), plot)

    
    #print(sky.components[0].freq_ref_P)    
    #print(model, np.mean(beta), np.std(beta[mask_ug > 0.1]))
    sigma = np.std(beta_128[mask_ug128 > 0.1])
    print(model, np.mean(beta_128), sigma) 

    
    ax.hist(beta_128[mask_ug128 > 0.1], density=True, histtype='step',
            label=f'{model}: sigma = {sigma:.4f}')
    
    cl_beta = hp.alm2cl(alm_beta_masked) / w2

    
    print(np.sqrt(get_noise_variance(cl_beta)))

    b_amp, gamma = fit_power_law(cl_beta)
    ells = np.arange(cl_beta.size)
    ax2.plot(cl_beta,
             label=model + f' B={b_amp:.3f}, gamma={gamma:.3f}',
             color=f'C{midx}')
    
    ax2.plot(ells, b_amp * ells ** gamma, ls='dashed', color=f'C{midx}')
    
    
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'beta_hist'))
plt.close(fig)

ax2.set_yscale('log')
ax2.legend(frameon=False)
fig2.savefig(opj(imgdir, 'cl_beta'))
plt.close(fig2)
    
    
    #beta_dust_mean = np.sum(beta_dust * mask_ug) / np.sum(mask_ug)
    ##sky.components[0].mbb_index[:] = (beta_dust - beta_dust_mean) * scaling + beta_dust_mean
    #t_dust = sky.components[0].mbb_temperature
    #sky.components[0].mbb_temperature[:] = np.sum(t_dust * mask_ug) / np.sum(mask_ug)

    
    # for freq in freqs:
        
    #     central_freq = central_freq_dict[freq]

    #     imap = sky.get_emission(central_freq * up.Hz)

    #     # Convert to CMB units.
    #     imap = imap.to(up.uK_CMB, equivalencies=up.cmb_equivalencies(central_freq * up.Hz))
        
    #     # Downgrade
    #     lmax = 200
    #     # Conver to T, E, B.
    #     alm = hp.map2alm(imap, 2 * nside, use_pixel_weights=True)
    #     imap = hp.alm2map(alm[2], nside)
    #     alm = hp.map2alm(imap * galmask_dict[nside], lmax, use_pixel_weights=True)        
        
        #hp.write_alm(opj(odir, f'pysm_{model}_{freq}.fits'), alm, overwrite=True)
