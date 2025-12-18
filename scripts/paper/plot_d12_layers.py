import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pysm3

from pixell import enmap, reproject, curvedsky, enplot

opj = os.path.join

odir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(odir, 'masks')
imgdir = opj(odir, 'img', 'd12_layers')
os.makedirs(imgdir, exist_ok=True)

lmax = 2048
ainfo = curvedsky.alm_info(lmax)
oversample = 1
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                        2 * np.pi / (2 * oversample * lmax + 1)])
omap = enmap.zeros((3,) + shape, wcs)

sky = pysm3.Sky(nside=2048, preset_strings=['d12'])

num_layers = sky.components[0].mbb_index.shape[0]

mask = hp.read_map(opj(maskdir, 'mask.fits'))
mask_2048 = hp.ud_grade(mask, 2048)

for lidx in range(num_layers):

    print(lidx)
    
    amp = sky.components[0].layers[lidx] # IQU.
    beta = sky.components[0].mbb_index[lidx]
    temp = sky.components[0].mbb_temperature[lidx]

    for pidx in range(3):        
        hp.mollview(amp[pidx])
        plt.savefig(opj(imgdir, f'amp_{lidx}_{pidx}'))
        plt.close()


    for pidx in range(3):
        vmin = np.quantile(amp[pidx].value * mask_2048, 0.02)
        vmax = np.quantile(amp[pidx].value * mask_2048, 0.98)
        hp.mollview(amp[pidx] * mask_2048, min=vmin, max=vmax)
        plt.savefig(opj(imgdir, f'amp_masked_{lidx}_{pidx}'))
        plt.close()
        
    hp.mollview(beta)
    plt.savefig(opj(imgdir, f'beta_{lidx}'))
    plt.close()

    hp.mollview(beta * mask_2048)
    plt.savefig(opj(imgdir, f'beta_masked_{lidx}'))
    plt.close()
    
    hp.mollview(temp)
    plt.savefig(opj(imgdir, f'temp_{lidx}'))
    plt.close()

    alm_beta = hp.map2alm(beta, 200)
    beta_128 = hp.alm2map(alm_beta, 128)

    print(lidx, np.std(beta_128[mask > 0.1]))
    
    hp.mollview(beta_128 * mask)
    plt.savefig(opj(imgdir, f'beta_128_masked_{lidx}'))
    plt.close()
    
    

    
    # alm = hp.map2alm(amp)
    # curvedsky.alm2map(alm, omap, ainfo=ainfo)
    # for pidx in range(3):
    #     plot = enplot.plot(omap[pidx], colorbar=True, ticks=30, quantile=0.05)
    #     enplot.write(opj(imgdir, f'amp_{lidx}_{pidx}'), plot)

    # alm = hp.map2alm(beta)
    # curvedsky.alm2map(alm, omap[0], ainfo=ainfo)    
    # plot = enplot.plot(omap[0], colorbar=True, ticks=30, quantile=0.05)
    # enplot.write(opj(imgdir, f'beta_{lidx}'), plot)

    # alm = hp.map2alm(temp)
    # curvedsky.alm2map(alm, omap[0], ainfo=ainfo)        
    # plot = enplot.plot(omap[0], colorbar=True, ticks=30, quantile=0.05)
    # enplot.write(opj(imgdir, f'temp_{lidx}'), plot)
    


