import os

import numpy as np
import healpy as hp
from pixell import enmap, enplot, curvedsky

opj = os.path.join

almdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
imgdir = opj(almdir, 'img')
os.makedirs(imgdir, exist_ok=True)

freqs = ['wK', 'f030', 'f040', 'f090', 'f150', 'f230', 'f290', 'p353']

#dust_models = ['d1', 'd4', 'd10', 'd12']
#sync_models = ['s5', 's7', ]
dust_models = ['d0']
sync_models = []

lmax = 200
oversample = 2
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                        2 * np.pi / (2 * oversample * lmax + 1)])
ainfo = curvedsky.alm_info(lmax)
omap = enmap.zeros(shape, wcs)

for model in (dust_models + sync_models):

    for freq in freqs:
    
        alm = hp.read_alm(opj(almdir,  f'pysm_{model}_{freq}.fits'))
        curvedsky.alm2map(alm, omap, ainfo=ainfo)

        plot = enplot.plot(omap, colorbar=True, ticks=30, quantile=0.05)
        enplot.write(opj(imgdir, f'pysm_{model}_{freq}'), plot)
    
