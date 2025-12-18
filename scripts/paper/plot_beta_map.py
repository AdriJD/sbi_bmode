import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import ListedColormap
import healpy as hp
from pixell import enmap, reproject, cgrid

from mnms import utils

opj = os.path.join

pysmdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(pysmdir, 'masks')
imgdir = opj(pysmdir, 'img_beta')

os.makedirs(imgdir, exist_ok=True)


matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})


beta_d = hp.read_map(opj(pysmdir, 'pysm_d10_beta.fits'))

lmax = 200
oversample = 4
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                         2 * np.pi / (2 * oversample * lmax + 1)],
                                    variant='fejer1')

beta_enmap = reproject.healpix2map(beta_d, shape=shape, wcs=wcs, method='spline', order=1)

fig, ax = plt.subplots(dpi=300, figsize=(3.35, 1.75), constrained_layout=True)

lmin, lmax = 0, 360
bmin, bmax = -90, 90

im = ax.imshow(beta_enmap, extent=[lmin, lmax, bmin, bmax],
               origin='lower')

cbar = fig.colorbar(im, ax=ax, shrink=0.74, aspect=12, pad=-0.02, extend='both')
cbar.set_label(r'$\beta_{\mathrm{d}}$', size=10)
cbar.ax.tick_params(labelsize=8)

yticks = [-90, -45, 0, 45, 90]
ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in yticks]))

xticks = [180, 120, 60, 0, 360, 300, 240] 
labels = ['0', '60', '120', '180', '180', '240', '300']
ax.set_xticks(xticks)
ax.set_xticklabels(labels)

ax.set_ylabel(r'$b$ [$^\circ$]', fontsize=10)
ax.set_xlabel(r'$l$ [$^\circ$]', fontsize=10)

ax.tick_params(labelsize=8)

fig.savefig(opj(imgdir, 'beta_d10'))
plt.close(fig)
                       
                       


