'''
Combine SAT mask with Planck mask.
'''

import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pixell import enmap, enplot, reproject

opj = os.path.join

maskdir = '/u/adriaand/project/so/20250612_sat_mask'
maskdir_planck = '/u/adriaand/project/planck/20250825_2015_gal_masks'
pysmdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
imgdir = opj(pysmdir, 'img')
odir = opj(pysmdir, 'masks')
os.makedirs(odir, exist_ok=True)

nside = 128
lmax = 200
oversample = 1
shape, wcs = enmap.fullsky_geometry(res=[np.pi / (oversample * lmax),
                                        2 * np.pi / (2 * oversample * lmax + 1)])

# Load SAT mask.
mask = hp.read_map(opj(maskdir, 'mask_apo10.0_MSS2_SAT1_f090_coadd_gal.fits'))
mask = hp.ud_grade(mask, nside)

# Load 70% sky mask.
galmask = hp.read_map(opj(maskdir_planck, 'HFI_Mask_GalPlane-apo0_2048_R2.00.fits'), field=3)
galmask = galmask.astype(np.float64)
galmask = hp.ud_grade(galmask, nside)

hp.mollview(galmask)
fig = plt.gcf()
fig.savefig(opj(imgdir, "galmask_3"))
plt.close(fig)

small_pix = galmask < 0.5
galmask[small_pix] = 0
galmask[~small_pix] = 1

mask *= galmask

# Project to pixell
mask_enmap = reproject.healpix2map(mask, shape=shape, wcs=wcs, method='spline', order=0)
small_pix = mask_enmap < 1e-4
mask_enmap[small_pix] = 0
mask_enmap[~small_pix] = 1

# Apodize
mask_enmap = enmap.apod_mask(mask_enmap, width=np.radians(10))

plot = enplot.plot(mask_enmap, colorbar=True, ticks=30, quantile=0.)
enplot.write(opj(imgdir, f'mask'), plot)

# project back to healpy
mask = reproject.map2healpix(mask_enmap, nside=nside, method='spline', order=3)

hp.mollview(mask)
fig = plt.gcf()
fig.savefig(opj(imgdir, "mask_hp"))
plt.close(fig)

# Save at nside 128
hp.write_map(opj(odir, 'mask.fits'), mask, overwrite=True, dtype=np.float32)

mask_enmap = reproject.healpix2map(mask, shape=shape, wcs=wcs, method='spline', order=0)
plot = enplot.plot(mask_enmap, colorbar=True, ticks=30, quantile=0.)
enplot.write(opj(imgdir, f'mask2'), plot)
