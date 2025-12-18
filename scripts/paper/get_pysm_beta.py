import os

import numpy as np
import healpy as hp
import pysm3
import pysm3.units as up

opj = os.path.join

odir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
imgdir = opj(odir, 'img')
os.makedirs(odir, exist_ok=True)
os.makedirs(imgdir, exist_ok=True)

sky = pysm3.Sky(nside=2048, preset_strings=["d10"])
dust = sky.components[0]
beta_d = dust.mbb_index

beta_d = hp.alm2map(hp.map2alm(beta_d, 200, use_pixel_weights=True), 128)

hp.write_map(opj(odir, 'pysm_d10_beta.fits'), beta_d)
