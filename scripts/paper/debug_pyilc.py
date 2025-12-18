import os

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

opj = os.path.join

#idir = '/u/adriaand/project/so/20240521_sbi_bmode/run44/tmpv52ya8rf'
#idir2 = '/u/adriaand/project/so/20240521_sbi_bmode/run44/tmplzq4ubiz'
idir = '/u/adriaand/project/so/20240521_sbi_bmode/run80/tmpk1kzlcoa'
idir2 = '/u/adriaand/project/so/20240521_sbi_bmode/run80/tmp_dj8iybq'
imgdir = '/u/adriaand/project/so/20240521_sbi_bmode/debug_pyilc_run80'

os.makedirs(imgdir, exist_ok=True)

for f1 in range(8):
    for split in range(2):
        imap = hp.read_map(opj(idir2, f'map_split{split}_freq{f1}.fits'), field=None)
        fig, ax = plt.subplots(dpi=300)
        plt.axes(ax)
        hp.mollview(imap, hold=True)
        fig.savefig(opj(imgdir, f'map_split{split}_freq{f1}'))
        plt.close(fig)
    
#exit()
for wscale in range(4):
    for f1 in range(8):
        print(wscale, f1)
        try:
            imap = hp.read_map(opj(idir, f'_needletcoeffmap_freq{f1}_scale{wscale}.fits'), field=None)
        except FileNotFoundError:
            continue
        else:

            fig, ax = plt.subplots(dpi=300)
            plt.axes(ax)
            hp.mollview(imap, hold=True)
            fig.savefig(opj(imgdir, f'_needletcoeffmap_freq{f1}_scale{wscale}'))
            plt.close(fig)
                    
        

#exit()

for wscale in range(4):

    fig, axs = plt.subplots(dpi=300, nrows=8, ncols=8, figsize=(10, 10), constrained_layout=True)
    
    for f1 in range(8):
        for f2 in range(f1, 8):

            imap = hp.read_map(opj(idir, f'_needletcoeff_covmap_freq{f1}_freq{f2}_scale{wscale}.fits'),
                               field=None)
            plt.axes(axs[f1,f2])
            hp.mollview(imap, hold=True)

    fig.savefig(opj(imgdir, f'cov_scale{wscale}'))
    plt.close(fig)

            
