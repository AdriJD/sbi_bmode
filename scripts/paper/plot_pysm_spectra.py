import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

opj = os.path.join

pysmdir = '/u/adriaand/project/so/20240521_sbi_bmode/pysm'
maskdir = opj(pysmdir, 'masks')
idir = opj(pysmdir, 'spectra')
testdir = '/u/adriaand/project/so/20240521_sbi_bmode/run75'
imgdir = opj(pysmdir, 'img_compare_run75')
os.makedirs(imgdir, exist_ok=True)

#dust_models = ['d1', 'd4', 'd10', 'd12']
#sync_models = ['s5', 's7']
dust_models = ['d1', 'd12']
sync_models = ['s7']

mask = hp.read_map(opj(maskdir, 'mask.fits'))
npix = mask.size
pmap = 4 * np.pi / npix
w2 = np.sum((mask ** 2) * pmap) / np.pi / 4.
print(f'{w2=}')

testset = np.load(opj(testdir, 'data_draws_test_mf.npy'))
mean = np.mean(testset, axis=0)
std = np.std(testset, axis=0)

fig, ax = plt.subplots(dpi=300)

for sim_idx in range(0, testset.shape[0]):
    ax.plot(testset[sim_idx], color='gray', alpha=0.2, lw=0.5, zorder=0)

for dust_model in dust_models:
    for sync_model in sync_models:
        print(dust_model, sync_model)

        data = np.load(opj(idir, f'spectra_{dust_model}_{sync_model}.npy'))
        # data /= w2 # NOTE
        
        ax.plot(data, label=f'{dust_model}_{sync_model}', lw=0.5)

ax.set_yscale('symlog', linthresh=0.0005)
ax.legend(frameon=False, ncols=3)
fig.savefig(opj(imgdir, 'pysm_spectra'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)

for sim_idx in range(0, testset.shape[0]):
    ax.plot((testset[sim_idx] - mean), color='gray', alpha=0.2, lw=0.5, zorder=0)

for dust_model in dust_models:
    for sync_model in sync_models:
        print(dust_model, sync_model)

        data = np.load(opj(idir, f'spectra_{dust_model}_{sync_model}.npy'))
        #data /= w2
        
        ax.plot((data - mean), label=f'{dust_model}_{sync_model}', lw=0.5)

#ax.set_yscale('symlog', linthresh=0.0005)
ax.legend(frameon=False, ncols=3)
fig.savefig(opj(imgdir, 'pysm_spectra_norm_mean'))
plt.close(fig)

fig, ax = plt.subplots(dpi=300)

for sim_idx in range(0, testset.shape[0]):
    ax.plot((testset[sim_idx] - mean) / std, color='gray', alpha=0.2, lw=0.5, zorder=0)

for dust_model in dust_models:
    for sync_model in sync_models:
        print(dust_model, sync_model)

        data = np.load(opj(idir, f'spectra_{dust_model}_{sync_model}.npy'))
        #data /= w2
        
        ax.plot((data - mean) / std, label=f'{dust_model}_{sync_model}', lw=0.5)

#ax.set_yscale('symlog', linthresh=0.0005)
ax.legend(frameon=False, ncols=3)
fig.savefig(opj(imgdir, 'pysm_spectra_norm_std'))
plt.close(fig)
