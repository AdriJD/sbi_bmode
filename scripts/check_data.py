import os

import numpy as np
import matplotlib.pyplot as plt

opj = os.path.join

idir = '/mnt/ceph/users/aduivenvoorden/project/so/20240521_sbi_bmode/run20'
imgdir = opj(idir, 'img')

os.makedirs(imgdir, exist_ok=True)

data_true = np.load(opj(idir, f'data_norm.npy'))
#data_true = np.load(opj(idir, f'data.npy'))

fig, ax = plt.subplots(dpi=300)
for ridx in range(3):
    data = np.load(opj(idir, f'data_draws_round_{ridx:03d}.npy'))

    nsim = data.shape[0]
    for idx in range(0,nsim,10):
        ax.plot(data[idx], color=f'C{ridx}', alpha=0.5, lw=0.5, label=f'round {ridx + 1}' if idx == 0 else None)
        
ax.plot(data_true, color='black', lw=0.5, label='data')
#ax.set_yscale('symlog', linthresh=1e-6)
ax.set_xlabel(r'bin')
ax.set_ylabel(r'$C_{\ell}$')
#ax.set_ylim(1e-6)
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'data_draws'))
plt.close(fig)
    

fig, ax = plt.subplots(dpi=300)
for ridx in range(3):
    data = np.load(opj(idir, f'data_draws_round_{ridx:03d}.npy'))

    nsim = data.shape[0]
    for idx in range(0,nsim,10):
        ax.plot(data[idx] - data_true, color=f'C{ridx}', alpha=0.5, lw=0.5, label=f'round {ridx + 1}' if idx == 0 else None)
        
#ax.set_yscale('symlog', linthresh=1e-6)
ax.set_xlabel(r'bin')
ax.set_ylabel(r'$C_{\ell}$')
#ax.set_ylim(1e-6)
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'data_draws_residual'))
plt.close(fig)
    

