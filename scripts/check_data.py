import os

import numpy as np
import matplotlib.pyplot as plt

opj = os.path.join

idir = '/u/adriaand/project/so/20240521_sbi_bmode/run41'
imgdir = opj(idir, 'img')

os.makedirs(imgdir, exist_ok=True)

data_true = np.load(opj(idir, f'data_norm.npy'))
#data_true = np.load(opj(idir, f'data.npy'))
data_true_unnorm = np.load(opj(idir, f'data.npy'))

data_mean = np.load(opj(idir, 'data_mean.npy'))
data_std = np.load(opj(idir, 'data_std.npy'))

fig, ax = plt.subplots(dpi=300)
for ridx in range(100):
    try:
        data = np.load(opj(idir, f'data_draws_round_{ridx:03d}.npy'))
    except:
        break
    nsim = data.shape[0]
    for idx in range(0,nsim,10):
        ax.plot(data[idx], color=f'C{ridx}', alpha=0.5, lw=0.5, label=f'round {ridx + 1}' if idx == 0 else None)
        
ax.plot(data_true, color='black', lw=0.5, label='data')
ax.set_yscale('symlog', linthresh=1e-1)
ax.set_xlabel(r'bin')
ax.set_ylabel(r'$C_{\ell}$')
#ax.set_ylim(1e-6)
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'data_draws'))

#ax.set_ylim(-5, 5)
#fig.savefig(opj(imgdir, 'data_draws_zoom'))
#plt.close(fig)

fig, ax = plt.subplots(dpi=300)
for ridx in range(100):
    try:
        data = np.load(opj(idir, f'data_draws_round_{ridx:03d}.npy'))                       
    except:
        break
    else:
        data = data * data_std + data_mean
    nsim = data.shape[0]
    for idx in range(0,nsim,10):
        ax.plot(data[idx], color=f'C{ridx}', alpha=0.5, lw=0.5, label=f'round {ridx + 1}' if idx == 0 else None)
        
ax.plot(data_true_unnorm, color='black', lw=0.5, label='data')
ax.set_yscale('symlog', linthresh=1e-5)
ax.set_xlabel(r'bin')
ax.set_ylabel(r'$C_{\ell}$')
ax.set_ylim(2 * data_true_unnorm.min(), 2 * data_true_unnorm.max())
ax.legend(frameon=False)
fig.savefig(opj(imgdir, 'data_draws_unnorm'))
                       

fig, ax = plt.subplots(dpi=300)
for ridx in range(100):
    try:
        data = np.load(opj(idir, f'data_draws_round_{ridx:03d}.npy'))
    except:
        break

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
    

