import os

import numpy as np
import matplotlib.pyplot as plt

opj = os.path.join

imgdir = '/u/adriaand/project/so/20240521_sbi_bmode/pyilc'
os.makedirs(imgdir, exist_ok=True)

def get_gauss(ells, fwhm):

    return np.exp(-ells * (ells + 1) * fwhm ** 2 / 16 / np.log(2))

def get_kernels(lmax, fwhms):

    n_kernel = len(fwhms) + 1
    ells = np.arange(lmax + 1)
    kernels = np.zeros((n_kernel, lmax+1))
    
    for idx in range(n_kernel):

        if idx == 0:
            kernels[idx] = get_gauss(ells, fwhms[idx])

        elif idx == n_kernel - 1:
            kernels[idx] = np.sqrt(1 - get_gauss(ells, fwhms[idx-1]) ** 2)

        else:
            kernels[idx] = np.sqrt(get_gauss(ells, fwhms[idx]) ** 2 - get_gauss(ells, fwhms[idx-1]) ** 2)

    return kernels

def get_nilc_fwhms(kernels, n_deproj, n_freq, b_tol):

    n_kernel = kernels.shape[0]
    nilc_fwhms = np.zeros(n_kernel)
    ells = np.arange(kernels.shape[-1])
    
    for idx in range(n_kernel):

        n_modes = np.sum((2 * ells + 1) * kernels[idx] ** 2)
        nilc_fwhms[idx] = np.sqrt(8 * np.log(2) * 2 * (np.abs(1 + n_deproj - n_freq) / (b_tol * n_modes)))

    return nilc_fwhms

n_deproj = 4
n_freq = 8
b_tol = 0.01
#lmax = 200
lmax = 3 * 128 - 2
fwhms = np.radians(np.asarray([300, 120, 60]) / 60)
kernels = get_kernels(lmax, fwhms)
nilc_fwhms = get_nilc_fwhms(kernels, n_deproj, n_freq, b_tol)

fig, ax = plt.subplots(dpi=300)
for idx in range(kernels.shape[0]):
    ax.plot(kernels[idx], label=f'FWHM={np.degrees(nilc_fwhms[idx])}')
ax.legend()
fig.savefig(opj(imgdir, 'nilc_kernels'))
plt.close(fig)



fig, ax = plt.subplots(dpi=300)
ax.plot(np.sum(kernels ** 2, axis=0), label=f'FWHM={nilc_fwhms[idx]}')
fig.savefig(opj(imgdir, 'nilc_kernels_squared'))
plt.close(fig)
