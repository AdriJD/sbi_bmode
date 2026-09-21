import os
import numpy as np
import matplotlib.pyplot as plt

opj = os.path.join

basedir = '/ptmp/bing/2026_sbi_bmode'
tarpdir = opj(basedir, 'tarp89t_d')
imgdir = opj(tarpdir, 'img')
os.makedirs(imgdir, exist_ok=True)

alpha = np.load(opj(tarpdir, 'tarp_alpha.npy'))
ecp = np.load(opj(tarpdir, 'tarp_ecp.npy'))
ecp_boot = np.load(opj(tarpdir, 'tarp_ecp_boot.npy'))
ecp_std = np.std(ecp_boot, axis=0)

ecp_marg = np.load(opj(tarpdir, 'tarp_ecp_marg.npy'))
ecp_marg_boot = np.load(opj(tarpdir, 'tarp_ecp_marg_boot.npy'))
ecp_marg_std = np.std(ecp_marg_boot, axis=1)

fig, ax = plt.subplots(figsize=(3.55, 3.55), dpi=300)

# Ideal calibration
ax.plot(alpha, alpha, 'k--', lw=1, label='Ideal')

# Joint TARP
ax.plot(alpha, ecp, lw=1, label='Joint TARP')
ax.fill_between(
    alpha,
    ecp - 2 * ecp_std,
    ecp + 2 * ecp_std,
    alpha=0.3
)

# Marginal TARP
for pidx, param in zip([0, 1], [r'$r$', r'$A_{\mathrm{lens}}$']):
    ax.plot(alpha, ecp_marg[pidx], lw=1, label=param)
    ax.fill_between(
        alpha,
        ecp_marg[pidx] - 2 * ecp_marg_std[pidx],
        ecp_marg[pidx] + 2 * ecp_marg_std[pidx],
        alpha=0.3
    )

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

ax.set_xlabel(r'Credibility level $1-\alpha$')
ax.set_ylabel(r'$\mathrm{ECP}$')

ax.tick_params(
    'both', which='both',
    direction='in', right=True, top=True
)

ax.grid(
    color='black',
    linestyle='dotted',
    linewidth=0.5
)

ax.legend(frameon=False, loc='upper left')

fig.savefig(opj(imgdir, 'tarp.png'), bbox_inches='tight')
fig.savefig(opj(imgdir, 'tarp.pdf'), bbox_inches='tight')

plt.show()