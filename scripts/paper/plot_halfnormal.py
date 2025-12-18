import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples
import torch

from sbi_bmode import script_utils

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
imgdir = opj(basedir, 'test_halfnormal')

os.makedirs(imgdir, exist_ok=True)

distr = torch.distributions.HalfNormal(torch.tensor([0.5]))

nsamp1 = 500_000
nsamp2 = 500_000

samples1 = np.asarray(distr.sample((nsamp1,)))
samples2 = np.asarray(distr.sample((nsamp2,)))

fig, ax = plt.subplots(dpi=300)
ax.hist(samples1, bins=100, histtype='step', density=True)
ax.hist(samples2, bins=100, histtype='step', density=True)
fig.savefig(opj(imgdir, 'hist'))
plt.close(fig)

prior_samples1 = getdist_MCSamples(
    samples=samples1, names=['r_tensor'],
    ranges={'r_tensor' : (0.0, None)},
    label=f'{nsamp1} samples, 0.2')
prior_samples2 = getdist_MCSamples(
    samples=samples2, names=['r_tensor'],
    ranges={'r_tensor' : (0.0, None)},
    label=f'{nsamp2} samples, 0.5')

prior_samples1.smooth_scale_1D = 0.2
prior_samples2.smooth_scale_1D = 0.5

prior_samples1.boundary_correction_order = 0
prior_samples2.boundary_correction_order = 0

prior_samples1.mult_bias_correction_order = 1
prior_samples2.mult_bias_correction_order = 1

samples_g = []
samples_g.append(prior_samples1)
samples_g.append(prior_samples2)

g = getdist_plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(samples_g, filled=False)

#for i, name in enumerate(param_names):
#    # This will add a line to the 1D marginal only
#    g.add_1d(prior_samples, param=name, ls='--', color='gray', label='prior', ax=g.subplots[i, i])

g.export(opj(imgdir, f'getist_corner.png'), dpi=300)
