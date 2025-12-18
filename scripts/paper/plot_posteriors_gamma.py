import os

import numpy as np
from scipy.stats import halfnorm, multivariate_normal
from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples

opj = os.path.join

basedir = '/mnt/home/aduivenvoorden/project/so/20240521_sbi_bmode'

param_names = ['r_tensor', 'A_lens', 'A_d_BB', 'alpha_d_BB', 'beta_dust', 'amp_beta_dust', 'gamma_beta_dust']
param_labels = [r'$r$', r'$A_L$', r'$A_d^{BB}$', r'$\alpha_d^{BB}$', r'$\beta_d$', r'$A_{\beta}$', r'$\gamma_{\beta}$']
param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

param_truths = [0.01, 1.05, 4, -0.3, 1.55, 0.4, -3.5]

param_limits = {'r_tensor' : (0., None), 'A_lens' : (0., None), 'A_d_BB' : (0., None),
                'amp_beta_dust' : (0., 1.59), 'gamma_beta_dust' : (-6., -1.5)}

samples_g = []
rundir = 'run29'
idir = opj(basedir, rundir)
imgdir = opj(idir, 'img')
os.makedirs(imgdir, exist_ok=True)
nrounds = 3

for ridx in range(nrounds):

    
    samples = np.load(opj(idir, f'samples_round_{ridx:03d}.npy'))
    
    samples = getdist_MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=f'round_{ridx}')
    
    samples_g.append(samples)

g = getdist_plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(samples_g, filled=True,
                markers=param_truths,
                contour_colors=['C%d'%i for i in range(nrounds)],
                )
g.export(opj(imgdir, f'rounds_corner.png'), dpi=300)
