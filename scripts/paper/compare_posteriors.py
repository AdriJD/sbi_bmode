import os

import numpy as np
from scipy.stats import halfnorm, multivariate_normal
from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'

#n_params = 5
#param_names = ['r_tensor', 'A_lens', 'A_d_BB', 'alpha_d_BB', 'beta_dust']
param_names = ['r_tensor', 'A_lens', 'A_d_BB', 'alpha_d_BB', 'beta_dust', 'A_s_BB',
               'alpha_s_BB', 'beta_sync', 'rho_ds']
#param_labels = [r'$r$', r'$A_L$', r'$A_d^{BB}$', r'$\alpha_d^{BB}$', r'$\beta_d$']
param_labels = [r'$r$', r'$A_L$', r'$A_d^{BB}$', r'$\alpha_d^{BB}$', r'$\beta_d$',
                r'$A_s^{BB}$', r'$\alpha_s^{BB}$', r'$\beta_s$', r'$\rho_{ds}$']

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

r_true = 0.005

#param_truths = [0.005, 1, 5, -0.2, 1.59]
param_truths = [0.01, 0.45, 4., -0.3, 1.55, 1.5, -0.6, -2.8, 0.1]

samples_g = []
oname = '55'
dirs2compare = ['mcmc55']
for sidx, subdir in enumerate(dirs2compare):

    idir = opj(basedir, subdir)
    if sidx == len(dirs2compare) - 1:
        imgdir = opj(idir, 'img')
        os.makedirs(imgdir, exist_ok=True)
    samples = np.load(opj(idir, 'samples.npy'))
    samples = getdist_MCSamples(
        samples=samples, names=param_labels, labels=param_labels_g,
        label=f'{subdir}', ranges={'$r$': [0, None]})
    samples_g.append(samples)

g = getdist_plots.get_subplot_plotter(width_inch=8)
g.triangle_plot(samples_g, filled=True,
                markers=param_truths,
                contour_colors=['C%d'%i for i in range(len(dirs2compare))],
                )
g.export(opj(imgdir, f'{oname}_corner.png'), dpi=300)
