import os

import numpy as np
from scipy.stats import halfnorm, multivariate_normal
from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples

opj = os.path.join

basedir = '/mnt/home/aduivenvoorden/project/so/20240521_sbi_bmode/run13i'
imgdir = opj(basedir, 'img')
os.makedirs(imgdir, exist_ok=True)

n_params = 5
param_names = ['r_tensor', 'A_lens', 'A_d_BB', 'alpha_d_BB', 'beta_dust']
param_labels = [r'$r$', r'$A_L$', r'$A_d^{BB}$', r'$\alpha_d^{BB}$', r'$\beta_d$']
param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

def draw_prior():

    ndraws = 10_000
    draws = np.zeros((ndraws, 5))

    draws[:,0] = halfnorm.rvs(loc=0, scale=0.1, size=ndraws)
    draws[:,1:] = multivariate_normal.rvs(
        mean=[1, 5, -0.2, 1.59], cov=np.diag(np.asarray([0.1, 2, 0.5, 0.11]) ** 2), size=ndraws)
    return draws

#prior_draws = draw_prior()
#prior_samples = getdist_MCSamples(
#    samples=prior_draws, names=param_labels, labels=param_labels_g)

#for r_true in [0.01, 0.05]:
for r_true in [0.005]:    

    param_truths = [r_true, 1, 5, -0.2, 1.59]
    #param_truths = [r_true, 1.1, 3, 0.2, 1.5]    
    samples_g = []
    
    #for nsim in [500, 1000, 2000, 4000]:
    #for nsim in [2000, 3000]:
    #for nsim in [3000]:
    #for nsim in [10000]:
    #for nsim in [50000]:
    for nsim in [10000]:                        

        #idir = opj(basedir, f'r{r_true:.2e}_s-1_nt{nsim}_ns10000_nr1')
        #idir = opj(basedir, f'r{r_true:.2e}_s225186655513525153114758457104258967436_nt{nsim}_ns10000_nr3')
        idir = basedir
        samples = np.load(opj(idir, 'samples.npy'))
        samples = getdist_MCSamples(
            samples=samples, names=param_labels, labels=param_labels_g,
            label=f'{nsim}', ranges={'$r$': [0, None]})
        samples_g.append(samples)

        g = getdist_plots.get_subplot_plotter(width_inch=8)
        g.triangle_plot(samples, filled=True,
                        markers=param_truths
                       )
        #for i in range(n_params):
        #    #g.add_1d(prior_samples, param_labels[i], color='gray', ls='solid')
        #    g.plot_1d(prior_samples, param_labels[i], color='gray', ls='solid')        
        #g.add_1d(prior_samples, param_labels, color='gray', ls='solid')

        #g.triangle_plot(prior_samples, filled=True,
        #                markers=param_truths
        #                )


        g.export(opj(imgdir, f'corner_r{r_true:.2e}_{nsim}.png'))

    # samples = np.load(opj(idir, 'samples_mcmc.npy'))
    # samples = getdist_MCSamples(
    #     samples=samples, names=param_labels, labels=param_labels_g,
    #     label=f'mcmc', ranges={'$r$': [0, None]})
    # samples_g.append(samples)

    # g = getdist_plots.get_subplot_plotter(width_inch=8)
    # g.triangle_plot(samples, filled=True,
    #                 markers=param_truths
    #                )
    # g.export(opj(imgdir, f'corner_mcmc_r{r_true:.2e}_{nsim}.png'))

        
        
    g = getdist_plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(samples_g, filled=True,
                    markers=param_truths,
                    contour_colors=['C%d'%i for i in range(nsim)],
                    )
    g.export(opj(imgdir, f'r{r_true:.2e}_corner.png'))
