import os
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.text as mtext

matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

from getdist import plots as getdist_plots
from getdist import MCSamples as getdist_MCSamples
import torch

from sbi_bmode import script_utils

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'

def draw_from_prior(prior_list, nsamp):
    '''
    Extract prior limits.

    Parameters
    ----------
    prior_list : list of torch.distribition instances
        Priors.
    nsamp : int
        Number of prior draws.
    
    Returns
    -------
    draw : (nsamp, n_prior) array
        Prior draws.
    '''

    out = np.zeros((nsamp, len(prior_list)))

    for pidx, prior in enumerate(prior_list):
        out[:,pidx] = np.asarray(prior.sample((nsamp,)))[:,0]
        
    return out
    
param_label_dict = {'r_tensor' : r'$100r$',
                    'A_lens' : r'$A_{\mathrm{lens}}$',
                    'A_d_BB' : r'$A_{\mathrm{d}}$',
                    'alpha_d_BB' : r'$\alpha_{\mathrm{d}}$',
                    'beta_dust' : r'$\beta_{\mathrm{d}}$',
                    'A_s_BB' : r'$A_{\mathrm{s}}$',
                    'alpha_s_BB' : r'$\alpha_{\mathrm{s}}$',
                    'beta_sync' : r'$\beta_{\mathrm{s}}$',
                    'rho_ds' : r'$100\rho_{\mathrm{ds}}$'}

#oname = '66t73_mcmc66'
oname = '66t73_mcmc66_rerun2'
dirs2compare = ['mcmc66_rerun2', 'run66_optuna/trial_0073']
labels = ['Multi-frequency likelihood', 'Joint NILC SBI']

with open(opj(basedir, 'run66', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)
    
data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)
# Scale r and rho.
params_dict['r_tensor']['prior_params'][0] *= 100
params_dict['r_tensor']['true_value'] *= 100
params_dict['rho_ds']['prior_params'][1] *= 100
params_dict['rho_ds']['true_value'] *= 100
prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $

#param_truths = [0.005, 1, 5, -0.2, 1.59]
#param_truths = [0.01, 1.05, 4, -0.3, 1.55, 0.4, -3.5]

#param_limits = {'r_tensor' : (0., None), 'A_lens' : (0., None), 'A_d_BB' : (0., None),
#                'amp_beta_dust' : (0., 1.59), 'gamma_beta_dust' : (-6., -1.5)}
param_limits = script_utils.get_param_limits(prior, param_names)

# Sample from prior.

print(f'{param_names=}')
print(f'{param_labels=}')
print(f'{param_truths=}')
print(f'{param_limits=}')

torch.manual_seed(0)
nsamp = 500_000
prior_draw = draw_from_prior(prior, nsamp)

getdist_plots.default_settings.figure_legend_frame = False
getdist_plots.default_settings.norm_1d_density = False
#getdist_plots.default_settings.solid_colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 1, len(dirs2compare)+1)]
getdist_plots.default_settings.figure_legend_ncol = 1

getdist_plots.default_settings.fontsize = 10
getdist_plots.default_settings.axes_fontsize = 10
getdist_plots.default_settings.legend_fontsize = 12
getdist_plots.default_settings.axes_labelsize = 12

getdist_plots.default_settings.scaling = False


colors = ['C%d'%i for i in range(len(dirs2compare))]
#colors = ['black'] + colors

samples_g = []
#for sidx, subdir in enumerate(dirs2compare):

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits,
    label=f'prior')
prior_samples.smooth_scale_1D = 0.2

#samples_g.append(prior_samples) # NOTE

for sidx, (subdir, label) in enumerate(zip(dirs2compare, labels)):

    idir = opj(basedir, subdir)
    if sidx == len(dirs2compare) - 1:
        imgdir = opj(idir, 'img')
        os.makedirs(imgdir, exist_ok=True)
    samples = np.load(opj(idir, 'samples.npy'))

    print(samples.shape)

    if samples.ndim == 3:
        # Flatten chain dim.
        samples = samples.reshape((samples.shape[0] * samples.shape[1], -1))
    if samples.ndim == 4:
        # Flatten chain dim.
        assert samples.shape[0] == 1
        samples = samples.reshape((samples.shape[1] * samples.shape[2], -1))
        
    print('after', samples.shape)

    # Scale r and rho.
    samples[:,0] *= 100
    samples[:,-1] *= 100    
       
    
    samples = getdist_MCSamples(
        samples=samples, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=f'{label}')
    
    samples_g.append(samples)
        
g = getdist_plots.get_subplot_plotter(width_inch=7.1)
g.triangle_plot(samples_g, filled=True,
                markers=param_truths,
                contour_colors=colors
                )

for i, name in enumerate(param_names):
    # This will add a line to the 1D marginal only
    g.add_1d(prior_samples, param=name, ls='solid', color='gray', label='prior', ax=g.subplots[i, i],
             zorder=1)

leg = g.fig.legends[0]

handles, legend_labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
prior_handle = Line2D([], [], color='gray', ls='solid', label='Prior', lw=1.5)

# Re-order.
handles = handles + [prior_handle]
legend_labels = legend_labels + ['Prior']

for L in g.fig.legends:  # remove old legend(s)
    L.remove()

g.fig.legend(handles,
             legend_labels,
             loc=(0.4, 0.85),
             frameon=g.settings.figure_legend_frame,
             #fontsize=10,
             ncols=g.settings.figure_legend_ncol)

g.export(opj(imgdir, f'{oname}_corner.png'), dpi=300)
g.export(opj(imgdir, f'{oname}_corner.pdf'), dpi=300)
