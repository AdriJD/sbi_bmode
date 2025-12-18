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
                    'A_lens' : r'$A_{\mathrm{lens}}$'}

#oname = '66t73_mcmc66_cosmo_only'
oname = '66t73_mcmc66_rerun2_cosmo_only'
#imgdir = opj(basedir, 'run65_optuna_b/trial_0109')
imgdir = opj(basedir, 'run66_optuna/trial_0073/img')
os.makedirs(imgdir, exist_ok=True)

subdirs = ['mcmc66_rerun2']
subdirs_b = ['run66_optuna/trial_0073']

labels = ['MF like']          
labels_b = ['SBI']

with open(opj(basedir, 'run66', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)

data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)

params_dict.pop('A_d_BB')
params_dict.pop('alpha_d_BB')
params_dict.pop('beta_dust')
params_dict.pop('A_s_BB')
params_dict.pop('alpha_s_BB')
params_dict.pop('beta_sync')
params_dict.pop('rho_ds')

params_dict['r_tensor']['prior_params'][0] *= 100
params_dict['r_tensor']['true_value'] *= 100

prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
#param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $
param_limits = script_utils.get_param_limits(prior, param_names)

torch.manual_seed(0)
nsamp = 500_000
prior_draw = draw_from_prior(prior, nsamp)

getdist_plots.default_settings.figure_legend_frame = False
getdist_plots.default_settings.norm_1d_density = False
getdist_plots.default_settings.figure_legend_ncol = 1

getdist_plots.default_settings.fontsize = 9
getdist_plots.default_settings.axes_fontsize = 9

getdist_plots.default_settings.legend_fontsize = 9
getdist_plots.default_settings.axes_labelsize = 11

getdist_plots.default_settings.scaling = False

getdist_plots.default_settings.axis_marker_lw = 1.

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits, label='prior')

prior_samples.smooth_scale_1D = 0.2

samples_g = []

for sidx, (subdir, label) in enumerate(zip((subdirs + subdirs_b), (labels + labels_b))):

    samples_arr = np.load(opj(basedir, subdir, f'samples.npy'))

    if samples_arr.ndim == 3:
        # Flatten chain dim.
        samples_arr = samples_arr.reshape((samples_arr.shape[0] * samples_arr.shape[1], -1))
    if samples_arr.ndim == 4:
        assert samples_arr.shape[0] == 1
        # Flatten chain dim.
        samples_arr = samples_arr.reshape((samples_arr.shape[1] * samples_arr.shape[2], -1))
    
    samples_arr = samples_arr[:,:2]
    samples_arr[:,0] *= 100

    
    print(samples_arr.shape)
    
    samples = getdist_MCSamples(
        samples=samples_arr, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=label)
    
    samples_g.append(samples)

                            
g = getdist_plots.get_subplot_plotter(width_inch=3.35)
g.triangle_plot(samples_g, filled=True,
                markers={'r_tensor' : 0.01 * 100, 'A_lens' : 0.45},
                line_args=[{'lw': 1.5, 'ls' : 'solid', 'color' : c} for c in ['C0', 'C1']],
                contour_args=[{'lw': 1.5, 'ls' : 'solid', 'color' : c} for c in ['C0', 'C1']]                
                )

for i, name in enumerate(param_names):
    # This will add a line to the 1D marginal only
    #g.add_1d(prior_samples, param=name, ls='--', color='gray', lw=1.5, ax=g.subplots[i, i])
    g.add_1d(prior_samples, param=name, ls='solid', color='gray', lw=1.5, ax=g.subplots[i, i],
             zorder=1)    


leg = g.fig.legends[0]

handles, legend_labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
#prior_handle = Line2D([], [], color='gray', ls='--', label='Prior', lw=1.5)
prior_handle = Line2D([], [], color='gray', ls='solid', label='Prior', lw=1.5)

# Re-order.
handles = handles + [prior_handle]
legend_labels = legend_labels + ['Prior']

for L in g.fig.legends:  # remove old legend(s)
    L.remove()

g.fig.legend(handles,
             legend_labels,
             #loc=(0.3, 0.93),
             frameon=g.settings.figure_legend_frame,
             fontsize=10,
             ncols=g.settings.figure_legend_ncol)

    
g.export(opj(imgdir, f'{oname}_corner.png'), dpi=450)
g.export(opj(imgdir, f'{oname}_corner.pdf'), dpi=450)
                                       

