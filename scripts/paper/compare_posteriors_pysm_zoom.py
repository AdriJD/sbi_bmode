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
                    'amp_beta_dust' : r'$B_{\mathrm{d}}$',
                    'gamma_beta_dust' : r'$\gamma_{\mathrm{d}}$'}

imgdir = opj(basedir, 'pysm_compared')
os.makedirs(imgdir, exist_ok=True)
oname = 'nilc_pysm_comparison'
dirs2compare = ['sample76t_d1s5', 'sample76t_d1s7', 'sample76t_d4s5',
                'sample76t_d10s5', 'sample76t_d10alts5', 'sample76t_d12s5']
labels = ['d1\_s5', 'd1\_s7', 'd4\_s5',
          'd10\_s5', 'd10x1.6\_s5', 'd12\_s5']
sim_idx = 3

with open(opj(basedir, 'run76', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)

data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)

params_dict.pop('A_d_BB')
params_dict.pop('alpha_d_BB')
params_dict.pop('beta_dust')
params_dict.pop('A_s_BB')
params_dict.pop('alpha_s_BB')
params_dict.pop('beta_sync')
params_dict.pop('rho_ds')
params_dict.pop('amp_beta_sync')
params_dict.pop('gamma_beta_sync')

# Scale r and rho.
params_dict['r_tensor']['prior_params'][0] *= 100
params_dict['r_tensor']['true_value'] *= 100
prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $
param_limits = script_utils.get_param_limits(prior, param_names)
torch.manual_seed(0)
nsamp = 500_000
prior_draw = draw_from_prior(prior, nsamp)

getdist_plots.default_settings.figure_legend_frame = False
getdist_plots.default_settings.norm_1d_density = False
getdist_plots.default_settings.solid_colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 1, len(dirs2compare)+1)]
getdist_plots.default_settings.figure_legend_ncol = 2

getdist_plots.default_settings.fontsize = 8
getdist_plots.default_settings.axes_fontsize = 8
getdist_plots.default_settings.legend_fontsize = 10
getdist_plots.default_settings.axes_labelsize = 12

getdist_plots.default_settings.scaling = False

samples_g = []
samples_g_insert = []

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits, label='prior')

prior_samples.smooth_scale_1D = 0.2


for sidx, subdir in enumerate(dirs2compare):

    samples_arr = np.load(opj(basedir, subdir, f'samples_test.npy'))[sim_idx]
    # Scale r and rho.
    samples_arr = samples_arr[:, [0,1,5,6]]
    samples_arr[:,0] *= 100

    samples = getdist_MCSamples(
        samples=samples_arr, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=r'$\mathtt{' + f'{labels[sidx]}' + '}$')
    
    samples_g.append(samples)

contour_colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 1, len(dirs2compare))]
line_colors = contour_colors
line_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors]

contour_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors]

filled_arr = [True] * len(dirs2compare)
filled_arr[1] = False
filled_arr[4] = False
contour_args[1]['ls'] = 'dashed'
contour_args[1]['lw'] = 1
contour_args[1]['color'] = line_colors[0]
contour_args[4]['ls'] = 'dashed'
contour_args[4]['lw'] = 1
contour_args[4]['color'] = line_colors[3]

line_args[1]['ls'] = 'dashed'
line_args[1]['lw'] = 1
line_args[1]['color'] = line_colors[0]
line_args[4]['ls'] = 'dashed'
line_args[4]['lw'] = 1
line_args[4]['color'] = line_colors[3]

#g = getdist_plots.get_subplot_plotter(width_inch=7.1)
g = getdist_plots.get_subplot_plotter(width_inch=5)
g.triangle_plot(samples_g, filled=filled_arr,
                #markers=param_truths,
                contour_args=contour_args,
                line_args=line_args
                )

r_tensor_aidx = 0
A_lens_aidx = 1

for ax in g.subplots[:,r_tensor_aidx]:
    ax.axvline(param_truths['r_tensor'], color="gray", ls="--", lw=0.5)
for ax in g.subplots[1:,A_lens_aidx]:
    ax.axvline(0.5, color="gray", ls="--", lw=0.5)

g.subplots[A_lens_aidx,r_tensor_aidx].axhline(0.5, color="gray", ls="--", lw=1)
    
for i, name in enumerate(param_names):
    # This will add a line to the 1D marginal only
    g.add_1d(prior_samples, param=name, ls='--', color='gray', lw=0.5, ax=g.subplots[i, i])

leg = g.fig.legends[0]

handles, legend_labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
prior_handle = Line2D([], [], color='gray', ls='--', label='Prior', lw=0.5)

# Re-order.
handles = handles + [prior_handle]
legend_labels = legend_labels + ['Prior']

for L in g.fig.legends:  # remove old legend(s)
    L.remove()

g.fig.legend(handles,
             legend_labels,
             loc=(0.4, 0.8),
             frameon=g.settings.figure_legend_frame,
             fontsize=10,
             ncols=g.settings.figure_legend_ncol)
print(g.fig.legends)
    
g.export(opj(imgdir, f'{oname}_corner_zoom.png'), dpi=450)
