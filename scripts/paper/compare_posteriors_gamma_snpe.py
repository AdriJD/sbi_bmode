import os
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.text as mtext

#font = {'family' : 'serif',
#         'size'   : 10,
#         'serif':  'cmr10'
#         }

# #matplotlib.rc('font', **font)
# matplotlib.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],  # matches LaTeX's default
#     #"font.serif": ["cmr10"],  # matches LaTeX's default    
#     "mathtext.fontset": "cm",  # for math rendered outside full LaTeX
#     "axes.unicode_minus": False, 
# })

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

def hdi_unimodal(samples, cred_mass=0.68):
    '''
    Compute the Highest Density Interval (HDI) from samples only.
    Works for unimodal 1D distributions.

    Parameters
    ----------
    samples : (nsamp) array-like
        1D posterior samples.
    cred_mass : float
        Credible mass, e.g. 0.68, 0.95.

    Returns
    -------
    (low, high) : tuple
        Lower and upper bounds of the HDI.
    '''
    
    x = np.sort(np.asarray(samples))
    
    n = x.size
    if n == 0:
        raise ValueError("No samples")
    
    k = int(np.ceil(cred_mass * n))  # ensure coverage >= cred_mass
    if k < 1 or k > n:
        raise ValueError("cred_mass leads to invalid window length")
    
    # Candidate windows [i, i+k-1].
    widths = x[k-1:] - x[:n-k+1]
    j = np.argmin(widths)
    
    return (x[j], x[j+k-1])

param_label_dict = {'r_tensor' : r'$100r$',
                    'A_lens' : r'$A_{\mathrm{lens}}$',
                    'A_d_BB' : r'$A_{\mathrm{d}}$',
                    'alpha_d_BB' : r'$\alpha_{\mathrm{d}}$',
                    'beta_dust' : r'$\beta_{\mathrm{d}}$',
                    'amp_beta_dust' : r'$B_{\mathrm{d}}$',
                    'gamma_beta_dust' : r'$\gamma_{\mathrm{d}}$',
                    'A_s_BB' : r'$A_{\mathrm{s}}$',
                    'alpha_s_BB' : r'$\alpha_{\mathrm{s}}$',
                    'beta_sync' : r'$\beta_{\mathrm{s}}$',
                    'amp_beta_sync' : r'$B_{\mathrm{s}}$',
                    'gamma_beta_sync' : r'$\gamma_{\mathrm{s}}$',
                    'rho_ds' : r'$100\rho_{\mathrm{ds}}$'}

oname = '74'
subdir = 'run74'
subdir2 = 'run65_optuna/trial_0120'
subdir2_5000 = 'run65_optuna_5000/trial_0106'
subdir2_10000 = 'run65_optuna_10000/trial_0198'
subdir2_25000 = 'run65_optuna_25000/trial_0132'

idir = opj(basedir, subdir)
idir2 = opj(basedir, subdir2)
idir2_5000 = opj(basedir, subdir2_5000)
idir2_10000 = opj(basedir, subdir2_10000)
idir2_25000 = opj(basedir, subdir2_25000)
imgdir = opj(idir, 'img')
os.makedirs(imgdir, exist_ok=True)

labels = ['SNPE']
#iters = [0, 5, 10, 15, 20, 25, 30, 33]
iters = np.arange(3, 12)
niter = len(iters)
num_sims = np.zeros(12, dtype=int)
num_sims[:6] = 864
num_sims[6:] = 3456
num_sims = np.cumsum(num_sims)

with open(opj(basedir, 'run74', 'config.yaml'), 'r') as yfile:
    config = yaml.safe_load(yfile)

data_dict, fixed_params_dict, params_dict = script_utils.parse_config(config)

params_dict['r_tensor']['prior_params'][0] *= 100
params_dict['r_tensor']['true_value'] *= 100
params_dict['rho_ds']['prior_params'][1] *= 100
params_dict['rho_ds']['true_value'] *= 100

prior, param_names = script_utils.get_prior(params_dict)
param_labels = [param_label_dict[p] for p in param_names]
param_truths = script_utils.get_true_params(params_dict)

param_labels_g = [l[1:-1] for l in param_labels] # getdist will put in $
param_limits = script_utils.get_param_limits(prior, param_names)

# Sample from prior.
print(param_names)
print(param_labels)
print(param_truths)
print(param_limits)

torch.manual_seed(0)
nsamp = 500_000
prior_draw = draw_from_prior(prior, nsamp)

getdist_plots.default_settings.figure_legend_frame = False
getdist_plots.default_settings.norm_1d_density = False
getdist_plots.default_settings.solid_colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 1, niter+1)]
getdist_plots.default_settings.figure_legend_ncol = 3
#getdist_plots.default_settings.fontsize = 10
#getdist_plots.default_settings.legend_fontsize = 12
#getdist_plots.default_settings.axes_labelsize = 12

#getdist_plots.default_settings.fontsize = 10
#getdist_plots.default_settings.axes_fontsize = 10
#getdist_plots.default_settings.legend_fontsize = 12
#getdist_plots.default_settings.axes_labelsize = 12
getdist_plots.default_settings.fontsize = 6
getdist_plots.default_settings.axes_fontsize = 6
getdist_plots.default_settings.legend_fontsize = 10
getdist_plots.default_settings.axes_labelsize = 10

getdist_plots.default_settings.scaling = False
#getdist_plots.default_settings.direct_scaling = True

samples_g = []
samples_g_insert = []

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits, label='prior')

prior_samples.smooth_scale_1D = 0.2

for idx, sidx in enumerate(iters):
    print(idx)
    samples_arr = np.load(opj(idir, f'samples_round_{sidx:03d}.npy'))
    samples_arr[:,0] *= 100
    samples_arr[:,-1] *= 100    
    
    print(idx, sidx, f'sigma_r = {np.abs(np.diff(hdi_unimodal(samples_arr[:,0]))) / 2}, mean: {np.mean(samples_arr[:,0])}')
    
    samples = getdist_MCSamples(
        samples=samples_arr, names=param_names, labels=param_labels_g,
        ranges=param_limits,
        label=f'{round(num_sims[sidx],-2)}')        
    
    samples_g.append(samples)

    samples_insert = getdist_MCSamples(
        samples=samples_arr[:,:2], names=param_names[:2], labels=param_labels_g[:2],
        ranges={k : param_limits[k] for k in ['r_tensor', 'A_lens']})
    
    samples_g_insert.append(samples_insert)

samples2_arr = np.load(opj(idir2, f'samples.npy'))
samples2_arr[:,0] *= 100
samples2_arr[:,-1] *= 100

print(f'samples2: sigma_r = {np.abs(np.diff(hdi_unimodal(samples2_arr[:,0]))) / 2}, mean: {np.mean(samples2_arr[:,0])}')

          
samples2 = getdist_MCSamples(
    samples=samples2_arr, names=param_names, labels=param_labels_g,
    ranges=param_limits,
    label=f'NPE ' r'($N_{\mathrm{sim}} = 51840$)')

samples_g.append(samples2)

samples2_5000_arr = np.load(opj(idir2_5000, f'samples.npy'))
samples2_10000_arr = np.load(opj(idir2_10000, f'samples.npy'))
samples2_25000_arr = np.load(opj(idir2_25000, f'samples.npy'))

samples2_5000_arr[:,0] *= 100
samples2_10000_arr[:,0] *= 100
samples2_25000_arr[:,0] *= 100
samples2_5000_arr[:,-1] *= 100
samples2_10000_arr[:,-1] *= 100
samples2_25000_arr[:,-1] *= 100

samples2_5000_insert = getdist_MCSamples(
    samples=samples2_5000_arr[:,:2], names=param_names[:2], labels=param_labels_g[2:],
    ranges={k : param_limits[k] for k in ['r_tensor', 'A_lens']})

samples2_10000_insert = getdist_MCSamples(
    samples=samples2_10000_arr[:,:2], names=param_names[:2], labels=param_labels_g[2:],
    ranges={k : param_limits[k] for k in ['r_tensor', 'A_lens']})

samples2_25000_insert = getdist_MCSamples(
    samples=samples2_25000_arr[:,:2], names=param_names[:2], labels=param_labels_g[2:],
    ranges={k : param_limits[k] for k in ['r_tensor', 'A_lens']})

samples_g_insert.append(samples2_5000_insert)
samples_g_insert.append(samples2_10000_insert)
samples_g_insert.append(samples2_25000_insert)

samples2_insert = getdist_MCSamples(
    samples=samples2_arr[:,:2], names=param_names[:2], labels=param_labels_g[2:],
    ranges={k : param_limits[k] for k in ['r_tensor', 'A_lens']})

samples_g_insert.append(samples2_insert)

contour_colors = [plt.cm.viridis_r(i) for i in np.linspace(0, 1, niter)] + ['firebrick']
line_colors = contour_colors
line_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors[:-1]]
line_args += [{'lw': 0.4, 'ls' : 'dashed', 'color' : 'firebrick'}]

contour_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors[:-1]]
contour_args += [{'lw': 0.4, 'ls' : 'dashed', 'color' : 'firebrick'}]


contour_colors_insert = [plt.cm.viridis_r(i) for i in np.linspace(0, 1, niter)] + ['firebrick']*3
line_colors_insert = contour_colors_insert
line_insert_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors_insert[:-1]]
line_insert_args += [{'lw': 0.4, 'ls' : 'dashed', 'color' : 'firebrick'}] * 4

contour_insert_args = [{'lw': 1., 'ls' : 'solid', 'color' : c} for c in line_colors[:-1]]
contour_insert_args += [{'lw': 1., 'ls' : 'solid', 'color' : 'firebrick'}]
contour_insert_args += [{'lw': 1., 'ls' : 'dotted', 'color' : 'firebrick'}]
contour_insert_args += [{'lw': 1., 'ls' : 'dashdot', 'color' : 'firebrick'}]
contour_insert_args += [{'lw': 2., 'ls' : 'dashed', 'color' : 'firebrick'}]

g = getdist_plots.get_subplot_plotter(width_inch=7.1)
g.triangle_plot(samples_g, filled=[True] * niter + [False],
                markers=param_truths,
                #contour_colors=contour_colors,
                contour_args=contour_args,
                line_args=line_args
                )

for i, name in enumerate(param_names):
    # This will add a line to the 1D marginal only
    #g.add_1d(prior_samples, param=name, ls='--', color='gray', lw=0.5, ax=g.subplots[i, i])
    g.add_1d(prior_samples, param=name, ls='solid', color='gray', lw=0.5, ax=g.subplots[i, i],
             zorder=1)    


inset_ax = g.fig.add_axes([0.643, 0.54, 0.3, 0.3])
g_inset = getdist_plots.get_subplot_plotter(subplot_size=0.3)
#g_inset.fig = g.fig
#g_inset._subplot = inset_ax
#print(g_inset)
#print(inset_ax)
#g_inset.triangle_plot(samples_g_insert, filled=[True] * niter + [False],
#                markers={k : param_truths[k] for k in ['r_tensor', 'A_lens']},
#                contour_args=contour_args,
#                line_args=line_args
#                )
g.plot_2d(samples_g_insert, 'r_tensor', 'A_lens', filled=[True] * niter + [False] * 4, ax=inset_ax,
          #colors=contour_colors, 
          contour_args=contour_insert_args,
          marker_args={k : param_truths[k] for k in ['r_tensor', 'A_lens']},          
          lims=[0.003 * 100, 0.017 * 100, 0.33, 0.55])

inset_ax.axvline(param_truths['r_tensor'], color='grey', linestyle='--', linewidth=1)
inset_ax.axhline(param_truths['A_lens'], color='grey', linestyle='--', linewidth=1)

inset_handles = [Line2D([], [], linestyle='solid', color='firebrick', lw=1),
                 Line2D([], [], linestyle='dotted', color='firebrick', lw=1),
                 Line2D([], [], linestyle='dashdot', color='firebrick', lw=1),
                 Line2D([], [], linestyle='dashed', color='firebrick', lw=2)]
inset_labels = [f'NPE ' r'($N_{\mathrm{sim}} = 5000$)',
                f'NPE ' r'($N_{\mathrm{sim}} = 10000$)',
                f'NPE ' r'($N_{\mathrm{sim}} = 25000$)',
                f'NPE ' r'($N_{\mathrm{sim}} = 51840$)']

inset_ax.legend(inset_handles, inset_labels, frameon=False, fontsize=8,
                loc='lower right')

#for ax in g_inset.fig.axes:
#    for artist in ax.get_children():
#        artist.remove()   # detach from old fig
#        inset_ax.add_artist(artist)

# Get the existing legend (figure-level in most getdist setups)
leg = g.fig.legends[0]

#handles, labels = leg.legendHandles, [t.get_text() for t in leg.get_texts()]
handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
#prior_handle = Line2D([], [], color='gray', ls='--', label='Prior', lw=0.5)
prior_handle = Line2D([], [], color='gray', ls='solid', label='Prior', lw=0.5)

#header_handle = Line2D([], [], linestyle='none')  # invisible line
#header_label = r"$N_{\mathrm{sim}}$"
header_handle = r"SNPE cumulative $N_{\mathrm{sim}}$:"
header_label = r""

# Re-order.
handles = [prior_handle] + handles[-1:] + [header_handle] + handles[:-1]
labels = ['Prior'] + labels[-1:] + [header_label] + labels[:-1]


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title

for L in g.fig.legends:  # remove old legend(s)
    L.remove()

print(f'{g.settings.legend_fontsize=}')
g.fig.legend(handles,
             labels,
             loc=leg._loc if hasattr(leg, '_loc') else 'upper right',
             frameon=g.settings.figure_legend_frame,
             fontsize=8,
             ncols=g.settings.figure_legend_ncol,
             #handler_map=handlers)
             handler_map={str: LegendTitle({'fontsize': 8})})
    
g.export(opj(imgdir, f'{oname}_corner.png'), dpi=450)
g.export(opj(imgdir, f'{oname}_corner.pdf'), dpi=450)
