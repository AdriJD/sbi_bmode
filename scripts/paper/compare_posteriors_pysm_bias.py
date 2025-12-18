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
    
param_label_dict = {'r_tensor' : r'$r$',
                    'A_lens' : r'$A_{\mathrm{lens}}$'}

imgdir = opj(basedir, 'pysm_compared')
os.makedirs(imgdir, exist_ok=True)

# loop over fg combinations
fg_combs = ['d1_s5', 'd1_s7', 'd4_s5', 'd10_s5', 'd10alt_s5', 'd12_s5']
#fg_combs = ['d10alt_s5']
fg_names = ['d1\_s5', 'd1\_s7', 'd4\_s5', 'd10\_s5', 'd10x1.6\_s5', 'd12\_s5']
subdir_dict = {'d1_s5' : {'NILC' : 'sample76t_d1s5', 'HILC' : 'sample77t_d1s5'},
               'd1_s7' : {'NILC' : 'sample76t_d1s7', 'HILC' : 'sample77t_d1s7'},
               'd4_s5' : {'NILC' : 'sample76t_d4s5', 'HILC' : 'sample77t_d4s5'},
               'd10_s5' : {'NILC' : 'sample76t_d10s5', 'HILC' : 'sample77t_d10s5'},
               'd10alt_s5' : {'NILC' : 'sample76t_d10alts5', 'HILC' : 'sample77t_d10alts5'},
               'd12_s5' : {'NILC' : 'sample76t_d12s5', 'HILC' : 'sample77t_d12s5'}}

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
params_dict.pop('amp_beta_dust')
params_dict.pop('gamma_beta_dust')
params_dict.pop('amp_beta_sync')
params_dict.pop('gamma_beta_sync')

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
getdist_plots.default_settings.figure_legend_ncol = 2

#getdist_plots.default_settings.fontsize = 6
#getdist_plots.default_settings.axes_fontsize = 6
getdist_plots.default_settings.fontsize = 10
getdist_plots.default_settings.axes_fontsize = 10

getdist_plots.default_settings.legend_fontsize = 10
getdist_plots.default_settings.axes_labelsize = 12

getdist_plots.default_settings.scaling = False

prior_samples = getdist_MCSamples(
    samples=prior_draw, names=param_names, labels=param_labels_g,
    ranges=param_limits, label='prior')

prior_samples.smooth_scale_1D = 0.2
cmap = plt.get_cmap("tab10")

#fig, axs = plt.subplots(3, 2, figsize=(3.35, 5), sharex=True, sharey=True)
#axs_flat = axs.flatten()

#g = getdist_plots.get_subplot_plotter(width_inch=3.35/2)
g = getdist_plots.get_subplot_plotter()
#settings = getdist_plots.GetDistPlotSettings(fig_width_inch=3.35)
#g = getdist_plots.GetDistPlotter()
g.settings.fig_width_inch = 3.35
g.settings.figure_legend_frame = False
g.settings.fig_width_inch = 3.35
g.settings.constrained_layout = True

samples_g_out = []
means_out = []
sigma_r_out = []

for fidx, fg_comb in enumerate(fg_combs):

    samples_g = []
    means_per_ctype = []
    sigma_r_per_ctype = []    
    
    for ctype in ['HILC', 'NILC']:    
        #ctype = 'NILC'
        subdir = subdir_dict[fg_comb][ctype]
        samples_arr_full = np.load(opj(basedir, subdir, f'samples_test.npy'))

        samples_arr_full[:,:,0] *= 100
        
        nsim = samples_arr_full.shape[0]
        means = np.mean(samples_arr_full[:,:,:2], axis=1)
        means_per_ctype.append(means)

        # Save sigma_r for table.
        print(samples_arr_full.shape)
        sigma_r = np.zeros(samples_arr_full.shape[0]) # sigma_r for each test.
        for idx in range(samples_arr_full.shape[0]):
            sigma_r[idx] =  np.abs(np.diff(hdi_unimodal(samples_arr_full[idx,:,0]))) / 2.  
        sigma_r_per_ctype.append(sigma_r)
        #sim_idxs_plot = list(range(0, nsim, 50))
        #nsim2plot = len(sim_idxs_plot)

        #for sim_idx in sim_idxs_plot:

        #    samples_arr = samples_arr_full[sim_idx,:,:2]
        #    samples = getdist_MCSamples(
        #        samples=samples_arr, names=param_names, labels=param_labels_g,
        #        ranges=param_limits)

        #samples_g.append(samples)


        samples = getdist_MCSamples(
            samples=means, names=param_names, labels=param_labels_g,
            ranges=param_limits,
            label=ctype)
        
        samples_g.append(samples)

    samples_g_out.append(samples_g)
    means_out.append(means_per_ctype)
    sigma_r_out.append(sigma_r_per_ctype)
    #contour_colors = [cmap(0)] * nsim2plot
    #contour_colors = [cmap(0), cmap(1)] 
    #line_colors = contour_colors
    #line_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors]
    #contour_args = [{'lw': 0.5, 'ls' : 'solid', 'color' : c} for c in line_colors]

    #g = getdist_plots.get_subplot_plotter(width_inch=7.1)
    #g.settings.figure_legend_frame = False
    #g.plot_2d(samples_g, 'r_tensor', 'A_lens', filled=True, ax=axs_flat[fidx],
    #getdist_plots.plot_2d(samples_g, 'r_tensor', 'A_lens', filled=True, ax=axs_flat[fidx],              
    #          #colors=contour_colors, 
    #          contour_args=contour_args,
    #          alpha=0.5,
    #          marker_args={k : param_truths[k] for k in ['r_tensor', 'A_lens']})

sigma_r_out = np.asarray(sigma_r_out)
    
fig, axs = plt.subplots(dpi=300, nrows=2, figsize=(5, 10))
for fidx, fg_comb in enumerate(fg_combs):
    for cidx, ctype in enumerate(['HILC', 'NILC']):

        sigma_r_mean = np.mean(sigma_r_out[fidx,cidx])
        mean_r_mean = np.mean(means_out[fidx][cidx][:,0])
        axs[cidx].hist(sigma_r_out[fidx,cidx], histtype='step', density=True,
                       label=f'{fg_comb}: m_r = {mean_r_mean:.3f}, s_r = {sigma_r_mean:.3f}')

for cidx, ctype in enumerate(['HILC', 'NILC']):
    axs[cidx].set_title(ctype)
    axs[cidx].legend()

axs[0].set_xlabel('sigma r')
axs[1].set_xlabel('sigma r')
plt.savefig(opj(imgdir, 'hist_sigma_r_mean'))
plt.close(fig)

#exit()
    
    
plot_roots = [
    [samples_g_out[0], samples_g_out[2], samples_g_out[4]],
    [samples_g_out[1], samples_g_out[3], samples_g_out[5]],
    ]
g.rectangle_plot(['r_tensor', 'r_tensor'], ['A_lens', 'A_lens', 'A_lens'],
                 plot_roots=plot_roots, filled=True, legend_labels=['Joint HILC', 'Joint NILC'],
                 alphas=[0.85, 0.7],
                 line_args=[{'lw': 1.5, 'ls' : 'solid', 'color' : c} for c in ['C0', 'C1']],
                 contour_args=[{'lw': 1.5, 'ls' : 'solid', 'color' : c} for c in ['C0', 'C1']])
                 #plot_texts=[['d1_s5', None, None], [None, None, None]])


# Plot posterior means as dots
#for midx, means in enumerate(means_per_ctype):
#    g.subplots[0,0].scatter(means[:,0], means[:,1], s=1, color=cmap(midx), zorder=0)
for aidx, ax in enumerate(g.subplots.ravel()):
    ax.scatter(means_out[aidx][0][:,0], means_out[aidx][0][:,1], s=1, c='C0', zorder=0, alpha=1)
    ax.scatter(means_out[aidx][1][:,0], means_out[aidx][1][:,1], s=1, c='C1', zorder=0, alpha=1)
    
for aidx, ax in enumerate(g.subplots.ravel()):
    #g.subplots[0,0].text(
    ax.text(    
        0.95, 0.05, r'$\mathtt{' + fg_names[aidx] + '}$',         
        transform=ax.transAxes, fontsize=12, ha='right', va='bottom', 
        #bbox=dict(facecolor="white", edgecolor="black", pad=3.5),
        multialignment="right")

# Plot contour for means

for xidx in range(2):
    for yidx in range(3):
        g.subplots[yidx,xidx].axvline(0.01 * 100, color="gray", ls="--", lw=1)    
        g.subplots[yidx,xidx].axhline(0.5, color="gray", ls="--", lw=1)

for ax in g.fig.axes:
    ax.xaxis.label.set_color('white')

g.subplots[0,0].yaxis.label.set_color('white')
g.subplots[1,0].yaxis.label.set_color('white')
g.subplots[2,0].yaxis.label.set_color('white')

g.fig.supxlabel(r'$100r$', x=0.58, y=0.006, fontsize=12)
g.fig.supylabel(r'$A_{\mathrm{lens}}$', y=0.55, x=-0.002, fontsize=12)

g.export(opj(imgdir, f'pysm_cosmo_bias.png'), dpi=450)
g.export(opj(imgdir, f'pysm_cosmo_bias.pdf'), dpi=450)

