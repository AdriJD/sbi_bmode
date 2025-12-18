import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size" : 10
})

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
pysmdir = opj(basedir, 'pysm_compared')
imgdir = pysmdir
os.makedirs(imgdir, exist_ok=True)

# loop over fg combinations
fg_combs = ['d1_s5', 'd1_s7', 'd4_s5', 'd10_s5', 'd10alt_s5', 'd12_s5']
fg_names = ['d1\_s5', 'd1\_s7', 'd4\_s5', 'd10\_s5', 'd10x1.6\_s5', 'd12\_s5']
subdir_dict = {'d1_s5' : {'NILC' : 'tarp78t_d1s5', 'HILC' : 'tarp79t_d1s5'},
               'd1_s7' : {'NILC' : 'tarp78t_d1s7', 'HILC' : 'tarp79t_d1s7'},
               'd4_s5' : {'NILC' : 'tarp78t_d4s5', 'HILC' : 'tarp79t_d4s5'},
               'd10_s5' : {'NILC' : 'tarp78t_d10s5', 'HILC' : 'tarp79t_d10s5'},
               'd10alt_s5' : {'NILC' : 'tarp78t_d10alts5', 'HILC' : 'tarp79t_d10alts5'},
               'd12_s5' : {'NILC' : 'tarp78t_d12s5', 'HILC' : 'tarp79t_d12s5'}}

fig = plt.figure(figsize=(3.35, 5), dpi=300)
gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.0, hspace=0.0)
axs = np.empty((3, 2), dtype=object)

for aidx in range(3):
    for ajdx in range(2):
        axs[aidx,ajdx] = fig.add_subplot(gs[aidx,ajdx])
ax_flat = axs.ravel()

ilc_labels = ['Joint HILC', 'Joint NILC']

for idx, fg_comb in enumerate(fg_combs):

    for tidx, ilc_type in enumerate(['HILC', 'NILC']):
        
        tarpdir = opj(basedir, subdir_dict[fg_comb][ilc_type])
    
        alpha = np.load(opj(tarpdir, 'tarp_alpha.npy'))
        ecp = np.load(opj(tarpdir, 'tarp_ecp.npy'))
        ecp_boot = np.load(opj(tarpdir, 'tarp_ecp_boot.npy'))

        ecp_std = np.std(ecp_boot, axis=0)

        ecp_marg = np.load(opj(tarpdir, 'tarp_ecp_marg.npy'))
        ecp_marg_boot = np.load(opj(tarpdir, 'tarp_ecp_marg_boot.npy'))
        ecp_marg_std = np.std(ecp_marg_boot, axis=1)

        ax = ax_flat[idx]
        
        
        # ax.plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.01)

        # ax.plot(alpha, ecp, color='b', label=r'$51840$')
        # ax.fill_between(alpha, ecp-ecp_std, ecp+ecp_std, alpha=0.7, color='C0', edgecolor='face', lw=0)
        # ax.fill_between(alpha, ecp-2*ecp_std, ecp+2*ecp_std, alpha=0.3, color='C0', edgecolor='face', lw=0)

        # ax.text(
        #     0.95, 0.05, 'Joint TARP',         
        #     transform=ax.transAxes,    # position in axes coords (0–1)
        #     fontsize=12, #fontweight="bold",
        #     ha='right', va='bottom', 
        #     multialignment="right"
        # )


        # leg = ax.legend(frameon=False, loc='upper left', ncols=1)
        # handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]

        # header_handle = r"$N_{\mathrm{sim}}$:"
        # header_label = r""

        # # Re-order.
        # handles = [handles[0]] + [header_handle] + handles[1:]
        # labels = [labels[0]] + [header_label] + labels[1:]


        # leg.remove()

        # ax.legend(handles,
        #               labels,
        #               loc='upper left',
        #               frameon=False,
        #               fontsize=12,
        #               ncols=1,
        #               handler_map={str: LegendTitle({'fontsize': 12})})


        # ax.set_ylabel(r'$\mathrm{ECP}$', fontsize=12)
        # ax.tick_params('both', which='both', direction='in', right=True, top=True)
        # ax.grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
        # ax.set_axisbelow(True)

        #ax.plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.2)
        ax.plot(alpha, alpha, color='black', ls='dashed', zorder=2.2)        
        pidx = 0
        #for pidx, param in zip(range(2), [r'$r$', r'$A_{\mathrm{lens}}$']):
        if tidx == 1:
            edgecolor = 'C1'
            lw = 0
            facecolor = 'C1'
            color = 'C1'
            zorder = 2.02
            zorder2 = 1
        else:
            edgecolor = 'face'
            lw = 0
            facecolor = 'C0'
            color = 'b'
            zorder = 2.1
            zorder2 = 2.01

        ax.plot(alpha, ecp_marg[pidx], color=color, label=ilc_labels[tidx], zorder=zorder, lw=1)
        #ax.fill_between(alpha, ecp_marg[pidx]-ecp_marg_std[pidx], ecp_marg[pidx]+ecp_marg_std[pidx],
        #                alpha=0.7, color=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder2)
        ax.fill_between(alpha, ecp_marg[pidx]-2*ecp_marg_std[pidx], ecp_marg[pidx]+2*ecp_marg_std[pidx],
                        alpha=0.3, color=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder2)

        if idx == 0:
            ax.legend(frameon=False, fontsize=10, ncols=2, bbox_to_anchor=(-0.05, 1.3), loc='upper left',
                      handlelength=2, columnspacing=1.4, handletextpad=0.6)
        ax.tick_params('both', which='both', direction='in', right=True, top=True)
        #ax.grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
        ax.set_axisbelow(True)

        #loc = (0.93, 0.05)
        loc = (0.07, 0.93)
        ax.text(
            *loc, r'$\mathtt{' + fg_names[idx] + '}$',         
            transform=ax.transAxes,    # position in axes coords (0–1)
            fontsize=12, #fontweight="bold",
            ha='left', va='top', 
            #bbox=dict(facecolor="white", edgecolor="black", pad=2.5),
            multialignment="right"
        )
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        
for ajdx in range(2):
    axs[-1,ajdx].xaxis.set_ticks([0, 0.5, 1])
    axs[-1,ajdx].xaxis.set_ticklabels(['0', '0.5', '1'])

for aidx in range(2):
    for ajdx in range(2):    
        axs[aidx,ajdx].xaxis.set_ticks([0, 0.5, 1])
        axs[aidx,ajdx].xaxis.set_ticklabels([])
    
for aidx in range(3):
    axs[aidx,0,].yaxis.set_ticks([0, 0.5, 1])
    axs[aidx,0,].yaxis.set_ticklabels(['0', '0.5', '1'])

for aidx in range(3):
    axs[aidx,1,].yaxis.set_ticks([0, 0.5, 1])
    axs[aidx,1,].yaxis.set_ticklabels([])
    
            
fig.supylabel(r'$\mathrm{ECP}$', y=0.50, x=-0.03, fontsize=12)        
fig.supxlabel(r'Credibility level $1-\alpha$', x=0.50, fontsize=12)
        
fig.savefig(opj(imgdir, 'ecp_marg_pysm'), bbox_inches='tight')
fig.savefig(opj(imgdir, 'ecp_marg_pysm.pdf'), bbox_inches='tight')

plt.close(fig)

