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

opj = os.path.join

basedir = '/u/adriaand/project/so/20240521_sbi_bmode'
tarpdir = opj(basedir, 'tarp65')
tarpdir_emoped = opj(basedir, 'tarp65_emoped')
tarpdir_5000 = opj(basedir, 'tarp65_5000')
tarpdir_10000 = opj(basedir, 'tarp65_10000')
tarpdir_25000 = opj(basedir, 'tarp65_25000')
imgdir = opj(tarpdir, 'img')
os.makedirs(imgdir, exist_ok=True)

alpha = np.load(opj(tarpdir, 'tarp_alpha.npy'))
ecp = np.load(opj(tarpdir, 'tarp_ecp.npy'))
ecp_boot = np.load(opj(tarpdir, 'tarp_ecp_boot.npy'))
ecp_emoped = np.load(opj(tarpdir_emoped, 'tarp_ecp_emoped.npy'))
ecp_emoped_boot = np.load(opj(tarpdir_emoped, 'tarp_ecp_emoped_boot.npy'))

ecp_std = np.std(ecp_boot, axis=0)
ecp_emoped_std = np.std(ecp_emoped_boot, axis=0)

ecp_5000 = np.load(opj(tarpdir_5000, 'tarp_ecp.npy'))
ecp_boot_5000 = np.load(opj(tarpdir_5000, 'tarp_ecp_boot.npy'))
ecp_std_5000 = np.std(ecp_boot_5000, axis=0)

ecp_10000 = np.load(opj(tarpdir_10000, 'tarp_ecp.npy'))
ecp_boot_10000 = np.load(opj(tarpdir_10000, 'tarp_ecp_boot.npy'))
ecp_std_10000 = np.std(ecp_boot_10000, axis=0)

ecp_25000 = np.load(opj(tarpdir_25000, 'tarp_ecp.npy'))
ecp_boot_25000 = np.load(opj(tarpdir_25000, 'tarp_ecp_boot.npy'))
ecp_std_25000 = np.std(ecp_boot_25000, axis=0)

ecp_marg = np.load(opj(tarpdir, 'tarp_ecp_marg.npy'))
ecp_marg_boot = np.load(opj(tarpdir, 'tarp_ecp_marg_boot.npy'))
ecp_marg_std = np.std(ecp_marg_boot, axis=1)


fig2 = plt.figure(figsize=(3.35, 5), dpi=300)
gs = gridspec.GridSpec(3, 2, figure=fig2, wspace=-0.025, hspace=0.0,
                       #height_ratios=[1, 1, 1.15])
                       height_ratios=[1, 1, 1])
axs = np.empty((3, 2), dtype=object)

for aidx in range(3):
    for ajdx in range(2):
        axs[aidx,ajdx] = fig2.add_subplot(gs[aidx,ajdx])
        axs[aidx,ajdx].set_box_aspect(1)
ax_flat = axs.ravel()

for ax in axs[2]:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - 0.02, pos.width, pos.height])

for ajdx in range(2):
    axs[-1,ajdx].xaxis.set_ticks([0, 0.5, 1])
    axs[-1,ajdx].xaxis.set_ticklabels(['0', '0.5', '1'])

for aidx in range(2):
    for ajdx in range(2):    
        axs[aidx,ajdx].xaxis.set_ticks([0, 0.5, 1])
        axs[aidx,ajdx].xaxis.set_ticklabels([])


for aidx in range(3):
    for ajdx in range(2):    
        
        axs[aidx,ajdx].set_xlim(-0.1, 1.1)
        axs[aidx,ajdx].set_ylim(-0.1, 1.1)
        axs[aidx,ajdx].tick_params('both', which='both', direction='in', right=True, top=True)

axs[0,0].plot(alpha, ecp_5000, color='C1', lw=1, ls='solid')#, label=r'$N_{\mathrm{sim}} = 5000$')
axs[0,1].plot(alpha, ecp_10000, color='C1', lw=1, ls='solid')#, label=r'$N_{\mathrm{sim}} = 10000$')
axs[1,0].plot(alpha, ecp_25000, color='C1', lw=1, ls='solid')#, label=r'$N_{\mathrm{sim}} = 25000$')

axs[0,0].fill_between(alpha, ecp_5000-2*ecp_std_5000, ecp_5000+2*ecp_std_5000,
                      alpha=0.3, color='C1', edgecolor='face', lw=0)

axs[0,1].fill_between(alpha, ecp_10000-2*ecp_std_10000, ecp_10000+2*ecp_std_10000,
                      alpha=0.3, color='C1', edgecolor='face', lw=0)

#axs[1,0].fill_between(alpha, ecp_25000-ecp_std_25000, ecp_25000+ecp_std_25000, alpha=0.7, color='C1', edgecolor='face', lw=0)
axs[1,0].fill_between(alpha, ecp_25000-2*ecp_std_25000, ecp_25000+2*ecp_std_25000,
                      alpha=0.3, color='C1', edgecolor='face', lw=0)


axs[1,1].plot(alpha, ecp, color='b', lw=1)#, label=r'$N_{\mathrm{sim}} = 51840$')

#axs[1,1].fill_between(alpha, ecp-ecp_std, ecp+ecp_std, alpha=0.7, color='C0', edgecolor='face', lw=0)
axs[1,1].fill_between(alpha, ecp-2*ecp_std, ecp+2*ecp_std, alpha=0.3, color='C0', edgecolor='face', lw=0)

nsims = [5000, 10000, 25000, 51840]
idx = 0
for aidx in range(2):
    for ajdx in range(2):
        axs[aidx,ajdx].text(
            0.065, 0.8, r'$N_{\mathrm{sim}} = ' + str(nsims[idx]) + '$', 
            transform=axs[aidx,ajdx].transAxes,    # position in axes coords (0–1)
            fontsize=12, #fontweight="bold",
            ha='left', va='bottom', 
            multialignment="left"
        )
        idx += 1 

for pidx, param in zip(range(2), [r'$r$', r'$A_{\mathrm{lens}}$']):
    if pidx == 1:
        edgecolor = 'C2'
        lw = 0
        facecolor = 'C2'
        color = 'C2'
        #zorder = 2.02
        #zorder2 = 1
    else:
        edgecolor = 'C3'
        lw = 0
        facecolor = 'C3'
        color = 'C3'
        #zorder = 2.1
        #zorder2 = 2.01
        
    axs[2,pidx].plot(alpha, ecp_marg[pidx], color=color, label=param, lw=1)#, zorder=zorder)
    #axs[2,pidx].fill_between(alpha, ecp_marg[pidx]-ecp_marg_std[pidx], ecp_marg[pidx]+ecp_marg_std[pidx],
    #                         alpha=0.7, color=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder2)
    axs[2,pidx].fill_between(alpha, ecp_marg[pidx]-2*ecp_marg_std[pidx], ecp_marg[pidx]+2*ecp_marg_std[pidx],
                         alpha=0.3, color=facecolor, edgecolor=edgecolor, lw=lw)#, zorder=zorder2)


axs[1,1].text(
    0.95, 0.05, 'Joint TARP',         
    transform=axs[1,1].transAxes,    # position in axes coords (0–1)
    fontsize=12, #fontweight="bold",
    ha='right', va='bottom', 
    #bbox=dict(facecolor="white", edgecolor="black", pad=3.5),
    multialignment="right"
)

    
for aidx in range(3):
    axs[aidx,0,].yaxis.set_ticks([0, 0.5, 1])
    axs[aidx,0,].yaxis.set_ticklabels(['0', '0.5', '1'])

for aidx in range(3):
    axs[aidx,1,].yaxis.set_ticks([0, 0.5, 1])
    axs[aidx,1,].yaxis.set_ticklabels([])

for aidx in range(3):
    for ajdx in range(2):    

        axs[aidx,ajdx].plot(alpha, alpha, color='black', ls='dashed',
                            label='Ideal' if (aidx, ajdx) == (0, 0) else None, zorder=2.01)
        
axs[0,0].legend(frameon=False, loc=(0.062, 0.6), handletextpad=0.5, handlelength=1.5)
for pidx in range(2):
    axs[2,pidx].legend(frameon=False, loc='upper left', fontsize=11)

            
fig2.supylabel(r'$\mathrm{ECP}$', y=0.50, x=-0.04, fontsize=12)        
fig2.supxlabel(r'Credibility level $1-\alpha$', x=0.50, y=-0.02, fontsize=12)

fig2.savefig(opj(imgdir, 'ecp_comb'), bbox_inches='tight')
fig2.savefig(opj(imgdir, 'ecp_comb.pdf'), bbox_inches='tight')
plt.close(fig2)


# fig, axs = plt.subplots(nrows=2, dpi=300, constrained_layout=True, figsize=(3.55, 7.1),
#                         sharex=True, sharey=True)

# axs[0].plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.01)

# axs[0].plot(alpha, ecp_5000, color='C1', lw=1, ls='solid', label=r'$5000$')
# axs[0].plot(alpha, ecp_10000, color='C1', lw=1, ls='dashed', label=r'$10000$')
# axs[0].plot(alpha, ecp_25000, color='C1', lw=1, ls='dotted', label=r'$25000$')

# axs[0].plot(alpha, ecp, color='b', label=r'$51840$')
# axs[0].fill_between(alpha, ecp-ecp_std, ecp+ecp_std, alpha=0.7, color='C0', edgecolor='face', lw=0)
# axs[0].fill_between(alpha, ecp-2*ecp_std, ecp+2*ecp_std, alpha=0.3, color='C0', edgecolor='face', lw=0)

# axs[0].text(
#     0.95, 0.05, 'Joint TARP',         
#     transform=axs[0].transAxes,    # position in axes coords (0–1)
#     fontsize=12, #fontweight="bold",
#     ha='right', va='bottom', 
#     #bbox=dict(facecolor="white", edgecolor="black", pad=3.5),
#     multialignment="right"
# )

# #axs[0].fill_between(alpha, ecp_5000-ecp_std_5000, ecp_5000+ecp_std_5000, alpha=0.7, color='C3', edgecolor='face', lw=0)
# #axs[0].fill_between(alpha, ecp_5000-2*ecp_std_5000, ecp_5000+2*ecp_std_5000, alpha=0.3, color='C3', edgecolor='face', lw=0)


# leg = axs[0].legend(frameon=False, loc='upper left', ncols=1)
# handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
# #prior_handle = Line2D([], [], color='gray', ls='--', label='Prior', lw=0.5)

# header_handle = r"$N_{\mathrm{sim}}$:"
# header_label = r""

# # Re-order.
# handles = [handles[0]] + [header_handle] + handles[1:]
# labels = [labels[0]] + [header_label] + labels[1:]

# class LegendTitle(object):
#     def __init__(self, text_props=None):
#         self.text_props = text_props or {}
#         super(LegendTitle, self).__init__()

#     def legend_artist(self, legend, orig_handle, fontsize, handlebox):
#         x0, y0 = handlebox.xdescent, handlebox.ydescent
#         title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
#         handlebox.add_artist(title)
#         return title

# leg.remove()

# axs[0].legend(handles,
#               labels,
#               loc='upper left',
#               frameon=False,
#               fontsize=12,
#               ncols=1,
#               handler_map={str: LegendTitle({'fontsize': 12})})


# axs[0].set_ylabel(r'$\mathrm{ECP}$', fontsize=12)
# axs[0].tick_params('both', which='both', direction='in', right=True, top=True)
# axs[0].grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
# axs[0].set_axisbelow(True)

# axs[1].plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.2)

# for pidx, param in zip(range(2), [r'$r$', r'$A_{\mathrm{lens}}$']):
#     if pidx == 1:
#         edgecolor = 'C1'
#         lw = 1
#         facecolor = 'none'
#         color = 'C1'
#         zorder = 2.02
#         zorder2 = 1
#     else:
#         edgecolor = 'face'
#         lw = 0
#         facecolor = 'C0'
#         color = 'b'
#         zorder = 2.1
#         zorder2 = 2.01
        
#     axs[1].plot(alpha, ecp_marg[pidx], color=color, label=param, zorder=zorder)
#     axs[1].fill_between(alpha, ecp_marg[pidx]-ecp_marg_std[pidx], ecp_marg[pidx]+ecp_marg_std[pidx],
#                         alpha=0.7, color=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder2)
#     axs[1].fill_between(alpha, ecp_marg[pidx]-2*ecp_marg_std[pidx], ecp_marg[pidx]+2*ecp_marg_std[pidx],
#                         alpha=0.3, color=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder2)



# axs[1].legend(frameon=False, loc='upper left', fontsize=12)
# axs[1].set_xlabel(r'Credibility level $1-\alpha$', fontsize=12)
# axs[1].set_ylabel(r'$\mathrm{ECP}$', fontsize=12)
# axs[1].tick_params('both', which='both', direction='in', right=True, top=True)
# axs[1].grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
# axs[1].set_axisbelow(True)
# fig.savefig(opj(imgdir, 'ecp_comb'))
# plt.close(fig)

#fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(3.55, 3.55),
#                       sharex=True, sharey=True)
fig, axs = plt.subplots(nrows=2, dpi=300, constrained_layout=True, figsize=(3.55, 7.1),
                        sharex=True, sharey=True)

axs[0].plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.01)

for idx, (pidx, param) in enumerate(zip([2, 3, 4, 5, 6, 12],
                       [r'$A_{\mathrm{d}}$', r'$\alpha_{\mathrm{d}}$', r'$\beta_{\mathrm{d}}$',
                        r'$B_{\mathrm{d}}$', r'$\gamma_{\mathrm{d}}$', r'$\rho_{\mathrm{ds}}$'])):
    axs[0].plot(alpha, ecp_marg[pidx], color=f'C{idx}', label=param)
    # axs[0].fill_between(alpha, ecp_marg[pidx]-ecp_marg_std[pidx], ecp_marg[pidx]+ecp_marg_std[pidx],
    #                 alpha=0.7, color=f'C{pidx}', edgecolor='face', lw=0)
    # axs[0].fill_between(alpha, ecp_marg[pidx]-2*ecp_marg_std[pidx], ecp_marg[pidx]+2*ecp_marg_std[pidx],
    #                 alpha=0.3, color=f'C{pidx}', edgecolor='face', lw=0)



axs[0].legend(frameon=False, loc='upper left', ncols=2, fontsize=12)
#axs[0].set_xlabel(r'Credibility level $1-\alpha$', fontsize=12)
axs[0].set_ylabel(r'$\mathrm{ECP}$', fontsize=12)
axs[0].tick_params('both', which='both', direction='in', right=True, top=True)
axs[0].grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
axs[0].set_axisbelow(True)
fig.savefig(opj(imgdir, 'ecp_marg_dust'))
plt.close(fig)

#fig, ax = plt.subplots(dpi=300, constrained_layout=True, figsize=(3.55, 3.55),
#                       sharex=True, sharey=True)

axs[1].plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.01)


for idx, (pidx, param) in enumerate(zip([7, 8, 9, 10, 11, 12],
                       [r'$A_{\mathrm{s}}$', r'$\alpha_{\mathrm{s}}$', r'$\beta_{\mathrm{s}}$',
                        r'$B_{\mathrm{s}}$', r'$\gamma_{\mathrm{s}}$', r'$\rho_{\mathrm{ds}}$'])):

    axs[1].plot(alpha, ecp_marg[pidx], color=f'C{idx}', label=param)
    # axs[1].fill_between(alpha, ecp_marg[pidx]-ecp_marg_std[pidx], ecp_marg[pidx]+ecp_marg_std[pidx],
    #                 alpha=0.7, color=f'C{pidx}', edgecolor='face', lw=0)
    # axs[1].fill_between(alpha, ecp_marg[pidx]-2*ecp_marg_std[pidx], ecp_marg[pidx]+2*ecp_marg_std[pidx],
    #                 alpha=0.3, color=f'C{pidx}', edgecolor='face', lw=0)



axs[1].legend(frameon=False, loc='upper left', ncols=2, fontsize=12)
axs[1].set_xlabel(r'Credibility level $1-\alpha$', fontsize=12)
axs[1].set_ylabel(r'$\mathrm{ECP}$', fontsize=12)
axs[1].tick_params('both', which='both', direction='in', right=True, top=True)
axs[1].grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
axs[1].set_axisbelow(True)
fig.savefig(opj(imgdir, 'ecp_marg_fg'))
fig.savefig(opj(imgdir, 'ecp_marg_fg.pdf'))
plt.close(fig)

# Comparison between two reference distributions.

fig = plt.figure(figsize=(3.35, 2.25), dpi=300)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.0, hspace=0.0)
axs = np.empty((2), dtype=object)

for ajdx in range(2):
    axs[ajdx] = fig.add_subplot(gs[ajdx])
    axs[ajdx].set_box_aspect(1)
ax_flat = axs.ravel()

for ajdx in range(2):
    axs[ajdx].xaxis.set_ticks([0, 0.5, 1])
    axs[ajdx].xaxis.set_ticklabels(['0', '0.5', '1'])
    
    axs[ajdx].set_xlim(-0.1, 1.1)
    axs[ajdx].set_ylim(-0.1, 1.1)
    axs[ajdx].tick_params('both', which='both', direction='in', right=True, top=True)
    
axs[0].yaxis.set_ticks([0, 0.5, 1])
axs[0].yaxis.set_ticklabels(['0', '0.5', '1'])

axs[1].yaxis.set_ticks([0, 0.5, 1])
axs[1].yaxis.set_ticklabels([])
    
axs[0].plot(alpha, ecp, color='b', lw=1, ls='solid', label='Unif. ref.')
axs[1].plot(alpha, ecp_emoped, color='C1', lw=1, ls='solid', label='Data ref.')

axs[0].fill_between(alpha, ecp-2*ecp_std, ecp+2*ecp_std,
                      alpha=0.3, color='C0', edgecolor='face', lw=0)

axs[1].fill_between(alpha, ecp_emoped-2*ecp_emoped_std, ecp_emoped+2*ecp_emoped_std,
                      alpha=0.3, color='C1', edgecolor='face', lw=0)

axs[1].text(
    0.95, 0.05, 'Joint TARP',         
    transform=axs[1].transAxes,    # position in axes coords (0–1)
    fontsize=12, #fontweight="bold",
    ha='right', va='bottom', 
    multialignment="right"
)

for ajdx in range(2):    
    axs[ajdx].plot(alpha, alpha, color='black', ls='dashed',
                   label='Ideal' if ajdx == 0 else None, zorder=2.01)
        
#axs[0].legend(frameon=False, loc=(0.062, 0.6), handletextpad=0.5, handlelength=1.5)
for pidx in range(2):
    axs[pidx].legend(frameon=False, loc='upper left', handletextpad=0.5, handlelength=1.5)

            
fig.supylabel(r'$\mathrm{ECP}$', y=0.50, x=-0.04, fontsize=12)        
fig.supxlabel(r'Credibility level $1-\alpha$', x=0.50, y=-0.02, fontsize=12)

fig.savefig(opj(imgdir, 'ecp_comb_emoped'), bbox_inches='tight')
fig.savefig(opj(imgdir, 'ecp_comb_emoped.pdf'), bbox_inches='tight')
plt.close(fig)



exit()

fig, ax = plt.subplots(nrows=1, dpi=300, constrained_layout=True, figsize=(3.55, 3.55),
                        sharex=True, sharey=True)


ax.plot(alpha, alpha, color='black', ls='dashed', label='Ideal', zorder=2.2)

ax.plot(alpha, ecp, color='b', label=r'Uniform ref.', zorder=2.1)
ax.fill_between(alpha, ecp-ecp_std, ecp+ecp_std, alpha=0.7, color='C0', edgecolor='face', lw=0, zorder=2.01)
ax.fill_between(alpha, ecp-2*ecp_std, ecp+2*ecp_std, alpha=0.3, color='C0', edgecolor='face', lw=0, zorder=2.01)

ax.plot(alpha, ecp_emoped, color='C1', label=r'Data-dep. ref.', zorder=2.02)
ax.fill_between(alpha, ecp_emoped-ecp_emoped_std, ecp_emoped+ecp_emoped_std, alpha=0.7, color='none', edgecolor='C1', lw=1)
ax.fill_between(alpha, ecp_emoped-2*ecp_emoped_std, ecp_emoped+2*ecp_emoped_std, alpha=0.3, color='none', edgecolor='C1', lw=1)

ax.text(
    0.95, 0.05, 'Joint TARP',         
    transform=ax.transAxes,    # position in axes coords (0–1)
    fontsize=12, #fontweight="bold",
    ha='right', va='bottom', 
    multialignment="right"
)

ax.legend(frameon=False, loc='upper left', fontsize=12)
ax.set_xlabel(r'Credibility level $1-\alpha$', fontsize=12)
ax.set_ylabel(r'$\mathrm{ECP}$', fontsize=12)
ax.tick_params('both', which='both', direction='in', right=True, top=True)
ax.grid(color='black', linestyle='dotted', linewidth=0.5, zorder=0, which='both')
ax.set_axisbelow(True)
fig.savefig(opj(imgdir, 'ecp_comb_emoped'))
plt.close(fig)

