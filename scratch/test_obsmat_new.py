import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import yaml
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from sbi_bmode import sim_utils, script_utils

specdir = '/u/bing/sbi_bmode/data'
imgdir = '/u/bing/sbi_bmode/binh/figures'
os.makedirs(imgdir, exist_ok=True)

with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
data_dict, fixed_params_dict, params_dict, obsmat_dict = script_utils.parse_config(config)

data_dict['freq_strings'] = ['f090']

true_params = dict(
    r_tensor=0.1, A_lens=1.0, A_d_BB=5.0, alpha_d_BB=-0.2, beta_dust=1.59,
)

sim = sim_utils.CMBSimulator(
    specdir, data_dict, fixed_params_dict,
    use_obsmat=obsmat_dict['use_obsmat'],
    obsmat_dir=obsmat_dict.get('obsmat_dir'),
)

sim.use_obsmat = False
out_no_obsmat = sim.draw_data(**true_params, seed=0, return_maps=True)
sky_map = out_no_obsmat["obs_map"]

sim.use_obsmat = True
out_obsmat = sim.draw_data(**true_params, seed=0, return_maps=True)
obs_map = out_obsmat["obs_map"]

split, pol = 0, 0
fidx = sim.freq_strings.index('f090')
fstr = sim.freq_strings[fidx]

fig = plt.figure(figsize=(12, 5))

# Use the same color scale on both panels for a fair comparison.
vmax = np.nanmax(np.abs(sky_map[split, fidx, pol]))

hp.mollview(
    sky_map[split, fidx, pol], sub=(1, 2, 1),
    title=f"{fstr} Q — no ObsMat", unit=r"$\mu K$",
    min=-vmax, max=vmax, fig=fig,
)
hp.mollview(
    obs_map[split, fidx, pol], sub=(1, 2, 2),
    title=f"{fstr} Q — with ObsMat", unit=r"$\mu K$",
    min=-vmax, max=vmax, fig=fig,
)
plt.tight_layout()
fig.savefig(f"{imgdir}/{fstr}_obsmat_compare_fullsky.png", dpi=150, bbox_inches="tight")
plt.close(fig)