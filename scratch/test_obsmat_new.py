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

# -------------------------------------------------------
# Identity observation (no ObsMat)
# -------------------------------------------------------

identity_observation = {
    "type": "identity"
}

sim_identity = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=identity_observation,
)

out_identity = sim_identity.draw_data(
    **true_params,
    seed=0,
    return_maps=True,
)

sky_map = out_identity["obs_map"]


# -------------------------------------------------------
# ObsMat observation
# Use f090 ObsMat for all frequencies
# -------------------------------------------------------

obsmat_observation = {
    "type": "obsmat",
    "obsmat_dir": "/u/bing/so-data/mss2",
    "obsmat_freq": "f090",
}


sim_obsmat = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=obsmat_observation,
)


out_obsmat = sim_obsmat.draw_data(
    **true_params,
    seed=0,
    return_maps=True,
)

obs_map = out_obsmat["obs_map"]

split = 0
pol = 0

fidx = sim_identity.freq_strings.index('f090')
fstr = sim_identity.freq_strings[fidx]


fig = plt.figure(figsize=(12, 5))


# Same color scale
vmax = np.nanmax(
    np.abs(
        sky_map[split, fidx, pol]
    )
)


hp.mollview(
    sky_map[split, fidx, pol],
    sub=(1, 2, 1),
    title=f"{fstr} Q — identity",
    unit=r"$\mu K$",
    min=-vmax,
    max=vmax,
    fig=fig,
)


hp.mollview(
    obs_map[split, fidx, pol],
    sub=(1, 2, 2),
    title=f"{fstr} Q — ObsMat ({fstr})",
    unit=r"$\mu K$",
    min=-vmax,
    max=vmax,
    fig=fig,
)


plt.tight_layout()

fig.savefig(
    f"{imgdir}/{fstr}_identity_vs_obsmat.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)