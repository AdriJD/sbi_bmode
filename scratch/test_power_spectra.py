import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import yaml
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from sbi_bmode import sim_utils, script_utils


specdir = '/u/bing/sbi_bmode/data'
imgdir = '/u/bing/sbi_bmode/binh/figures'
os.makedirs(imgdir, exist_ok=True)


# -------------------------------------------------------
# Load config
# -------------------------------------------------------

with open('config_transfer.yaml') as f:
    config = yaml.safe_load(f)


(
    data_dict,
    fixed_params_dict,
    params_dict,
    observation_dict,
    transfer_function_dict,
) = script_utils.parse_config(config)


data_dict['freq_strings'] = ['f090']


true_params = dict(
    r_tensor=0.1,
    A_lens=1.0,
    A_d_BB=5.0,
    alpha_d_BB=-0.2,
    beta_dust=1.59,
)



# -------------------------------------------------------
# Identity observation
# -------------------------------------------------------

print("\n===== Identity simulation =====")

sim_identity = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict={"type": "identity"},
)


out_identity = sim_identity.draw_data(
    **true_params,
    seed=0,
    return_maps=True,
)


sky_map = out_identity["obs_map"]



# -------------------------------------------------------
# ObsMat observation
# -------------------------------------------------------

print("\n===== ObsMat simulation =====")

sim_obsmat = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=observation_dict,
)


out_obsmat = sim_obsmat.draw_data(
    **true_params,
    seed=0,
    return_maps=True,
)


obs_map = out_obsmat["obs_map"]



# -------------------------------------------------------
# Select split and frequency
# -------------------------------------------------------

split = 0
fidx = 0

sky_QU = sky_map[split, fidx]
obs_QU = obs_map[split, fidx]


print("sky QU shape:", sky_QU.shape)
print("obs QU shape:", obs_QU.shape)



# -------------------------------------------------------
# NaMaster pseudo-Cl
# -------------------------------------------------------

print("\n===== NaMaster pseudo-Cl =====")


mask_file = transfer_function_dict["mask_file"]
ell_bin = transfer_function_dict["ell_bin"]
apod_scale = transfer_function_dict["apod_scale"]
apod_type = transfer_function_dict["apod_type"]


cl_sky = sim_utils.estimate_pseudo_cl(
    sky_QU,
    sky_QU,
    ell_bin,
    data_dict["nside"],
    mask_file,
    apod_scale=apod_scale,
    apod_type=apod_type,
)


cl_obs = sim_utils.estimate_pseudo_cl(
    obs_QU,
    obs_QU,
    ell_bin,
    data_dict["nside"],
    mask_file,
    apod_scale=apod_scale,
    apod_type=apod_type,
)


ell = cl_sky["ell"]


print("BB shape:", cl_sky["BB"].shape)



# -------------------------------------------------------
# Plot NaMaster BB spectra
# -------------------------------------------------------

plt.figure(figsize=(8,5))


plt.loglog(
    ell,
    cl_sky["BB"],
    label="Identity",
)


plt.loglog(
    ell,
    cl_obs["BB"],
    label="ObsMat",
)


plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}$")
plt.legend()
plt.grid(True)


plt.savefig(
    f"{imgdir}/BB_identity_vs_obsmat.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()



# -------------------------------------------------------
# Plot C_ell(identity) / C_ell(ObsMat)
# -------------------------------------------------------

ratio = np.divide(
    cl_sky["BB"],
    cl_obs["BB"],
    out=np.zeros_like(cl_sky["BB"]),
    where=cl_obs["BB"] > 0,
)


plt.figure(figsize=(8,5))


plt.plot(
    ell,
    ratio,
)


plt.xlabel(r"$\ell$")
plt.ylabel(
    r"$C_\ell^{BB,\mathrm{identity}}/"
    r"C_\ell^{BB,\mathrm{ObsMat}}$"
)

plt.grid(True)


plt.savefig(
    f"{imgdir}/BB_identity_over_obsmat.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()



# -------------------------------------------------------
# Optional: healpy anafast sanity check
# -------------------------------------------------------

print("\n===== healpy anafast =====")


# healpy needs T,Q,U
T_sky = np.zeros_like(sky_QU[0])
T_obs = np.zeros_like(obs_QU[0])


cl_sky_ana = hp.anafast(
    [T_sky, sky_QU[0], sky_QU[1]],
    pol=True,
    lmax=data_dict["lmax"],
)


cl_obs_ana = hp.anafast(
    [T_obs, obs_QU[0], obs_QU[1]],
    pol=True,
    lmax=data_dict["lmax"],
)


# healpy output:
# TT, EE, BB, TE, EB, TB

cl_sky_BB_ana = cl_sky_ana[2]
cl_obs_BB_ana = cl_obs_ana[2]

ell_ana = np.arange(len(cl_sky_BB_ana))


plt.figure(figsize=(8,5))


plt.loglog(
    ell_ana[2:],
    cl_sky_BB_ana[2:],
    label="anafast Identity",
)


plt.loglog(
    ell_ana[2:],
    cl_obs_BB_ana[2:],
    label="anafast ObsMat",
)


plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{BB}$")
plt.legend()
plt.grid(True)


plt.savefig(
    f"{imgdir}/anafast_BB_identity_vs_obsmat.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()



print("\nDone.")
print(f"Figures saved in {imgdir}")