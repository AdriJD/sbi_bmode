import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import yaml
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from sbi_bmode import sim_utils, script_utils


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

specdir = "/u/bing/sbi_bmode/data"

outdir = "/u/bing/sbi_bmode/binh/figures"
os.makedirs(outdir, exist_ok=True)


combined_mask_file = (
    "/u/bing/sbi_bmode/data/"
    "combined_mask_nside128_ring.fits"
)


# -------------------------------------------------------
# Load config
# -------------------------------------------------------

with open("config_transfer.yaml") as f:
    config = yaml.safe_load(f)


(
    data_dict,
    fixed_params_dict,
    params_dict,
    observation_dict,
    transfer_function_dict,
) = script_utils.parse_config(config)


data_dict["freq_strings"] = ["f090"]



# -------------------------------------------------------
# Parameters
# -------------------------------------------------------

true_params = dict(
    r_tensor=0.1,
    A_lens=1.0,
    A_d_BB=5.0,
    alpha_d_BB=-0.2,
    beta_dust=1.59,
)



# -------------------------------------------------------
# Generate sky map
# -------------------------------------------------------

print("\n===== Generate sky map =====")

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


sky_map = out_identity["sky_map"]

print("Sky map shape:", sky_map.shape)



# -------------------------------------------------------
# Apply ObsMat
# -------------------------------------------------------

print("\n===== Apply ObsMat =====")

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


obs_map = sim_obsmat.apply_observation_map(
    sky_map
)


print("Obs map shape:", obs_map.shape)



# -------------------------------------------------------
# Select split and frequency
# -------------------------------------------------------

split = 0
freq = 0


sky_QU = sky_map[split, freq]
obs_QU = obs_map[split, freq]


print("Q/U shape:", obs_QU.shape)



# -------------------------------------------------------
# Build ObsMat mask
# -------------------------------------------------------

print("\n===== Build ObsMat mask =====")


threshold = 1e-12


obs_mask = (
    (np.abs(obs_QU[0]) > threshold)
    &
    (np.abs(obs_QU[1]) > threshold)
)


obs_mask = obs_mask.astype(float)


print(
    f"ObsMat fsky = {obs_mask.mean():.4f}"
)



# -------------------------------------------------------
# Load Galactic + point source mask
# -------------------------------------------------------

print("\n===== Load Galactic + PS mask =====")


gal_ps_mask = hp.read_map(
    combined_mask_file,
    dtype=np.float64,
)


print(
    f"Galactic+PS fsky = {gal_ps_mask.mean():.4f}"
)



# -------------------------------------------------------
# Combine masks
# -------------------------------------------------------

print("\n===== Combine masks =====")


final_mask = obs_mask * (1 - gal_ps_mask)


print(
    f"Final fsky = {final_mask.mean():.4f}"
)



# -------------------------------------------------------
# Save final mask
# -------------------------------------------------------

final_mask_file = (
    f"{specdir}/full_mask_nside{data_dict['nside']}.fits"
)


hp.write_map(
    final_mask_file,
    final_mask,
    overwrite=True,
)


print("Saved mask:")
print(final_mask_file)



# -------------------------------------------------------
# Apply final mask to observed map
# -------------------------------------------------------

obs_Q_masked = obs_QU[0] * final_mask
obs_U_masked = obs_QU[1] * final_mask



# -------------------------------------------------------
# Plot masks
# -------------------------------------------------------

print("\n===== Plot masks =====")


fig = plt.figure(figsize=(16, 8))


hp.mollview(
    obs_mask,
    title=(
        "ObsMat observation mask\n"
        rf"$f_{{sky}}={obs_mask.mean():.3f}$"
    ),
    sub=(2, 2, 1),
    fig=fig,
)


hp.mollview(
    gal_ps_mask,
    title=(
        "Galactic + point source mask\n"
        rf"$f_{{sky}}={gal_ps_mask.mean():.3f}$"
    ),
    sub=(2, 2, 2),
    fig=fig,
)


hp.mollview(
    final_mask,
    title=(
        "Combined mask\n"
        rf"$f_{{sky}}={final_mask.mean():.3f}$"
    ),
    sub=(2, 2, 3),
    fig=fig,
)


vmax = np.max(np.abs(obs_Q_masked))


hp.mollview(
    obs_Q_masked,
    title="Observed Q map × combined mask",
    sub=(2, 2, 4),
    fig=fig,
    min=-vmax,
    max=vmax,
    cmap="RdBu_r",
    unit=r"$\mu K$",
)


plt.savefig(
    f"{outdir}/obsmat_mask_visualization.png",
    dpi=150,
    bbox_inches="tight",
)


plt.close()


print("\nSaved figure:")
print(f"{outdir}/obsmat_mask_visualization.png")