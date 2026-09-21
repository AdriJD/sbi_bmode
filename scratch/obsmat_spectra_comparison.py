import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import yaml

from sbi_bmode import so_utils, sim_utils, script_utils
from sbi_bmode import spectra_utils


# ============================================================
# Configuration
# ============================================================

with open("config_transfer.yaml") as f:
    config = yaml.safe_load(f)

(
    data_dict,
    fixed_params_dict,
    params_dict,
    observation_dict,
    transfer_dict,
) = script_utils.parse_config(config)

true_params = script_utils.get_true_params(params_dict)

specdir = "/u/bing/sbi_bmode/data"

figdir = "/u/bing/sbi_bmode/binh/figures"
os.makedirs(figdir, exist_ok=True)

maskfile = (
    "/u/bing/sbi_bmode/data/"
    "obsmat_planck70_apod_C2_3deg.fits"
)


# ============================================================
# Construct simulators
# ============================================================

sim_obsmat = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=observation_dict,
    mask_file=maskfile,
)

sim_no_obsmat = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict={"type": "identity"},
    mask_file=maskfile,
)


# ============================================================
# Draw SAME realization with and without ObsMat
# ============================================================

out_obsmat = sim_obsmat.draw_data(
    **true_params,
    seed=0,
    return_maps=True,
)

out_no_obsmat = sim_no_obsmat.draw_data(
    **true_params,
    seed=0,
    return_maps=True,
)


# ============================================================
# Check shapes
#
# Expected:
#
# (nsplit, nfreq, npol, npix)
# ============================================================

print("sky_map with ObsMat :", out_obsmat["sky_map"].shape)
print("obs_map with ObsMat :", out_obsmat["obs_map"].shape)
print("sky_map no ObsMat   :", out_no_obsmat["sky_map"].shape)


# ============================================================
# Frequency
# ============================================================

freq_idx = 0  # f090

print(
    "Frequency:",
    sim_obsmat.freq_strings[freq_idx],
    sim_obsmat.freqs[freq_idx] / 1e9,
    "GHz",
)


# ============================================================
# Example maps
# ============================================================

hp.mollview(
    out_obsmat["sky_map"][0, freq_idx, 0],
    title="Simulated sky map (Q) @ f090",
    unit=r"$\mu$K",
)

hp.mollview(
    out_obsmat["sky_map"][0, freq_idx, 1],
    title="Simulated sky map (U) @ f090",
    unit=r"$\mu$K",
)

obs_q = out_obsmat["obs_map"][0, freq_idx, 0]

hp.mollview(
    obs_q,
    min=-np.nanmax(np.abs(obs_q)),
    max=np.nanmax(np.abs(obs_q)),
    cmap="coolwarm",
    unit=r"$\mu$K",
    norm="hist",
    title="Observed sky map (Q) @ f090",
)


# ============================================================
# Pseudo-Cl estimator
# ============================================================

def estimate_pseudo_cl(map_a, map_b, mask_apod, bins):
    """
    Estimate pseudo-Cl spectra between two Q/U maps
    using NaMaster.

    Parameters
    ----------
    map_a : (2, npix)
        Q/U map from split A.

    map_b : (2, npix)
        Q/U map from split B.

    mask_apod : (npix,)
        Apodized mask.

    bins : pymaster.NmtBin
        NaMaster binning object.

    Returns
    -------
    dict
        ell, EE, EB, BE, BB
    """

    f_a = nmt.NmtField(
        mask_apod,
        [map_a[0], map_a[1]],
    )

    f_b = nmt.NmtField(
        mask_apod,
        [map_b[0], map_b[1]],
    )

    cl_EE, cl_EB, cl_BE, cl_BB = nmt.compute_full_master(
        f_a,
        f_b,
        bins,
    )

    return {
        "ell": bins.get_effective_ells(),
        "EE": cl_EE,
        "EB": cl_EB,
        "BE": cl_BE,
        "BB": cl_BB,
    }


# ============================================================
# Load mask
# ============================================================

mask = hp.read_map(
    maskfile,
    verbose=False,
)

nside_mask = hp.npix2nside(mask.size)

print("Mask nside:", nside_mask)
print("Mask npix :", mask.size)


# The filename indicates that this is already an apodized
# C2 3-degree mask, so do NOT apodize it again.
mask_apod = mask


# ============================================================
# NaMaster binning
# ============================================================

lmax = sim_obsmat.lmax
delta_ell = 15

bins = nmt.NmtBin.from_nside_linear(
    nside_mask,
    delta_ell,
    is_Dell=False,
)


# ============================================================
# Extract split maps
#
# Shape:
#
# (nsplit, nfreq, npol, npix)
#
# Therefore:
#
# [split, frequency, polarization, pixel]
# ============================================================

sky0 = out_no_obsmat["sky_map"][
    0, freq_idx, :, :
]

sky1 = out_no_obsmat["sky_map"][
    1, freq_idx, :, :
]

obs0 = out_obsmat["obs_map"][
    0, freq_idx, :, :
]

obs1 = out_obsmat["obs_map"][
    1, freq_idx, :, :
]


print()
print("Selected maps:")
print("sky split 0:", sky0.shape)
print("sky split 1:", sky1.shape)
print("obs split 0:", obs0.shape)
print("obs split 1:", obs1.shape)


# ============================================================
# Cross-spectrum:
#
# NO OBSMAT
#
# split 0 x split 1
# ============================================================

print()
print("Computing sky cross-spectrum...")

cl_sky = estimate_pseudo_cl(
    sky0,
    sky1,
    mask_apod,
    bins,
)


# ============================================================
# Cross-spectrum:
#
# WITH OBSMAT
#
# split 0 x split 1
# ============================================================

print("Computing ObsMat cross-spectrum...")

cl_obs = estimate_pseudo_cl(
    obs0,
    obs1,
    mask_apod,
    bins,
)


# ============================================================
# Input theoretical spectra
# ============================================================

# ------------------------------------------------------------
# CMB BB
# ------------------------------------------------------------

cmb_cov = spectra_utils.get_combined_cmb_spectrum(
    true_params["r_tensor"],
    true_params["A_lens"],
    sim_obsmat.cov_scalar_ell,
    sim_obsmat.cov_tensor_ell,
)

cmb_bb = np.asarray(
    cmb_cov[1, 1]
)


# ------------------------------------------------------------
# Dust BB
# ------------------------------------------------------------

dust_cov = spectra_utils.get_dust_spectra(
    true_params["A_d_BB"],
    true_params["alpha_d_BB"],
    sim_obsmat.lmax,
    sim_obsmat.freqs,
    true_params["beta_dust"],
    sim_obsmat.temp_dust,
    sim_obsmat.freq_pivot_dust,
)

dust_bb = np.asarray(
    dust_cov[freq_idx, freq_idx]
)


# ------------------------------------------------------------
# Synchrotron BB
# ------------------------------------------------------------

sync_cov = spectra_utils.get_sync_spectra(
    true_params["A_s_BB"],
    true_params["alpha_s_BB"],
    sim_obsmat.lmax,
    sim_obsmat.freqs,
    true_params["beta_sync"],
    sim_obsmat.freq_pivot_sync,
)

sync_bb = np.asarray(
    sync_cov[freq_idx, freq_idx]
)


# ------------------------------------------------------------
# Dust x Synchrotron
# ------------------------------------------------------------

cross_cov = spectra_utils.get_dust_sync_cross_spectra(
    true_params["rho_ds"],
    true_params["A_d_BB"],
    true_params["alpha_d_BB"],
    true_params["A_s_BB"],
    true_params["alpha_s_BB"],
    sim_obsmat.lmax,
    sim_obsmat.freqs,
    true_params["beta_dust"],
    sim_obsmat.temp_dust,
    true_params["beta_sync"],
    sim_obsmat.freq_pivot_dust,
    sim_obsmat.freq_pivot_sync,
)

cross_bb = np.asarray(
    cross_cov[freq_idx, freq_idx]
)


# ============================================================
# Total theoretical BB
#
# C_l^BB =
#
# CMB
# + Dust
# + Sync
# + 2 Dust x Sync
# ============================================================

cl_theory = (
    cmb_bb
    + dust_bb
    + sync_bb
    + 2.0 * cross_bb
)


# ============================================================
# Beam
#
# Maps are beam-convolved:
#
# C_l -> B_l^2 C_l
# ============================================================

beam2 = (
    sim_obsmat.b_ells[freq_idx] ** 2
)

cl_theory_beam = (
    cl_theory * beam2
)


# ============================================================
# Convert C_l -> D_l
#
# D_l = l(l+1)/(2 pi) C_l
# ============================================================

ell_theory = np.arange(
    sim_obsmat.lmax + 1
)

Dl_theory = (
    ell_theory
    * (ell_theory + 1)
    / (2.0 * np.pi)
    * cl_theory_beam
)


ell_bin = cl_sky["ell"]

Dl_sky = (
    ell_bin
    * (ell_bin + 1)
    / (2.0 * np.pi)
    * cl_sky["BB"]
)

Dl_obs = (
    ell_bin
    * (ell_bin + 1)
    / (2.0 * np.pi)
    * cl_obs["BB"]
)


# ============================================================
# Plot 1:
#
# Input theory vs sky vs ObsMat
# ============================================================

good_theory = (
    (ell_theory >= 2)
    & np.isfinite(Dl_theory)
    & (Dl_theory > 0)
)

good_sky = (
    (ell_bin >= 2)
    & np.isfinite(Dl_sky)
    & (Dl_sky > 0)
)

good_obs = (
    (ell_bin >= 2)
    & np.isfinite(Dl_obs)
    & (Dl_obs > 0)
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.loglog(
    ell_theory[good_theory],
    Dl_theory[good_theory],
    lw=2,
    label="Input theory",
)


ax.loglog(
    ell_bin[good_sky],
    Dl_sky[good_sky],
    "o-",
    ms=3,
    label="Sky: split 0 × split 1",
)


ax.loglog(
    ell_bin[good_obs],
    Dl_obs[good_obs],
    "o-",
    ms=3,
    label="ObsMat: split 0 × split 1",
)


ax.set_xlabel(
    r"$\ell$"
)

ax.set_ylabel(
    r"$D_\ell^{BB}\,[\mu{\rm K}^2]$"
)

ax.set_title(
    f"BB cross-spectrum @ "
    f"{sim_obsmat.freq_strings[freq_idx]}"
)


ax.legend()

ax.grid(
    True,
    which="both",
    alpha=0.3,
)


fig.tight_layout()


outfile = os.path.join(
    figdir,
    f"bb_cross_spectrum_obsmat_vs_sky_"
    f"{sim_obsmat.freq_strings[freq_idx]}",
)


fig.savefig(
    outfile + ".png",
    dpi=200,
    bbox_inches="tight",
)

fig.savefig(
    outfile + ".pdf",
    bbox_inches="tight",
)


print()
print("Saved:")
print(outfile + ".png")
print(outfile + ".pdf")


plt.show()
plt.close(fig)


# ============================================================
# ObsMat transfer function
#
# T_l =
#
# C_l(ObsMat)
# -------------
# C_l(sky)
#
# using split 0 x split 1
# ============================================================

ratio = np.full(
    cl_obs["BB"].shape,
    np.nan,
    dtype=float,
)


valid = (
    np.isfinite(cl_sky["BB"])
    & np.isfinite(cl_obs["BB"])
    & (cl_sky["BB"] != 0)
)


ratio[valid] = (
    cl_obs["BB"][valid]
    / cl_sky["BB"][valid]
)


good_ratio = (
    (ell_bin >= 2)
    & np.isfinite(ratio)
)


# ============================================================
# Plot 2:
#
# ObsMat BB transfer function
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.plot(
    ell_bin[good_ratio],
    ratio[good_ratio],
    "o-",
    ms=3,
)


ax.axhline(
    1.0,
    linestyle="--",
)


ax.set_xlabel(
    r"$\ell$"
)

ax.set_ylabel(
    r"$C_\ell^{BB,\mathrm{ObsMat}}/"
    r"C_\ell^{BB,\mathrm{sky}}$"
)

ax.set_title(
    f"ObsMat BB transfer function @ "
    f"{sim_obsmat.freq_strings[freq_idx]}"
)


ax.grid(
    True,
    alpha=0.3,
)


fig.tight_layout()


outfile = os.path.join(
    figdir,
    f"bb_obsmat_transfer_"
    f"{sim_obsmat.freq_strings[freq_idx]}",
)


fig.savefig(
    outfile + ".png",
    dpi=200,
    bbox_inches="tight",
)

fig.savefig(
    outfile + ".pdf",
    bbox_inches="tight",
)


print()
print("Saved:")
print(outfile + ".png")
print(outfile + ".pdf")


plt.show()
plt.close(fig)