import os
import numpy as np
from sbi_bmode import sim_utils, script_utils
import yaml
import matplotlib.pyplot as plt

with open('/u/bing/sbi_bmode/scripts/configs/config_compute_transfer_4.yaml') as f:
    config = yaml.safe_load(f)

data_dict, fixed_params_dict, params_dict, observation_dict, transfer_dict = \
    script_utils.parse_config(config)

true_params = script_utils.get_true_params(params_dict)

specdir = '/u/bing/sbi_bmode/data'
figdir = "/ptmp/bing/sbi_bmode_storage/figures/compute_transfer_4"
os.makedirs(figdir, exist_ok=True)
nsims = transfer_dict['nsims']

# ------------------------------------------------------------------
# Simulators
# ------------------------------------------------------------------

sim_sky = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict={'type': 'identity'},
    apply_highpass_filter=False
)

sim_obs = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=observation_dict,
    apply_highpass_filter=False
)

fidx = sim_sky.freq_strings.index(observation_dict['obsmat_freq'])

nsplit = data_dict['nsplit']
assert nsplit == 2, (
    "estimate_transfer_function only supports nsplit == 2 "
    f"(got nsplit={nsplit})"
)

maps_sky = np.zeros((nsims, nsplit, 2, sim_sky.minfo.npix))
maps_obs = np.zeros((nsims, nsplit, 2, sim_obs.minfo.npix))

for i in range(nsims):
    out_sky = sim_sky.draw_data(
        **true_params,
        seed=i,
        return_maps=True,
    )

    out_obs = sim_obs.draw_data(
        **true_params,
        seed=i,
        return_maps=True,
    )

    maps_sky[i] = out_sky["obs_map"][:, fidx]   # (nsplit, 2, npix)
    maps_obs[i] = out_obs["obs_map"][:, fidx]

# ------------------------------------------------------------------
# Estimate transfer function
# ------------------------------------------------------------------

sqrt_transfer_ell, diag = sim_utils.estimate_transfer_function(
    maps_sky,
    maps_obs,
    delta_ell=transfer_dict['ell_bin'],
    nside=data_dict['nside'],
    lmax=data_dict['lmax'],
    mask_file=transfer_dict['mask_file'],
    apod_mask_file=transfer_dict.get('apod_mask_file'),
    apod_scale=transfer_dict['apod_scale'],
    apod_type=transfer_dict['apod_type'],
    return_diagnostics=True,
)

out_path = f"{specdir}/{transfer_dict['output_file']}"
np.save(out_path, sqrt_transfer_ell)

print(f"Saved amplitude-level transfer function to {out_path}")

# ------------------------------------------------------------------
# Plot 1 : Transfer function (binned, power-space)
# ------------------------------------------------------------------

plt.figure(figsize=(7,5))

plt.errorbar(
    diag["ell"],
    diag["transfer_bins_mean"],
    yerr=diag["transfer_bins_err"],
    fmt="o",
    capsize=3,
    label="Monte Carlo mean",
)

plt.plot(
    diag["ell"],
    diag["bb_obs_mean"] / diag["bb_sky_mean"],
    "-",
    lw=2,
    label=r"$\langle C_\ell^{obs}\rangle/\langle C_\ell^{sky}\rangle$",
)

plt.axhline(1.0, color="k", ls="--", alpha=0.5)

plt.xlim(1, 200)
plt.ylim(-0.05, 1)

plt.xlabel(r"$\ell$")
plt.ylabel(r"$T_\ell$ (power)")
plt.title("Transfer Function (power-space, binned)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(f"{figdir}/transfer_function_binned.png", dpi=200)


# ------------------------------------------------------------------
# Plot 2 : BB spectra
# ------------------------------------------------------------------

plt.figure(figsize=(7,5))

plt.errorbar(
    diag["ell"],
    diag["bb_sky_mean"],
    yerr=diag["bb_sky_err"],
    fmt="o-",
    capsize=3,
    label="Sky",
)

plt.errorbar(
    diag["ell"],
    diag["bb_obs_mean"],
    yerr=diag["bb_obs_err"],
    fmt="s-",
    capsize=3,
    label="Observed",
)

plt.yscale("log")
plt.xlim(0, 200)

plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{BB}$")
plt.title("Mean BB Spectra")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(f"{figdir}/bb_spectra.png", dpi=200)


# ------------------------------------------------------------------
# Plot 3 : Full interpolated transfer function (amplitude vs power)
# ------------------------------------------------------------------

ells = np.arange(len(sqrt_transfer_ell))

fig, ax1 = plt.subplots(figsize=(7,5))

ax1.plot(ells, sqrt_transfer_ell, lw=2, color="C0", label=r"amplitude, $\sqrt{T_\ell}$")
ax1.plot(ells, diag["transfer_ell_power"], lw=2, ls="--", color="C1", label=r"power, $T_\ell$")

ax1.set_xlim(0, 200)
ax1.set_ylim(-0.05, 1)
ax1.set_xlabel(r"$\ell$")
ax1.set_ylabel(r"$T_\ell$")
ax1.set_title("Interpolated Transfer Function")
ax1.grid(alpha=0.3)
ax1.legend()

plt.tight_layout()
plt.savefig(f"{figdir}/transfer_function_full.png", dpi=200)

plt.show()