import numpy as np
from sbi_bmode import sim_utils, script_utils
import yaml
import matplotlib.pyplot as plt

with open('/u/bing/sbi_bmode/scripts/configs/config_transfer_1.yaml') as f:
    config = yaml.safe_load(f)

data_dict, fixed_params_dict, params_dict, observation_dict, transfer_dict = \
    script_utils.parse_config(config)

true_params = script_utils.get_true_params(params_dict)

specdir = '/u/bing/sbi_bmode/data'
figdir = "/u/bing/sbi_bmode/binh/figures"
nsims = transfer_dict['nsims']

# ------------------------------------------------------------------
# Simulators
# ------------------------------------------------------------------

sim_sky = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict={'type': 'identity'},
)

sim_obs = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=observation_dict,
)

fidx = sim_sky.freq_strings.index(observation_dict['obsmat_freq'])

maps_sky = np.zeros((nsims, 2, sim_sky.minfo.npix))
maps_obs = np.zeros((nsims, 2, sim_obs.minfo.npix))

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

    maps_sky[i] = out_sky["obs_map"][0, fidx]
    maps_obs[i] = out_obs["obs_map"][0, fidx]

# ------------------------------------------------------------------
# Estimate transfer function
# ------------------------------------------------------------------

transfer_ell, diag = sim_utils.estimate_transfer_function(
    maps_sky,
    maps_obs,
    ell_bin=transfer_dict['ell_bin'],
    nside=data_dict['nside'],
    lmax=data_dict['lmax'],
    mask_dir=transfer_dict['mask_file'],
    apod_scale=transfer_dict['apod_scale'],
    apod_type=transfer_dict['apod_type'],
    return_diagnostics=True,
)

out_path = f"{specdir}/{transfer_dict['output_file']}"
np.save(out_path, transfer_ell)

print(f"Saved transfer function to {out_path}")
# ------------------------------------------------------------------
# Plot 1 : Transfer function (binned)
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

plt.xlim(0, 200)

plt.xlabel(r"$\ell$")
plt.ylabel(r"$T_\ell$")
plt.title("Transfer Function")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(f"{specdir}/transfer_function_binned.png", dpi=200)



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
# Plot 3 : Full transfer function
# ------------------------------------------------------------------

ells = np.arange(len(transfer_ell))

plt.figure(figsize=(7,5))

plt.plot(
    ells,
    transfer_ell,
    lw=2,
)

plt.xlim(0, 200)

plt.xlabel(r"$\ell$")
plt.ylabel(r"$T_\ell$")
plt.title("Interpolated Transfer Function")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{figdir}/transfer_function_full.png", dpi=200)

plt.show()