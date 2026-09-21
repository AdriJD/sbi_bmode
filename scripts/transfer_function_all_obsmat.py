"""
Estimate the B-mode amplitude-level transfer function separately for
each ObsMat / frequency band.

This is a refactor of compute_transfer_4.py: the simulation + estimation
logic is pulled into `run_transfer_estimation`, which takes the band as
an argument. You can then either loop over several bands in one process,
or run a single band per SLURM array task.

Usage
-----
Serial, loop over all default bands:
    python estimate_transfer_per_band.py --config configs/config_compute_transfer_4.yaml

Single band (e.g. as a SLURM array task):
    python estimate_transfer_per_band.py --config configs/config_compute_transfer_4.yaml --freq f150
"""

import os
import argparse
import copy

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")  # safe on compute nodes without a display
import matplotlib.pyplot as plt

from sbi_bmode import sim_utils, script_utils

# Adjust to the bands you actually have ObsMats for.
DEFAULT_BANDS = ["f030", "f040", "f090", "f150", "f220", "f280"]


def run_transfer_estimation(base_config, freq, specdir, base_figdir):
    """
    Run the full sky/obs simulation + transfer-function estimation
    for a single frequency band.

    Parameters
    ----------
    base_config : dict
        Parsed YAML config (as loaded from config_compute_transfer_4.yaml).
        `data.freq_strings`, `observation.obsmat_freq`, and
        `transfer_function.output_file` are overridden per band.
    freq : str
        Band identifier, e.g. "f090".
    specdir : str
        Directory containing power spectrum files and where output
        transfer functions are saved.
    base_figdir : str
        Parent directory for diagnostic plots; a per-band subdirectory
        is created under it.

    Returns
    -------
    sqrt_transfer_ell : (lmax + 1) array
        Amplitude-level transfer function for this band.
    diag : dict
        Diagnostics dict returned by sim_utils.estimate_transfer_function.
    """

    config = copy.deepcopy(base_config)
    config["data"]["freq_strings"] = [freq]
    config["observation"]["obsmat_freq"] = freq
    config["transfer_function"]["output_file"] = f"transfer_function_{freq}.npy"
    # apod_mask_file is left as-is on purpose: the sky mask/apodization
    # doesn't depend on frequency, so it's safe (and cheaper) to reuse
    # the same cached apodized mask across bands.

    data_dict, fixed_params_dict, params_dict, observation_dict, transfer_dict = \
        script_utils.parse_config(config)

    true_params = script_utils.get_true_params(params_dict)

    figdir = os.path.join(base_figdir, freq)
    os.makedirs(figdir, exist_ok=True)
    nsims = transfer_dict["nsims"]

    # ------------------------------------------------------------------
    # Simulators
    # ------------------------------------------------------------------

    sim_sky = sim_utils.CMBSimulator(
        specdir,
        data_dict,
        fixed_params_dict,
        observation_dict={"type": "identity"},
        apply_highpass_filter=False,
    )

    sim_obs = sim_utils.CMBSimulator(
        specdir,
        data_dict,
        fixed_params_dict,
        observation_dict=observation_dict,
        apply_highpass_filter=False,
    )

    fidx = sim_sky.freq_strings.index(observation_dict["obsmat_freq"])

    nsplit = data_dict["nsplit"]
    assert nsplit == 2, (
        "estimate_transfer_function only supports nsplit == 2 "
        f"(got nsplit={nsplit})"
    )

    maps_sky = np.zeros((nsims, nsplit, 2, sim_sky.minfo.npix))
    maps_obs = np.zeros((nsims, nsplit, 2, sim_obs.minfo.npix))

    for i in range(nsims):
        out_sky = sim_sky.draw_data(**true_params, seed=i, return_maps=True)
        out_obs = sim_obs.draw_data(**true_params, seed=i, return_maps=True)

        maps_sky[i] = out_sky["obs_map"][:, fidx]
        maps_obs[i] = out_obs["obs_map"][:, fidx]

    # ------------------------------------------------------------------
    # Estimate transfer function
    # ------------------------------------------------------------------

    sqrt_transfer_ell, diag = sim_utils.estimate_transfer_function(
        maps_sky,
        maps_obs,
        delta_ell=transfer_dict["ell_bin"],
        nside=data_dict["nside"],
        lmax=data_dict["lmax"],
        mask_file=transfer_dict["mask_file"],
        apod_mask_file=transfer_dict.get("apod_mask_file"),
        apod_scale=transfer_dict["apod_scale"],
        apod_type=transfer_dict["apod_type"],
        return_diagnostics=True,
    )

    out_path = os.path.join(specdir, transfer_dict["output_file"])
    np.save(out_path, sqrt_transfer_ell)
    print(f"[{freq}] saved amplitude-level transfer function to {out_path}")

    _make_diagnostic_plots(freq, diag, sqrt_transfer_ell, figdir)

    return sqrt_transfer_ell, diag


def _make_diagnostic_plots(freq, diag, sqrt_transfer_ell, figdir):
    """Reproduce the three diagnostic plots from compute_transfer_4.py, per band."""

    # Plot 1: transfer function (binned, power-space)
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        diag["ell"], diag["transfer_bins_mean"], yerr=diag["transfer_bins_err"],
        fmt="o", capsize=3, label="Monte Carlo mean",
    )
    plt.plot(
        diag["ell"], diag["bb_obs_mean"] / diag["bb_sky_mean"], "-", lw=2,
        label=r"$\langle C_\ell^{obs}\rangle/\langle C_\ell^{sky}\rangle$",
    )
    plt.axhline(1.0, color="k", ls="--", alpha=0.5)
    plt.xlim(1, 200)
    plt.ylim(-0.05, 1)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$T_\ell$ (power)")
    plt.title(f"Transfer Function (power-space, binned) - {freq}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "transfer_function_binned.png"), dpi=200)
    plt.close()

    # Plot 2: BB spectra
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        diag["ell"], diag["bb_sky_mean"], yerr=diag["bb_sky_err"],
        fmt="o-", capsize=3, label="Sky",
    )
    plt.errorbar(
        diag["ell"], diag["bb_obs_mean"], yerr=diag["bb_obs_err"],
        fmt="s-", capsize=3, label="Observed",
    )
    plt.yscale("log")
    plt.xlim(0, 200)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell^{BB}$")
    plt.title(f"Mean BB Spectra - {freq}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "bb_spectra.png"), dpi=200)
    plt.close()

    # Plot 3: full interpolated transfer function (amplitude vs power)
    ells = np.arange(len(sqrt_transfer_ell))
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(ells, sqrt_transfer_ell, lw=2, color="C0", label=r"amplitude, $\sqrt{T_\ell}$")
    ax1.plot(ells, diag["transfer_ell_power"], lw=2, ls="--", color="C1", label=r"power, $T_\ell$")
    ax1.set_xlim(0, 200)
    ax1.set_ylim(-0.05, 1)
    ax1.set_xlabel(r"$\ell$")
    ax1.set_ylabel(r"$T_\ell$")
    ax1.set_title(f"Interpolated Transfer Function - {freq}")
    ax1.grid(alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "transfer_function_full.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="Path to the base YAML config (e.g. config_compute_transfer_4.yaml).",
    )
    parser.add_argument(
        "--freq", default=None,
        help="Single band to run, e.g. f090. If omitted, loops over DEFAULT_BANDS.",
    )
    parser.add_argument(
        "--specdir", default="/u/bing/sbi_bmode/data",
        help="Directory with power spectrum files; also where .npy transfer functions are saved.",
    )
    parser.add_argument(
        "--figdir", default="/ptmp/bing/sbi_bmode_storage/figures/compute_transfer_per_band",
        help="Parent directory for per-band diagnostic plots.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    os.makedirs(args.figdir, exist_ok=True)

    bands = [args.freq] if args.freq else DEFAULT_BANDS

    results = {}
    for freq in bands:
        print(f"\n=== Estimating transfer function for {freq} ===")
        sqrt_transfer_ell, _ = run_transfer_estimation(
            base_config, freq, args.specdir, args.figdir
        )
        results[freq] = sqrt_transfer_ell

    return results


if __name__ == "__main__":
    main()