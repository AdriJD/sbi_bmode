"""
Integration test for sbi_bmode.sim_lenshILC_utils.CMBSimulator.

Run from the repository root:

    cd /home/xsong/sbi_bmode
    python -m pytest -q -s sbi_bmode/tests/test_sim_lenshILC_utils.py

This test runs the actual simulation and lenshILC/CG reconstruction using
a small HEALPix geometry and the identity lensing operator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sbi_bmode.sim_lenshILC_utils import CMBSimulator


@pytest.fixture(scope="module")
def specdir() -> Path:
    """
    Return the repository data directory.

    Expected layout:

        repo_root/
        ├── data/
        │   ├── camb_lens_nobb.dat
        │   └── camb_lens_r1.dat
        └── sbi_bmode/
            └── tests/
                └── test_sim_lenshILC_utils.py
    """
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[2]
    data_dir = repo_root / "data"

    required = (
        data_dir / "camb_lens_nobb.dat",
        data_dir / "camb_lens_r1.dat",
    )

    missing = [path for path in required if not path.is_file()]
    if missing:
        pytest.fail(
            "Missing required CAMB spectrum files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    return data_dir


@pytest.fixture(scope="module")
def simulator(specdir: Path) -> CMBSimulator:
    data_dict = {
        # Small values keep the integration test reasonably fast.
        "lmax": 23,
        "lmin": 2,
        "nside": 8,
        "nsplit": 2,
        "delta_ell": 7,
        "highpass_delta_ell": 2,

        # These are the frequency keys actually defined in so_utils.
        # Their ordering must match the rows produced by build_A().
        "freq_strings": [
            "f030",
            "f040",
            "f090",
            "f150",
            "f230",
            "f290",
        ],
        "sensitivity_mode": "goal",
        "lknee_mode": "optimistic",
    }

    fixed_params_dict = {
        "freq_pivot_dust": 353e9,
        "freq_pivot_sync": 23e9,
        "temp_dust": 19.6,
    }

    sim = CMBSimulator(
        specdir=str(specdir),
        data_dict=data_dict,
        fixed_params_dict=fixed_params_dict,
        norm_params=None,
        score_params=None,
        apply_highpass_filter=False,
    )

    assert sim.nfreq == 6
    assert sim.nsplit == 2
    assert sim.freq_strings == data_dict["freq_strings"]

    return sim


def _draw_kwargs(seed: int) -> dict:
    """Parameters shared by all draw_data calls."""
    return {
        "r_tensor": 0.01,
        "A_d_BB": 5.0,
        "alpha_d_BB": -0.4,
        "beta_dust": 1.6,
        "seed": seed,

        # Keep intermediate products for shape checks.
        "return_maps": True,
        "return_alms": True,

        # Fast smoke-test mode: no lenspyx remapping.
        "use_lensing_operator": False,
        "lens_components": (),

        # Less stringent settings than production.
        "cg_bin_size": 8,
        "cg_maxiter": 100,
        "cg_tol": 1e-7,
    }


def test_draw_one_data_realization(simulator: CMBSimulator) -> None:
    """
    Run the complete draw_data path and validate the returned products.

    The expected final data ordering is:

        split-pair × polarization(EE, BB) × ell-bin
    """
    result = simulator.draw_data(**_draw_kwargs(seed=1234))

    required_keys = {
        "data",
        "maps",
        "data_alms",
        "component_alms",
        "component_names",
        "mixing_matrix",
    }
    assert required_keys.issubset(result.keys())

    nsplit = simulator.nsplit
    nfreq = simulator.nfreq
    npair = nsplit * (nsplit - 1) // 2
    nbin = simulator.bins.size - 1
    ncomp = len(result["component_names"])

    assert result["maps"].shape == (
        nsplit,
        nfreq,
        2,
        simulator.minfo.npix,
    )

    assert result["data_alms"].shape == (
        nsplit,
        nfreq,
        2,
        simulator.ainfo.nelem,
    )

    assert result["component_alms"].shape == (
        nsplit,
        ncomp,
        2,
        simulator.ainfo.nelem,
    )

    assert result["mixing_matrix"].shape == (
        nfreq,
        ncomp,
    )

    assert result["data"].shape == (
        npair * 2 * nbin,
    )

    assert "cmb" in result["component_names"]

    assert np.isfinite(result["maps"]).all()
    assert np.isfinite(result["data_alms"]).all()
    assert np.isfinite(result["component_alms"]).all()
    assert np.isfinite(result["mixing_matrix"]).all()
    assert np.isfinite(result["data"]).all()

    print("\nSimulation completed successfully.")
    print("component_names:", result["component_names"])
    print("mixing_matrix shape:", result["mixing_matrix"].shape)
    print("component_alms shape:", result["component_alms"].shape)
    print("data shape:", result["data"].shape)
    print("data:", result["data"])


def test_same_seed_is_reproducible(simulator: CMBSimulator) -> None:
    """The same random seed should reproduce the same final data vector."""
    kwargs = _draw_kwargs(seed=91)

    # Intermediate maps/alms are not needed for this test.
    kwargs["return_maps"] = False
    kwargs["return_alms"] = False

    first = simulator.draw_data(**kwargs)
    second = simulator.draw_data(**kwargs)

    np.testing.assert_allclose(
        first["data"],
        second["data"],
        rtol=1e-10,
        atol=1e-12,
    )
