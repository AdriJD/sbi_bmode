"""
Smoke/integration tests for the lenshILC-enabled CMBSimulator.

Run from the repository root with:

    pytest -q -s tests/test_sim_lenshILC_utils.py

The test deliberately uses:
- a small nside/lmax;
- two independent splits;
- the identity lensing operator;
- a loose CG tolerance;

so that it checks the full draw_data -> map2alm -> CG reconstruction ->
cross-spectra -> binned data-vector path without making the test unnecessarily
expensive.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Change this import only if the simulator module has a different filename.
from sbi_bmode.sim_lenshILC_utils import CMBSimulator


def _find_specdir() -> Path:
    """Locate the directory containing the two CAMB spectrum files."""
    required = ("camb_lens_nobb.dat", "camb_lens_r1.dat")

    candidates = []

    env_specdir = os.environ.get("SBI_BMODE_SPECDIR")
    if env_specdir:
        candidates.append(Path(env_specdir).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent / "data",
            here.parent.parent / "data",
            here.parent.parent / "sbi_bmode" / "data",
        ]
    )

    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    pytest.fail(
        "Could not find camb_lens_nobb.dat and camb_lens_r1.dat.\n"
        "Set SBI_BMODE_SPECDIR to the directory containing them.\n"
        f"Searched:\n{searched}"
    )


@pytest.fixture(scope="module")
def simulator() -> CMBSimulator:
    specdir = _find_specdir()

    # These channels must match the rows returned by lenshILC_utils.build_A().
    # Override them without editing the test, for example:
    #
    # SBI_BMODE_TEST_FREQS=f025,f027,f039,f093,f145,f225,f280,f350 pytest -q -s
    #
    freq_strings = os.environ.get(
        "SBI_BMODE_TEST_FREQS",
        "f025,f027,f039,f093,f145,f225,f280,f350",
    ).split(",")

    data_dict = {
        "lmax": 23,
        "lmin": 2,
        "nside": 8,
        "nsplit": 2,
        "delta_ell": 7,
        "highpass_delta_ell": 2,
        "freq_strings": freq_strings,
        "sensitivity_mode": "goal",
        "lknee_mode": "optimistic",
    }

    fixed_params_dict = {
        "freq_pivot_dust": 353e9,
        "freq_pivot_sync": 23e9,
        "temp_dust": 19.6,
    }

    return CMBSimulator(
        specdir=str(specdir),
        data_dict=data_dict,
        fixed_params_dict=fixed_params_dict,
        norm_params=None,
        score_params=None,
        apply_highpass_filter=False,
    )


def test_draw_data_identity_lensing_smoke(simulator: CMBSimulator) -> None:
    """Draw one realization and check the complete identity-lensing CG path."""
    result = simulator.draw_data(
        r_tensor=0.01,
        A_d_BB=5.0,
        alpha_d_BB=-0.4,
        beta_dust=1.6,
        seed=1234,
        return_maps=True,
        return_alms=True,
        run_lenshilc=True,
        use_lensing_operator=False,
        lens_components=(),
        cg_bin_size=8,
        cg_maxiter=100,
        cg_tol=1e-7,
    )

    required_keys = {
        "data",
        "maps",
        "data_alms",
        "component_alms",
        "component_names",
        "mixing_matrix",
    }
    assert required_keys.issubset(result), result.keys()

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
    assert result["mixing_matrix"].shape == (nfreq, ncomp)

    # data ordering is:
    # split-pair, polarization (EE then BB), ell-bin
    assert result["data"].shape == (npair * 2 * nbin,)

    assert "cmb" in result["component_names"]
    assert np.isfinite(result["mixing_matrix"]).all()
    assert np.isfinite(result["component_alms"]).all()
    assert np.isfinite(result["data"]).all()


def test_draw_data_is_reproducible(simulator: CMBSimulator) -> None:
    """The same seeds and solver settings should produce the same data vector."""
    kwargs = dict(
        r_tensor=0.01,
        A_d_BB=5.0,
        alpha_d_BB=-0.4,
        beta_dust=1.6,
        seed=91,
        use_lensing_operator=False,
        lens_components=(),
        cg_bin_size=8,
        cg_maxiter=100,
        cg_tol=1e-7,
    )

    first = simulator.draw_data(**kwargs)
    second = simulator.draw_data(**kwargs)

    np.testing.assert_allclose(
        first["data"],
        second["data"],
        rtol=1e-10,
        atol=1e-12,
    )


def test_different_seeds_change_the_data(simulator: CMBSimulator) -> None:
    """Different CMB/foreground/noise draws should not give identical vectors."""
    common = dict(
        r_tensor=0.01,
        A_d_BB=5.0,
        alpha_d_BB=-0.4,
        beta_dust=1.6,
        use_lensing_operator=False,
        lens_components=(),
        cg_bin_size=8,
        cg_maxiter=100,
        cg_tol=1e-7,
    )

    first = simulator.draw_data(seed=10, **common)
    second = simulator.draw_data(seed=11, **common)

    assert not np.allclose(first["data"], second["data"])
