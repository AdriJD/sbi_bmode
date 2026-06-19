import numpy as np

from sim_lenshILC_utils import CMBSimulator


specdir = "/path/to/specdir"

data_dict = {
    "lmax": 64,
    "lmin": 20,
    "nside": 32,
    "nsplit": 2,
    "delta_ell": 20,
    "freq_strings": ["f090", "f150"],
    "sensitivity_mode": "baseline",
    "lknee_mode": "optimistic",
    "highpass_delta_ell": 10,
}

fixed_params_dict = {
    "freq_pivot_dust": 353e9,
    "freq_pivot_sync": 23e9,
    "temp_dust": 19.6,
}

sim = CMBSimulator(
    specdir=specdir,
    data_dict=data_dict,
    fixed_params_dict=fixed_params_dict,
    apply_highpass_filter=True,
    mask_file=None,
)

out = sim.draw_data(
    r_tensor=0.01,
    A_d_BB=4.0,
    alpha_d_BB=-0.42,
    beta_dust=1.6,
    seed=1234,
    return_maps=True,
)

print("data vector shape:", out["data"].shape)
print("maps shape:", out["maps"].shape)
print("spectra shape:", out["spectra"].shape)
print("coadded spectra shape:", out["spectra_coadd"].shape)

assert out["maps"].shape == (
    data_dict["nsplit"],
    len(data_dict["freq_strings"]),
    2,
    12 * data_dict["nside"] ** 2,
)

assert np.all(np.isfinite(out["data"]))
assert np.all(np.isfinite(out["maps"]))

print("draw_data test passed")