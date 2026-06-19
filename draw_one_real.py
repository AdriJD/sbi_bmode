import numpy as np

from sbi_bmode.sim_lenshILC_utils import CMBSimulator


specdir = "/home/xsong/sbi_bmode/data"

data_dict = {
    "lmax": 191,
    "lmin": 20,
    "nside": 64,
    "nsplit": 2,
    "delta_ell": 20,
    "highpass_delta_ell": 10,
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

print("Building simulator...")

sim = CMBSimulator(
    specdir=specdir,
    data_dict=data_dict,
    fixed_params_dict=fixed_params_dict,
    norm_params=None,
    score_params=None,
    apply_highpass_filter=False,
)

print("Simulator built")
print("freqs:", sim.freq_strings)
print("bins:", sim.bins)
print("ainfo.nelem:", sim.ainfo.nelem)
print("minfo.npix:", sim.minfo.npix)

print("\nDrawing lensed realization...")

out = sim.draw_data(
    r_tensor=0.01,
    A_d_BB=5.0,
    alpha_d_BB=-0.4,
    beta_dust=1.6,
    seed=1234,

    return_maps=False,
    return_alms=True,

    # Turn on the fixed-phi lensing operator.
    use_lensing_operator=True,
    lens_seed=5678,
    lens_components=(0,),

    # Extra multipoles used for the deflection field.
    lens_dlmax=128,
    lens_epsilon=1e-6,

    # CG settings.
    cg_bin_size=20,
    cg_maxiter=300,
    cg_tol=1e-7,
)

print("\ndraw_data finished")

for key, value in out.items():
    if isinstance(value, np.ndarray):
        print(
            f"{key:20s}",
            "shape =", value.shape,
            "dtype =", value.dtype,
            "finite =", np.isfinite(value).all(),
        )
    else:
        print(f"{key:20s}", value)

npair = sim.nsplit * (sim.nsplit - 1) // 2
nbin = sim.bins.size - 1

data = out["data"].reshape(npair, 2, nbin)

print("\nEE bandpowers:")
print(data[:, 0])

print("\nBB bandpowers:")
print(data[:, 1])
