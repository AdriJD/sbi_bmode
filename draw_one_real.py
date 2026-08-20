import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from sbi_bmode.sim_lenshILC_utils import CMBSimulator
from sbi_bmode import lenshILC_utils


specdir = "/home/xsong/sbi_bmode/data"

data_dict = {
    "lmax": 500,
    "lmin": 20,
    "nside": 512,
    "nsplit": 2,
    "delta_ell": 20,
    "highpass_delta_ell": 10,
    "freq_strings": ["f030", "f040", "f090", "f150", "f230", "f290"],
    "sensitivity_mode": "goal",
    "lknee_mode": "optimistic",
}

fixed_params_dict = {
    "freq_pivot_dust": 353e9,
    "freq_pivot_sync": 23e9,
    "temp_dust": 19.6,
}

fiducial_params_dict = {
    "beta_dust": 1.6,
    "T_dust": 19.6,
    "beta_sync": None,
    "include_dust_beta1": False,
    "include_sync_beta1": False,
}

lenshILC_utils.fixed_params = fixed_params_dict


print("Building simulator...")

sim = CMBSimulator(specdir=specdir, data_dict=data_dict, fixed_params_dict=fixed_params_dict, fiducial_beta=fiducial_params_dict["beta_dust"], fiducial_T_dust=fiducial_params_dict["T_dust"], fiducial_beta_sync=fiducial_params_dict["beta_sync"], score_params=None, apply_highpass_filter=False)

print("Simulator built")
print("freqs:", sim.freq_strings)
print("bins:", sim.bins)
print("ainfo.nelem:", sim.ainfo.nelem)
print("minfo.npix:", sim.minfo.npix)
print("mixing matrix shape:", sim.A.shape)
print("mixing matrix:")
print(sim.A)


# Debugging: remove beams for now.
sim.b_ells[:] = 1.0


# Lensing setup.
lens_dlmax = 128
lens_epsilon = 1e-6
lmax_phi = sim.lmax + lens_dlmax

print("\nLensing setup")
print("CMB lmax:", sim.lmax)
print("lens dlmax:", lens_dlmax)
print("phi lmax:", lmax_phi)


# Draw one phi field to act as the observed/reconstruction field.
_, phi_planck_alm = lenshILC_utils.generate_defl(lmax_len=sim.lmax, nside=sim.nside, seed=999, dlmax=lens_dlmax, epsilon=lens_epsilon)

print("phi_planck_alm shape:", phi_planck_alm.shape)
print("phi_planck_alm lmax:", hp.Alm.getlmax(len(phi_planck_alm)))


# Fiducial C_L^{phi phi}.
S_l_full = lenshILC_utils.camb_clfile(str(lenshILC_utils.LENSPOTENTIAL_CLS))["pp"]
S_l = np.asarray(S_l_full[:lmax_phi + 1], dtype=float)

if len(S_l) != lmax_phi + 1:
    raise ValueError(f"S_l has length {len(S_l)}, expected {lmax_phi + 1}.")


# Toy reconstruction noise N_L.
N_l = np.zeros_like(S_l)
N_l[2:] = 1e-4 * S_l[2:]

S_l[:2] = 0.0
N_l[:2] = 0.0
for L in [10, 20, 50, 100, 200, 300, 500]:
    print(L, "N/S =", N_l[L] / S_l[L])
print("S_l shape:", S_l.shape)
print("N_l shape:", N_l.shape)
print("S_l finite:", np.isfinite(S_l).all())
print("N_l finite:", np.isfinite(N_l).all())
print("S_l min/max:", np.min(S_l[2:]), np.max(S_l[2:]))
print("N_l min/max:", np.min(N_l[2:]), np.max(N_l[2:]))


# Draw one realization.
print("\nDrawing one realization...")

out = sim.draw_data(r_tensor=0.01, A_d_BB=5.0, alpha_d_BB=-0.4, beta_dust=1.6, seed=1234, return_maps=False, return_unbinned_spectra=True, use_lensing_operator=True, phi_planck_alm=phi_planck_alm, phi_signal_cl=S_l, phi_noise_cl=N_l, lens_seed=5678, lens_components=(0,), lens_dlmax=lens_dlmax, lens_epsilon=lens_epsilon, cg_bin_size=20, cg_maxiter=100, cg_tol=1e-12)

print("\ndraw_data finished")


# Inspect output.
for key, value in out.items():
    if isinstance(value, np.ndarray):
        print(f"{key:30s} shape={value.shape} dtype={value.dtype} finite={np.isfinite(value).all()}")
    else:
        print(f"{key:30s} {value}")

print("\ndata:")
print(out["data"])


# Save output.
output_file = "test_output_cr.npz"
np.savez_compressed(output_file, **out, phi_planck_alm=phi_planck_alm, phi_signal_cl=S_l, phi_noise_cl=N_l)

print(f"\nSaved output to {output_file}")
