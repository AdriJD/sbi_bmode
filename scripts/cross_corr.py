import numpy as np
from sbi_bmode import sim_utils, script_utils
import yaml
import matplotlib.pyplot as plt
import pymaster as nmt
import matplotlib.colors as colors

with open('/u/bing/sbi_bmode/scripts/configs/config_tf.yaml') as f:
    config = yaml.safe_load(f)
    
data_dict, fixed_params_dict, params_dict, observation_dict, transfer_dict = \
    script_utils.parse_config(config)
    
true_params = script_utils.get_true_params(params_dict)
specdir = '/u/bing/sbi_bmode/data'
figdir = "/u/bing/sbi_bmode/binh/figures"
nsims = transfer_dict['nsims']

bins = nmt.NmtBin.from_nside_linear(
    data_dict['nside'],
    transfer_dict['ell_bin']
)

nbins = bins.get_n_bands()

sim_obsmat = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict={
        'type': 'obsmat',
        'obsmat_dir': '/u/bing/so-data/mss2',
        'obsmat_freq': 'f090'
    },
    apply_highpass_filter=False
)

sim_tf = sim_utils.CMBSimulator(
    specdir,
    data_dict,
    fixed_params_dict,
    observation_dict=observation_dict,
    apply_highpass_filter=False
)

fidx = sim_obsmat.freq_strings.index('f090')

cl_obsmat = np.zeros((nsims, nbins))
cl_tf = np.zeros((nsims, nbins))

for i in range(nsims):
    out_obsmat = sim_obsmat.draw_data(
        **true_params,
        seed=i,
        return_maps=True,
    )

    out_tf = sim_tf.draw_data(
        **true_params,
        seed=i,
        return_maps=True,
    )

    cl_obsmat[i] = sim_utils.estimate_pseudo_cl(
        out_obsmat["obs_map"][0, fidx],
        out_obsmat["obs_map"][1, fidx],
        ell_bin=transfer_dict["ell_bin"],
        nside=data_dict["nside"],
        mask_dir=transfer_dict["mask_file"],
        apod_scale=transfer_dict["apod_scale"],
        apod_type=transfer_dict["apod_type"],
    )["BB"]


    cl_tf[i] = sim_utils.estimate_pseudo_cl(
        out_tf["obs_map"][0, fidx],
        out_tf["obs_map"][1, fidx],
        ell_bin=transfer_dict["ell_bin"],
        nside=data_dict["nside"],
        mask_dir=transfer_dict["mask_file"],
        apod_scale=transfer_dict["apod_scale"],
        apod_type=transfer_dict["apod_type"],
    )["BB"]
    
mean_obsmat = np.mean(cl_obsmat, axis=0)
mean_tf = np.mean(cl_tf, axis=0)

cov_obsmat = np.cov(
    cl_obsmat,
    rowvar=False,
)

cov_tf = np.cov(
    cl_tf,
    rowvar=False,
)

sigma_obsmat = np.sqrt(np.diag(cov_obsmat))
sigma_tf = np.sqrt(np.diag(cov_tf))

corr_obsmat = cov_obsmat / np.outer(sigma_obsmat, sigma_obsmat)
corr_tf = cov_tf / np.outer(sigma_tf, sigma_tf)

ell_eff = bins.get_effective_ells()
ratio = mean_tf / mean_obsmat

np.savez(
    f"{figdir}/comparison_results.npz",

    ell_eff=ell_eff,

    cl_obsmat=cl_obsmat,
    cl_tf=cl_tf,

    mean_obsmat=mean_obsmat,
    mean_tf=mean_tf,

    cov_obsmat=cov_obsmat,
    cov_tf=cov_tf,

    ratio=ratio,
)

plt.figure(figsize=(7,5))

plt.errorbar(
    ell_eff,
    mean_obsmat,
    yerr=np.sqrt(np.diag(cov_obsmat)),
    fmt="o",
    label="ObsMat",
)

plt.errorbar(
    ell_eff,
    mean_tf,
    yerr=np.sqrt(np.diag(cov_tf)),
    fmt="s",
    label="Transfer function",
)

plt.yscale("log")
plt.xlim(0, 200)
plt.ylim(1e-7, 1e-4)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$D_\ell^{BB}$")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{figdir}/BB_mean_comparison.png",
    dpi=200,
)

plt.close()


# -------------------------------------------------------
# Plot covariance matrices
# -------------------------------------------------------

extent = [
    ell_eff[0],
    ell_eff[19],
    ell_eff[0],
    ell_eff[19],
]

plt.figure(figsize=(8,6))

im = plt.imshow(
    cov_tf[:19, :19],
    origin="lower",
    extent=extent,
    aspect="equal",
    cmap="coolwarm",
    vmin=0,
    vmax=1.5e-12,
)

plt.colorbar(im, label="Covariance")

plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell$")

plt.savefig("cov_tf.png", bbox_inches="tight", dpi=300)
plt.close()


plt.figure(figsize=(8,6))

im = plt.imshow(
    cov_obsmat[:19, :19],
    origin="lower",
    extent=extent,
    aspect="equal",
    cmap="coolwarm",
    vmin=0,
    vmax=1.5e-12,
)

plt.colorbar(im, label="Covariance")

plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell$")

plt.savefig("cov_obsmat.png", bbox_inches="tight", dpi=300)
plt.close()

plt.figure(figsize=(7,5))

plt.plot(
    ell_eff,
    ratio,
    "o-"
)

plt.axhline(1, color="k", ls="--")

plt.xlabel(r"$\ell$")
plt.ylabel(
    r"$C_\ell^{BB}(\mathrm{TF})/"
    r"C_\ell^{BB}(\mathrm{ObsMat})$"
)

plt.grid(alpha=0.3)

plt.savefig(
    f"{figdir}/TF_vs_ObsMat_ratio.png",
    dpi=200,
)

plt.close()

plt.figure(figsize=(8,6))

im = plt.imshow(
    corr_tf[:19, :19],
    origin="lower",
    extent=extent,
    aspect="equal",
    cmap="coolwarm",
    vmax=1,
    vmin=-0.8
)

plt.colorbar(im, label="Correlation")

plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell$")

plt.savefig("../binh/figures/corr_tf.png", bbox_inches="tight", dpi=300)
plt.close()


plt.figure(figsize=(8,6))

im = plt.imshow(
    corr_obsmat[:19, :19],
    origin="lower",
    extent=extent,
    aspect="equal",
    cmap="coolwarm",
    vmax=1,
    vmin=-0.8
)

plt.colorbar(im, label="Correlation")

plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell$")

plt.savefig("../binh/figures/corr_obsmat.png", bbox_inches="tight", dpi=300)
plt.close()