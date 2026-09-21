import numpy as np
import matplotlib.pyplot as plt

results = np.load("../data/comparison_results.npz")

cov_obsmat = results['cov_obsmat']
cov_tf = results['cov_tf']
ell_eff = results['ell_eff']
sigma_obsmat = np.sqrt(np.diag(cov_obsmat))
sigma_tf = np.sqrt(np.diag(cov_tf))

corr_obsmat = cov_obsmat / np.outer(sigma_obsmat, sigma_obsmat)
corr_tf = cov_tf / np.outer(sigma_tf, sigma_tf)

extent = [
    ell_eff[0],
    ell_eff[19],
    ell_eff[0],
    ell_eff[19],
]

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