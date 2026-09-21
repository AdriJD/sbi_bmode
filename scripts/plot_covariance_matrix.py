import matplotlib.pyplot as plt
import numpy as np 

data = np.load("../binh/figures/comparison_results.npz")

cov_obsmat = data['cov_obsmat']
cov_tf = data['cov_tf']
ell_eff = data['ell_eff']

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

