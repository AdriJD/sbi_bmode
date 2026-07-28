import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os


mask_file = "/u/bing/sbi_bmode/data/combined_mask_nside256_ring.fits"

outdir = "/u/bing/sbi_bmode/binh/figures"
os.makedirs(outdir, exist_ok=True)


# -------------------------------
# Read mask
# -------------------------------

mask = hp.read_map(
    mask_file,
    dtype=np.float64,
)

print("Original nside:", hp.get_nside(mask))
print("npix:", mask.size)


# -------------------------------
# Downgrade to nside 128
# -------------------------------

mask_nside128 = hp.ud_grade(
    mask,
    nside_out=128,
    order_in="RING",
    order_out="RING",
    power=0,        # important for masks
)


print("New nside:", hp.get_nside(mask_nside128))


# -------------------------------
# Plot
# -------------------------------

plt.figure(figsize=(10, 6))

hp.mollview(
    mask_nside128,
    title="Combined mask (nside=128)",
    unit="Mask",
    min=0,
    max=1,
    cmap="viridis",
)

plt.savefig(
    os.path.join(outdir, "combined_mask_nside128.png"),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# -------------------------------
# Save downgraded map
# -------------------------------

hp.write_map(
    "/u/bing/sbi_bmode/data/combined_mask_nside128_ring.fits",
    mask_nside128,
    overwrite=True,
)

print("Saved downgraded mask.")