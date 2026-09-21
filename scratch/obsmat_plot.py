from sbi_bmode import so_utils

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os


# ============================================================
# Load ObsMats
# ============================================================

obsmat_dir = "/u/bing/so-data/mss2"

freqs = ["f030", "f040", "f090", "f150", "f230", "f290"]

obsmat = so_utils.load_obs_matrix(freqs, obsmat_dir)


# ============================================================
# Input sky
# ============================================================

nside = 128
npix = hp.nside2npix(nside)

I_in = np.zeros(npix)
Q_in = np.zeros(npix)
U_in = np.zeros(npix)

# One Q pixel = 1
ipix_nest = 163286

Q_in[ipix_nest] = 1.0

sky_in = np.concatenate([
    I_in,
    Q_in,
    U_in,
])


# ============================================================
# Output directory
# ============================================================

output_dir = "/ptmp/bing/2026_sbi_bmode/img/obsmat"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Apply ObsMat and plot Q for each frequency
# ============================================================

for freq in freqs:

    print(f"Processing {freq}...")

    # Apply observation matrix
    sky_out = obsmat[freq].apply(sky_in)

    # Split I, Q, U
    idx = sky_out.shape[0] // 3

    I_out = sky_out[:idx]
    Q_out = sky_out[idx:2 * idx]
    U_out = sky_out[2 * idx:]

    # Copy for plotting
    Q_plot = Q_out.copy()

    # Hide exactly zero pixels
    Q_plot[Q_plot == 0] = hp.UNSEEN

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(10, 6))

    hp.mollview(
        Q_plot,
        title=f"Q output — {freq}",
        unit=r"$\mu$K",
        nest=True,
        cmap="coolwarm",
        min=-0.05,
        max=0.92,
        fig=plt.gcf(),
        norm="hist"
    )

    # ========================================================
    # Save
    # ========================================================

    outfile = os.path.join(
        output_dir,
        f"Q_out_{freq}.png",
    )

    plt.savefig(
        outfile,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {outfile}")