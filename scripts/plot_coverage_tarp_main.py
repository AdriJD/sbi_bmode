#!/usr/bin/env python3

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load TARP results
# ============================================================

def load_tarp(tarpdir):
    """
    Load TARP results from a directory.
    """

    alpha = np.load(os.path.join(tarpdir, "tarp_alpha.npy"))
    ecp = np.load(os.path.join(tarpdir, "tarp_ecp.npy"))
    ecp_boot = np.load(os.path.join(tarpdir, "tarp_ecp_boot.npy"))

    ecp_std = np.std(ecp_boot, axis=0)

    ecp_marg = np.load(
        os.path.join(tarpdir, "tarp_ecp_marg.npy")
    )

    ecp_marg_boot = np.load(
        os.path.join(tarpdir, "tarp_ecp_marg_boot.npy")
    )

    ecp_marg_std = np.std(
        ecp_marg_boot,
        axis=1
    )

    return {
        "alpha": alpha,
        "ecp": ecp,
        "ecp_std": ecp_std,
        "ecp_marg": ecp_marg,
        "ecp_marg_std": ecp_marg_std,
    }


# ============================================================
# Format axis
# ============================================================

def format_axis(ax, title):
    """
    Apply common formatting to a TARP plot.
    """

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    ax.set_xlabel(r"Credibility level $1-\alpha$")
    ax.set_ylabel(r"$\mathrm{ECP}$")

    ax.set_title(title)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        color="black",
        alpha=0.5,
    )


# ============================================================
# Plot TARP
# ============================================================

def plot_tarp(
    tarpdirs,
    labels,
    mode,
    imgdir,
    name,
    title,
):
    """
    Plot TARP results.

    Parameters
    ----------
    tarpdirs : list[str]
        Directories containing TARP results.

    labels : list[str]
        Labels corresponding to each TARP directory.

    mode : {"joint", "marginal"}
        Plot joint or marginal TARP.

    imgdir : str
        Output directory.

    name : str
        Base name of output files.

    title : str
        Plot title.
    """

    if len(tarpdirs) != len(labels):
        raise ValueError(
            "The number of TARP directories must match "
            "the number of labels."
        )

    os.makedirs(imgdir, exist_ok=True)

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    results = []

    for tarpdir in tarpdirs:

        if not os.path.isdir(tarpdir):
            raise FileNotFoundError(
                f"TARP directory does not exist:\n{tarpdir}"
            )

        print(f"Loading: {tarpdir}")

        results.append(load_tarp(tarpdir))

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    nresults = len(results)

    if nresults <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(nresults)]
    else:
        cmap = plt.get_cmap("viridis")
        colors = [
            cmap(i / max(nresults - 1, 1))
            for i in range(nresults)
        ]

    # ========================================================
    # JOINT TARP
    # ========================================================

    if mode == "joint":

        fig, ax = plt.subplots(
            figsize=(3.55, 3.55),
            dpi=300,
        )

        # Ideal calibration
        alpha = results[0]["alpha"]

        ax.plot(
            alpha,
            alpha,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Ideal",
        )

        # ----------------------------------------------------
        # Plot each result
        # ----------------------------------------------------

        for i, result in enumerate(results):
            ecp = result["ecp"]
            ecp_std = result["ecp_std"]

            ax.plot(
                alpha,
                ecp,
                color=colors[i],
                linewidth=1.2,
                label=labels[i],
            )

            ax.fill_between(
                alpha,
                ecp - 2.0 * ecp_std,
                ecp + 2.0 * ecp_std,
                color=colors[i],
                alpha=0.08,
            )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------

        format_axis(ax, title)
        ax.legend(
            loc="upper left",
            fontsize=7,
            frameon=False,
        )

        fig.tight_layout()

        png_file = os.path.join(
            imgdir,
            f"{name}.png",
        )

        pdf_file = os.path.join(
            imgdir,
            f"{name}.pdf",
        )

        fig.savefig(
            png_file,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf_file,
            bbox_inches="tight",
        )

        print(f"Saved: {png_file}")
        print(f"Saved: {pdf_file}")

        plt.show()

    # ========================================================
    # MARGINAL TARP
    # ========================================================

    elif mode == "marginal":

        # ----------------------------------------------------
        # Parameter names
        # ----------------------------------------------------

        parameters = [
            (0, r"$r$", "r"),
            (1, r"$A_{\mathrm{lens}}$", "Alens"),
        ]

        # ----------------------------------------------------
        # Make one figure for each parameter
        # ----------------------------------------------------

        for pidx, parameter_name, parameter_tag in parameters:

            fig, ax = plt.subplots(
                figsize=(3.55, 3.55),
                dpi=300,
            )

            # ------------------------------------------------
            # Ideal calibration
            # ------------------------------------------------

            alpha = results[0]["alpha"]

            ax.plot(
                alpha,
                alpha,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Ideal",
            )

            # ------------------------------------------------
            # Plot each result
            # ------------------------------------------------

            for i, result in enumerate(results):
                ecp_marg = result["ecp_marg"][pidx]

                ecp_marg_std = result["ecp_marg_std"][pidx]

                ax.plot(
                    alpha,
                    ecp_marg,
                    color=colors[i],
                    linewidth=1.2,
                    label=labels[i],
                )

                ax.fill_between(
                    alpha,
                    ecp_marg - 2.0 * ecp_marg_std,
                    ecp_marg + 2.0 * ecp_marg_std,
                    color=colors[i],
                    alpha=0.08,
                )

            # ------------------------------------------------
            # Formatting
            # ------------------------------------------------

            format_axis(
                ax,
                f"{title}: {parameter_name}",
            )

            ax.legend(
                loc="upper left",
                fontsize=7,
                frameon=False,
            )

            fig.tight_layout()

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            png_file = os.path.join(
                imgdir,
                f"{name}_{parameter_tag}.png",
            )

            pdf_file = os.path.join(
                imgdir,
                f"{name}_{parameter_tag}.pdf",
            )

            fig.savefig(
                png_file,
                dpi=300,
                bbox_inches="tight",
            )

            fig.savefig(
                pdf_file,
                bbox_inches="tight",
            )

            print(f"Saved: {png_file}")
            print(f"Saved: {pdf_file}")

            plt.show()

            plt.close(fig)

    else:

        raise ValueError(
            f"Unknown mode: {mode}. "
            "Choose 'joint' or 'marginal'."
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Plot TARP calibration results."
    )

    parser.add_argument(
        "--tarpdirs",
        nargs="+",
        required=True,
        help="Directories containing TARP results.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for each TARP result.",
    )

    parser.add_argument(
        "--mode",
        choices=["joint", "marginal"],
        required=True,
        help="Plot joint or marginal TARP.",
    )

    parser.add_argument(
        "--imgdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Base name for output files.",
    )

    parser.add_argument(
        "--title",
        default="TARP",
        help="Plot title.",
    )

    args = parser.parse_args()

    plot_tarp(
        tarpdirs=args.tarpdirs,
        labels=args.labels,
        mode=args.mode,
        imgdir=args.imgdir,
        name=args.name,
        title=args.title,
    )


if __name__ == "__main__":
    main()