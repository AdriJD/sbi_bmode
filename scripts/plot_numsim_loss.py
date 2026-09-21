#!/usr/bin/env python3

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Extract first loss from results.txt
# ============================================================
def extract_first_loss(optdir):
    """
    Read <optdir>/img/results.txt and extract the first
    loss value from the first data row.
    """

    results_file = os.path.join(
        optdir,
        "img",
        "results.txt",
    )

    if not os.path.isfile(results_file):
        raise FileNotFoundError(
            f"Could not find:\n  {results_file}"
        )

    with open(results_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip header
            if line.startswith("#"):
                continue

            # Split data row
            values = line.split()

            # Need at least idx + loss
            if len(values) < 2:
                continue

            # First column = idx
            # Second column = loss
            loss = float(values[1])

            return loss

    raise ValueError(
        f"No data rows found in:\n  {results_file}"
    )


# ============================================================
# Plot
# ============================================================
def plot_loss_vs_nsim(
    optdirs,
    nsims,
    imgdir,
    name,
    title,
):
    """
    Plot high-fidelity test-set loss versus number of
    high-fidelity simulations.
    """

    # --------------------------------------------------------
    # Check inputs
    # --------------------------------------------------------
    if len(optdirs) != len(nsims):
        raise ValueError(
            f"Number of optdirs ({len(optdirs)}) does not "
            f"match number of nsims ({len(nsims)})."
        )

    if len(optdirs) == 0:
        raise ValueError(
            "At least one Optuna directory is required."
        )

    # --------------------------------------------------------
    # Extract losses
    # --------------------------------------------------------
    losses = []

    for optdir, nsim in zip(optdirs, nsims):

        loss = extract_first_loss(optdir)
        losses.append(loss)

        print(
            f"N_HF = {nsim:>6}  "
            f"loss = {loss:.8e}"
        )

    losses = np.asarray(losses)
    nsims = np.asarray(nsims)

    # --------------------------------------------------------
    # Sort by number of simulations
    # --------------------------------------------------------
    sort_idx = np.argsort(nsims)

    nsims = nsims[sort_idx]
    losses = losses[sort_idx]

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------
    os.makedirs(
        imgdir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(5, 5),
        dpi=300,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    ax.plot(
        nsims,
        losses,
        marker="o",
        linestyle="-",
        lw=1.2,
        ms=4,
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------
    ax.set_xlabel(
        r"Number of high-fidelity simulations N"
    )

    ax.set_ylabel(
        "Loss on high-fidelity test set"
    )

    ax.set_title(
        title
    )

    ax.tick_params(
        "both",
        which="both",
        direction="in",
        right=True,
        top=True,
    )

    ax.grid(
        color="black",
        linestyle="dotted",
        linewidth=0.5,
    )

    ax.set_xscale("log")
    # --------------------------------------------------------
    # Linear x-axis
    # --------------------------------------------------------
    ax.set_xticks(nsims)
    ax.set_xticklabels(
        [str(n) for n in nsims],
        rotation=-45,
        ha="left",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    png_file = os.path.join(
        imgdir,
        name + ".png",
    )

    pdf_file = os.path.join(
        imgdir,
        name + ".pdf",
    )

    fig.savefig(
        png_file,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    print()
    print("Saved:")
    print(f"  {png_file}")
    print(f"  {pdf_file}")
    print()

    plt.show()


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot high-fidelity test-set loss versus "
            "number of high-fidelity simulations."
        )
    )

    parser.add_argument(
        "--optdirs",
        nargs="+",
        required=True,
        help=(
            "Optuna directories. Each directory must contain "
            "img/results.txt."
        ),
    )

    parser.add_argument(
        "--nsims",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Number of high-fidelity simulations corresponding "
            "to each Optuna directory."
        ),
    )

    parser.add_argument(
        "--imgdir",
        required=True,
        help="Directory where the plot is saved.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Output filename without extension.",
    )

    parser.add_argument(
        "--title",
        default="High-fidelity test-set loss",
        help="Plot title.",
    )

    args = parser.parse_args()

    plot_loss_vs_nsim(
        optdirs=args.optdirs,
        nsims=args.nsims,
        imgdir=args.imgdir,
        name=args.name,
        title=args.title,
    )


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()