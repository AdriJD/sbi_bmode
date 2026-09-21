import os
import argparse
import pickle

import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from getdist import plots as getdist_plots
from getdist import MCSamples
from mpi4py import MPI

from sbi_bmode import script_utils


opj = os.path.join

comm = MPI.COMM_WORLD


# ============================================================
# Parameter labels
# ============================================================

PARAM_LABELS = {
    "r_tensor": r"$r$",
    "A_lens": r"$A_{\mathrm{lens}}$",
    "A_d_BB": r"$A_{\mathrm{d}}$",
    "alpha_d_BB": r"$\alpha_{\mathrm{d}}$",
    "beta_dust": r"$\beta_{\mathrm{d}}$",
    "amp_beta_dust": r"$B_{\mathrm{d}}$",
    "gamma_beta_dust": r"$\gamma_{\mathrm{d}}$",
    "A_s_BB": r"$A_{\mathrm{s}}$",
    "alpha_s_BB": r"$\alpha_{\mathrm{s}}$",
    "beta_sync": r"$\beta_{\mathrm{s}}$",
    "amp_beta_sync": r"$B_{\mathrm{s}}$",
    "gamma_beta_sync": r"$\gamma_{\mathrm{s}}$",
    "rho_ds": r"$\rho_{\mathrm{ds}}$",
}


# ============================================================
# Draw samples from prior
# ============================================================

def draw_from_prior(prior_list, nsamp):
    """
    Draw samples from the prior distributions.

    Parameters
    ----------
    prior_list : list
        List of torch distributions.

    nsamp : int
        Number of prior samples.

    Returns
    -------
    out : ndarray
        Array with shape (nsamp, n_parameters).
    """

    out = np.zeros((nsamp, len(prior_list)))

    for pidx, prior in enumerate(prior_list):

        out[:, pidx] = np.asarray(
            prior.sample((nsamp,))
        )[:, 0]

    return out


# ============================================================
# Sample posterior
# ============================================================

def sample_posterior(posterior, data_obs, nsamp=10000):
    """
    Draw samples from a posterior.

    First tries rejection sampling.
    If that fails, falls back to MCMC sampling.
    """

    try:

        samples = posterior.sample(
            (nsamp,),
            x=data_obs,
            show_progress_bars=False,
            max_sampling_time=8.0,
        )

        if samples.shape[0] < nsamp:
            raise RuntimeError(
                "Rejection sampling returned too few samples."
            )

        used_mcmc = False

    except RuntimeError:

        print(
            "Rejection sampling failed. "
            "Falling back to MCMC."
        )

        used_mcmc = True

        with torch.no_grad():

            samples = posterior.sample(
                (nsamp,),
                x=data_obs,
                show_progress_bars=False,
                reject_outside_prior=False,
            )

    if isinstance(samples, torch.Tensor):

        samples = samples.detach().cpu().numpy()

    samples = np.asarray(
        samples,
        dtype=np.float64,
    )

    return samples, used_mcmc


# ============================================================
# Load posterior
# ============================================================

def load_posterior(path):
    """
    Load posterior.pkl.
    """

    print("Loading posterior:")
    print(f"  {path}")

    with open(path, "rb") as f:
        posterior = pickle.load(f)

    return posterior


# ============================================================
# Load config
# ============================================================

def load_config(path):
    """
    Load YAML configuration.
    """

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


# ============================================================
# Get parameters and priors
# ============================================================

def get_parameters(config):
    """
    Get all parameter names and prior information.
    """

    (
        data_dict,
        fixed_params_dict,
        params_dict,
        observation_dict,
        transfer_dict,
    ) = script_utils.parse_config(config)

    prior, param_names = script_utils.get_prior(
        params_dict
    )

    return prior, param_names


# ============================================================
# Output path
# ============================================================

def make_output_path(output, test_idx):
    """
    Create output filename for a given test index.

    Example:
        test.png -> test_002.png
    """

    root, ext = os.path.splitext(output)

    if ext == "":
        ext = ".png"

    return f"{root}_{test_idx:03d}{ext}"


# ============================================================
# Gradient colors
# ============================================================

def make_gradient_colors(
    ncase,
    colormap_name,
    low=0.08,
    high=0.92,
):
    """
    Build a list of colors sampled from a Matplotlib
    colormap, one per posterior case.
    """

    cmap = plt.get_cmap(colormap_name)

    if ncase <= 1:

        sample_points = [
            0.5 * (low + high)
        ]

    else:

        sample_points = np.linspace(
            low,
            high,
            ncase,
        )

    colors = [
        mcolors.to_hex(cmap(x))
        for x in sample_points
    ]

    return colors


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    # --------------------------------------------------------
    # Posterior inputs
    # --------------------------------------------------------

    parser.add_argument(
        "--posteriors",
        nargs="+",
        required=True,
        help="Paths to posterior.pkl files.",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for the posterior cases.",
    )


    # --------------------------------------------------------
    # Configs
    # --------------------------------------------------------

    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config YAML for each posterior.",
    )


    # --------------------------------------------------------
    # Test data / truth
    # --------------------------------------------------------

    parser.add_argument(
        "--test-data",
        nargs="+",
        required=True,
        help="Test data .npy file for each posterior.",
    )

    parser.add_argument(
        "--test-params",
        nargs="+",
        required=True,
        help="Test parameter .npy file for each posterior.",
    )


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    parser.add_argument(
        "--test-idx",
        type=int,
        default=None,
        help=(
            "Index of test simulation. "
            "If omitted, plot all test simulations."
        ),
    )

    parser.add_argument(
        "--test-idx-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help=(
            "Inclusive range of test simulation indices "
            "to plot, e.g. --test-idx-range 1 200. "
            "Mutually exclusive with --test-idx."
        ),
    )

    parser.add_argument(
        "--nsamp",
        type=int,
        default=10000,
        help="Number of posterior samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20,
        help="Random seed.",
    )


    # --------------------------------------------------------
    # Parameter selection
    # --------------------------------------------------------

    parser.add_argument(
        "--cosmo-only",
        action="store_true",
        help="Plot only r and A_lens.",
    )

    parser.add_argument(
        "--param-idx",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Indices (0-based, into the full parameter list "
            "defined by the config) of parameters to plot, "
            "e.g. --param-idx 0 2 3. "
            "Overrides --cosmo-only."
        ),
    )


    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Use a color gradient across posterior cases.",
    )

    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help=(
            "Matplotlib colormap name used to generate "
            "a gradient of colors across posterior cases."
        ),
    )

    parser.add_argument(
        "--colormap-range",
        type=float,
        nargs=2,
        default=[0.08, 0.92],
        metavar=("LOW", "HIGH"),
        help=(
            "Fractional range of the colormap to sample "
            "from, in [0, 1]."
        ),
    )


    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    parser.add_argument(
        "--no-prior",
        action="store_true",
        help="Do not plot the prior.",
    )

    parser.add_argument(
        "--prior-samples",
        type=int,
        default=500000,
        help="Number of samples used to draw the prior.",
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output PNG filename. "
            "For multiple test indices, the index is appended."
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # Check inputs
    # ========================================================
    # Runs identically (and fails identically) on every rank.

    ncase = len(args.posteriors)

    if len(args.labels) != ncase:
        raise ValueError(
            "Number of labels must equal number of posteriors."
        )

    if len(args.configs) != ncase:
        raise ValueError(
            "Number of configs must equal number of posteriors."
        )

    if len(args.test_data) != ncase:
        raise ValueError(
            "Number of test-data files must equal "
            "number of posteriors."
        )

    if len(args.test_params) != ncase:
        raise ValueError(
            "Number of test-params files must equal "
            "number of posteriors."
        )

    if (
        args.test_idx is not None
        and args.test_idx_range is not None
    ):
        raise ValueError(
            "Specify either --test-idx or "
            "--test-idx-range, not both."
        )


    # ========================================================
    # Seed
    # ========================================================
    # Every rank uses the same seed. Ranks only ever operate on
    # disjoint test indices, so this does not create duplicate
    # posterior samples anywhere -- it just keeps the redundant
    # setup below (prior draws, color assignment) identical
    # bit-for-bit across ranks.

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if comm.rank == 0:
        print(f"\nRunning with {comm.size} MPI rank(s)")


    # ========================================================
    # Build colors
    # ========================================================

    if args.gradient:

        case_colors = make_gradient_colors(
            ncase,
            args.colormap,
            low=args.colormap_range[0],
            high=args.colormap_range[1],
        )

        if comm.rank == 0:
            print(
                f"\nUsing gradient colormap: "
                f"{args.colormap}"
            )

    else:

        # Explicit standard colors
        #
        # Case 0 -> blue
        # Case 1 -> red
        # Case 2 -> green
        # Case 3 -> purple
        # Case 4 -> orange
        # ...

        standard_colors = [
            "cornflowerblue",
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        case_colors = [
            standard_colors[i % len(standard_colors)]
            for i in range(ncase)
        ]

        if comm.rank == 0:
            print("\nUsing standard colors.")

    if comm.rank == 0:
        for label, color in zip(args.labels, case_colors):
            print(f"  {color}  {label}")


    # ========================================================
    # Get parameter information
    # ========================================================
    # Config parsing and prior construction are cheap, so every
    # rank just does this itself rather than broadcasting.

    config0 = load_config(
        args.configs[0]
    )

    # Always get FULL parameter set first.

    full_prior, full_param_names = get_parameters(
        config0
    )

    full_param_limits = (
        script_utils.get_param_limits(
            full_prior,
            full_param_names,
        )
    )

    nparam_full = len(
        full_param_names
    )


    # ========================================================
    # Determine parameters to plot
    # ========================================================

    if args.param_idx is not None:

        for idx in args.param_idx:

            if (
                idx < 0
                or idx >= nparam_full
            ):
                raise ValueError(
                    f"--param-idx {idx} is out of range "
                    f"[0, {nparam_full - 1}]."
                )

        selected_idx = args.param_idx

    elif args.cosmo_only:

        selected_idx = [0, 1]

    else:

        selected_idx = list(
            range(nparam_full)
        )


    param_names = [
        full_param_names[i]
        for i in selected_idx
    ]

    param_labels = [
        PARAM_LABELS[p]
        for p in param_names
    ]

    # GetDist wants labels without $

    param_labels_g = [
        label[1:-1]
        for label in param_labels
    ]


    # --------------------------------------------------------
    # Parameter limits for selected parameters
    # --------------------------------------------------------

    param_limits = full_param_limits


    if comm.rank == 0:
        print("\nParameters to plot:")
        for i, name in zip(selected_idx, param_names):
            print(f"  {i}: {name}")


    # ========================================================
    # Create prior samples
    # ========================================================
    # Deterministic given the seed set above, so every rank
    # draws the same prior samples independently rather than
    # broadcasting a ~500k-sample array.

    prior_samples = None

    if not args.no_prior:

        if comm.rank == 0:
            print("\n" + "=" * 70)
            print("GENERATING PRIOR SAMPLES")
            print("=" * 70)

        # Select only the priors corresponding
        # to the parameters being plotted.

        selected_prior = [
            full_prior[i]
            for i in selected_idx
        ]

        prior_draw = draw_from_prior(
            selected_prior,
            args.prior_samples,
        )

        prior_samples = MCSamples(
            samples=prior_draw,
            names=param_names,
            labels=param_labels_g,
            ranges=param_limits,
            label="prior",
        )

        # Same smoothing as your original code.

        prior_samples.smooth_scale_1D = 0.2

        if comm.rank == 0:
            print(f"Prior samples: {prior_draw.shape}")


    # ========================================================
    # Load all posteriors ONCE (rank 0), then broadcast
    # ========================================================
    # Posterior pickles can be sizeable, so only rank 0 touches
    # the filesystem for these; every other rank gets them via
    # comm.bcast, same pattern as sample_test_set.py.

    if comm.rank == 0:

        print("\n" + "=" * 70)
        print("LOADING POSTERIORS")
        print("=" * 70)

        posteriors = []

        for case_idx in range(ncase):

            label = args.labels[case_idx]
            print(f"\nCASE {case_idx}")
            print(f"Label: {label}")

            posterior = load_posterior(
                args.posteriors[case_idx]
            )

            posteriors.append(posterior)

    else:

        posteriors = None

    posteriors = comm.bcast(posteriors, root=0)


    # ========================================================
    # Load all test data ONCE (rank 0), then broadcast
    # ========================================================

    if comm.rank == 0:

        print("\n" + "=" * 70)
        print("LOADING TEST DATA")
        print("=" * 70)

        test_data = []
        test_params = []

        for case_idx in range(ncase):

            label = args.labels[case_idx]

            data_all = np.load(
                args.test_data[case_idx]
            )

            params_all = np.load(
                args.test_params[case_idx]
            )

            print(f"\nCASE {case_idx}: {label}")
            print(f"Test data shape:   {data_all.shape}")
            print(f"Test params shape: {params_all.shape}")

            test_data.append(data_all)
            test_params.append(params_all)

    else:

        test_data = None
        test_params = None

    test_data = comm.bcast(test_data, root=0)
    test_params = comm.bcast(test_params, root=0)


    # ========================================================
    # Check number of test simulations
    # ========================================================

    ntest_data = [
        data.shape[0]
        for data in test_data
    ]

    ntest_params = [
        params.shape[0]
        for params in test_params
    ]

    if len(set(ntest_data)) != 1:

        raise ValueError(
            "The test-data files have different "
            f"numbers of simulations: {ntest_data}"
        )

    if len(set(ntest_params)) != 1:

        raise ValueError(
            "The test-params files have different "
            f"numbers of simulations: {ntest_params}"
        )

    ntest = ntest_data[0]

    if comm.rank == 0:
        print("\n" + "=" * 70)
        print(f"NUMBER OF TEST SIMULATIONS: {ntest}")
        print("=" * 70)


    # ========================================================
    # Determine test indices
    # ========================================================

    if args.test_idx_range is not None:

        range_start, range_end = (
            args.test_idx_range
        )

        if (
            range_start < 0
            or range_end >= ntest
        ):

            raise ValueError(
                f"test-idx-range=({range_start}, "
                f"{range_end}) is outside valid range "
                f"[0, {ntest - 1}]."
            )

        if range_start > range_end:

            raise ValueError(
                f"test-idx-range start "
                f"({range_start}) must be <= "
                f"end ({range_end})."
            )

        test_indices = list(range(range_start, range_end + 1))

        if comm.rank == 0:
            print(
                f"\nPlotting test simulations "
                f"{range_start} -> {range_end} "
                f"(inclusive)"
            )

    elif args.test_idx is None:

        test_indices = list(range(ntest))

        if comm.rank == 0:
            print(
                f"\nPlotting ALL test simulations: "
                f"0 -> {ntest - 1}"
            )

    else:

        if (
            args.test_idx < 0
            or args.test_idx >= ntest
        ):

            raise ValueError(
                f"test_idx={args.test_idx} is outside "
                f"valid range [0, {ntest - 1}]."
            )

        test_indices = [args.test_idx]

        if comm.rank == 0:
            print(
                f"\nPlotting only test simulation "
                f"{args.test_idx}"
            )


    # ========================================================
    # Split test indices across MPI ranks
    # ========================================================
    # Each test index produces its own, independent output PNG
    # (via make_output_path), so unlike sample_test_set.py there
    # is nothing to Gatherv at the end -- ranks just work through
    # disjoint slices of test_indices and each writes its own
    # files.

    idxs_per_rank = np.array_split(
        np.array(test_indices), comm.size
    )
    indices_on_rank = [int(i) for i in idxs_per_rank[comm.rank]]

    if comm.rank == 0:
        print(
            f"\nDistributing {len(test_indices)} test "
            f"simulation(s) across {comm.size} rank(s)"
        )

    print(
        f"[rank {comm.rank}] assigned "
        f"{len(indices_on_rank)} test simulation(s): "
        f"{indices_on_rank}"
    )


    # ========================================================
    # Loop over test simulations assigned to this rank
    # ========================================================

    for test_idx in indices_on_rank:

        print("\n\n")
        print("#" * 80)
        print(f"# [rank {comm.rank}] TEST SIMULATION {test_idx}")
        print("#" * 80)


        # ----------------------------------------------------
        # Sample all posteriors
        # ----------------------------------------------------

        posterior_samples = []
        truths = []

        for case_idx in range(ncase):

            label = args.labels[case_idx]

            print("\n" + "=" * 70)
            print(f"[rank {comm.rank}] CASE {case_idx}")
            print(f"Label: {label}")
            print(f"Test index: {test_idx}")
            print("=" * 70)


            # ------------------------------------------------
            # Select observation
            # ------------------------------------------------

            data_obs = (
                test_data[case_idx][test_idx]
            )

            truth = (
                test_params[case_idx][test_idx]
            )

            truth = truth[
                selected_idx
            ]


            # ------------------------------------------------
            # Sample posterior
            # ------------------------------------------------

            print(
                f"[rank {comm.rank}] Sampling {args.nsamp} "
                f"posterior samples..."
            )

            samples, used_mcmc = (
                sample_posterior(
                    posteriors[case_idx],
                    data_obs,
                    nsamp=args.nsamp,
                )
            )

            print(
                f"Posterior sample shape: "
                f"{samples.shape}"
            )

            if used_mcmc:
                print("Sampling method: MCMC")
            else:
                print("Sampling method: rejection sampling")


            # ------------------------------------------------
            # Select requested parameters
            # ------------------------------------------------

            samples = samples[
                :,
                selected_idx
            ]


            # ------------------------------------------------
            # Check dimensions
            # ------------------------------------------------

            if (
                samples.shape[1]
                != len(param_names)
            ):

                raise ValueError(
                    f"{label}: posterior has "
                    f"{samples.shape[1]} selected "
                    f"parameters, but "
                    f"{len(param_names)} parameters "
                    f"were requested to plot."
                )


            # ------------------------------------------------
            # Create GetDist object
            # ------------------------------------------------

            gd_samples = MCSamples(
                samples=samples,
                names=param_names,
                labels=param_labels_g,
                ranges=param_limits,
                label=label,
            )

            posterior_samples.append(gd_samples)
            truths.append(truth)

            print("\nTruth:")
            for name, value in zip(param_names, truth):
                print(f"  {name:20s} = {value}")


        # ====================================================
        # Check whether all cases have same truth
        # ====================================================

        same_truth = all(
            np.allclose(
                truths[0],
                truth,
            )
            for truth in truths[1:]
        )

        if same_truth:

            print(
                "\nAll cases have the same "
                "true parameters."
            )

            truth = truths[0]

        else:

            print(
                "\nWARNING: true parameters "
                "differ between cases."
            )


        # ====================================================
        # Create triangle plot
        # ====================================================

        print(
            f"\n[rank {comm.rank}] Creating comparison plot "
            f"for test index {test_idx}..."
        )

        g = getdist_plots.get_subplot_plotter(
            width_inch=8,
        )

        g.settings.alpha_filled_add = 0.35
        g.settings.linewidth = 1.5
        g.settings.axes_fontsize = 12
        g.settings.lab_fontsize = 14
        g.settings.legend_fontsize = 14


        # ----------------------------------------------------
        # Posterior triangle plot
        # ----------------------------------------------------

        g.triangle_plot(
            posterior_samples,
            filled=True,
            legend_loc="upper right",
            contour_colors=case_colors,
            line_args=[
                {"color": c}
                for c in case_colors
            ],
        )


        # ====================================================
        # Add prior
        # ====================================================

        if prior_samples is not None:

            print("Adding prior to 1D marginals...")

            for i, name in enumerate(param_names):

                g.add_1d(
                    prior_samples,
                    param=name,
                    ls="--",
                    color="gray",
                    label="prior",
                    ax=g.subplots[i, i],
                )


        # ====================================================
        # Add true values
        # ====================================================

        if same_truth:

            # ------------------------------------------------
            # 1D panels
            # ------------------------------------------------

            for i, value in enumerate(truth):

                ax = g.subplots[i, i]

                ax.axvline(
                    value,
                    linestyle="--",
                    linewidth=1,
                    color="gray",
                )


            # ------------------------------------------------
            # 2D panels
            # ------------------------------------------------

            nparam = len(param_names)

            for i in range(nparam):

                for j in range(i):

                    # GetDist convention:
                    #
                    # x-axis -> parameter j
                    # y-axis -> parameter i

                    ax = g.subplots[i, j]

                    # True x value

                    ax.axvline(
                        truth[j],
                        linestyle="--",
                        linewidth=1,
                        color="gray",
                    )

                    # True y value

                    ax.axhline(
                        truth[i],
                        linestyle="--",
                        linewidth=1,
                        color="gray",
                    )


        # ====================================================
        # Save
        # ====================================================

        output_path = make_output_path(
            args.output,
            test_idx,
        )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(
            f"\n[rank {comm.rank}] Saving plot to:\n"
            f"{output_path}"
        )

        g.export(output_path, dpi=300)

        plt.close(g.fig)

        print(
            f"[rank {comm.rank}] Done: test index {test_idx}"
        )


    # ========================================================
    # Finished
    # ========================================================

    comm.Barrier()

    if comm.rank == 0:
        print("\n" + "=" * 80)
        print("ALL POSTERIOR PLOTS FINISHED")
        print("=" * 80)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()