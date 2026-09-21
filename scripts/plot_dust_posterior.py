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

from sbi_bmode import script_utils


opj = os.path.join

PARAM_INDICES = [0, 2, 3]


PARAM_LABELS = {
    "r_tensor": r"$r$",
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


def sample_posterior(posterior, data_obs, nsamp=10000):
    """
    Draw samples from a posterior.

    First tries rejection sampling. If that fails, falls back
    to MCMC sampling.
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
        print("Rejection sampling failed. Falling back to MCMC.")

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

    samples = np.asarray(samples, dtype=np.float64)

    return samples, used_mcmc


def load_posterior(path):
    """Load posterior.pkl."""
    print("Loading posterior:")
    print(f"  {path}")

    with open(path, "rb") as f:
        posterior = pickle.load(f)

    return posterior


def load_config(path):
    """Load YAML configuration."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


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

    prior, param_names = script_utils.get_prior(params_dict)

    return prior, param_names


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


def make_gradient_colors(ncase, colormap_name, low=0.08, high=0.92):
    """
    Build a list of hex colors sampled from a matplotlib colormap,
    one per posterior case, for a smooth gradient across cases.

    The `low`/`high` bounds avoid the very lightest/darkest ends of
    perceptually-uniform colormaps (e.g. viridis), which otherwise
    render as near-invisible contour outlines/fills.
    """
    cmap = plt.get_cmap(colormap_name)

    if ncase <= 1:
        sample_points = [0.5 * (low + high)]
    else:
        sample_points = np.linspace(low, high, ncase)

    colors = [mcolors.to_hex(cmap(x)) for x in sample_points]

    return colors


def main():

    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------
    # Posterior inputs
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Configs
    # ------------------------------------------------------------

    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config YAML for each posterior.",
    )

    # ------------------------------------------------------------
    # Test data / truth
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------

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
            "Inclusive range of test simulation indices to plot, "
            "e.g. --test-idx-range 1 200. "
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

    # ------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------

    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help=(
            "Matplotlib colormap name used to generate a gradient "
            "of colors across posterior cases (e.g. viridis, plasma, "
            "cividis, cool)."
        ),
    )

    parser.add_argument(
        "--colormap-range",
        type=float,
        nargs=2,
        default=[0.08, 0.92],
        metavar=("LOW", "HIGH"),
        help=(
            "Fractional range of the colormap to sample from, "
            "in [0, 1]. Avoids near-white/near-black extremes."
        ),
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output PNG filename. "
            "For multiple test indices, the index is appended."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------

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
            "Number of test-data files must equal number of posteriors."
        )

    if len(args.test_params) != ncase:
        raise ValueError(
            "Number of test-params files must equal number of posteriors."
        )

    if args.test_idx is not None and args.test_idx_range is not None:
        raise ValueError(
            "Specify either --test-idx or --test-idx-range, not both."
        )

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------
    # Build gradient colors, one per case
    # ------------------------------------------------------------

    if args.gradient:
        case_colors = make_gradient_colors(
            ncase,
            args.colormap,
            low=args.colormap_range[0],
            high=args.colormap_range[1],
        )
    else:
        case_colors = None

    print(f"\nUsing colormap: {args.colormap}")

    for label, color in zip(args.labels, case_colors):
        print(f"  {color}  {label}")

    # ------------------------------------------------------------
    # Get parameter information
    # ------------------------------------------------------------

    config0 = load_config(args.configs[0])

    prior_all, param_names_all = get_parameters(config0)

    # Select only parameters 0, 2, 3
    param_names = [
        param_names_all[i]
        for i in PARAM_INDICES
    ]

    # Select corresponding prior limits
    param_limits_all = script_utils.get_param_limits(
        prior_all,
        param_names_all,
    )

    param_limits = {
        param_names_all[i]: param_limits_all[param_names_all[i]]
        for i in PARAM_INDICES
    }

    # Get labels
    param_labels = [
        PARAM_LABELS[p]
        for p in param_names
    ]

    # GetDist wants labels without $
    param_labels_g = [
        label[1:-1]
        for label in param_labels
    ]

    print("\nParameters to plot:")

    for i, name in zip(PARAM_INDICES, param_names):
        print(f"  {i}: {name}")

    # ------------------------------------------------------------
    # Load all posteriors ONCE
    # ------------------------------------------------------------

    posteriors = []

    print("\n" + "=" * 70)
    print("LOADING POSTERIORS")
    print("=" * 70)

    for case_idx in range(ncase):

        label = args.labels[case_idx]

        print(f"\nCASE {case_idx}")
        print(f"Label: {label}")

        posterior = load_posterior(
            args.posteriors[case_idx]
        )

        posteriors.append(posterior)

    # ------------------------------------------------------------
    # Load all test data ONCE
    # ------------------------------------------------------------

    test_data = []
    test_params = []

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    for case_idx in range(ncase):

        label = args.labels[case_idx]

        data_all = np.load(
            args.test_data[case_idx]
        )

        params_all = np.load(
            args.test_params[case_idx]
        )

        print(f"\nCASE {case_idx}: {label}")

        print(
            f"Test data shape:   {data_all.shape}"
        )

        print(
            f"Test params shape: {params_all.shape}"
        )

        test_data.append(data_all)
        test_params.append(params_all)

    # ------------------------------------------------------------
    # Check number of test simulations
    # ------------------------------------------------------------

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
            "The test-data files have different numbers "
            f"of simulations: {ntest_data}"
        )

    if len(set(ntest_params)) != 1:
        raise ValueError(
            "The test-params files have different numbers "
            f"of simulations: {ntest_params}"
        )

    ntest = ntest_data[0]

    print("\n" + "=" * 70)
    print(f"NUMBER OF TEST SIMULATIONS: {ntest}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Determine test indices
    # ------------------------------------------------------------

    if args.test_idx_range is not None:

        range_start, range_end = args.test_idx_range

        if range_start < 0 or range_end >= ntest:
            raise ValueError(
                f"test-idx-range=({range_start}, {range_end}) is "
                f"outside valid range [0, {ntest - 1}]."
            )

        if range_start > range_end:
            raise ValueError(
                f"test-idx-range start ({range_start}) must be "
                f"<= end ({range_end})."
            )

        test_indices = range(range_start, range_end + 1)

        print(
            f"\nPlotting test simulations "
            f"{range_start} -> {range_end} (inclusive)"
        )

    elif args.test_idx is None:

        test_indices = range(ntest)

        print(
            f"\nPlotting ALL test simulations: "
            f"0 -> {ntest - 1}"
        )

    else:

        if args.test_idx < 0 or args.test_idx >= ntest:
            raise ValueError(
                f"test_idx={args.test_idx} is outside "
                f"valid range [0, {ntest - 1}]."
            )

        test_indices = [args.test_idx]

        print(
            f"\nPlotting only test simulation "
            f"{args.test_idx}"
        )

    # ------------------------------------------------------------
    # Loop over test simulations
    # ------------------------------------------------------------

    for test_idx in test_indices:

        print("\n\n")
        print("#" * 80)
        print(f"# TEST SIMULATION {test_idx}")
        print("#" * 80)

        # --------------------------------------------------------
        # Sample all posteriors for this test simulation
        # --------------------------------------------------------

        posterior_samples = []
        truths = []

        for case_idx in range(ncase):

            label = args.labels[case_idx]

            print("\n" + "=" * 70)
            print(f"CASE {case_idx}")
            print(f"Label: {label}")
            print(f"Test index: {test_idx}")
            print("=" * 70)

            # ----------------------------------------------------
            # Select one observation
            # ----------------------------------------------------

            data_obs = test_data[case_idx][test_idx]

            truth_all = test_params[case_idx][test_idx]

            # Select only parameters 0, 2, 3
            truth = truth_all[PARAM_INDICES]

            # ----------------------------------------------------
            # Sample posterior
            # ----------------------------------------------------

            print(
                f"Sampling {args.nsamp} posterior samples..."
            )

            samples, used_mcmc = sample_posterior(
                posteriors[case_idx],
                data_obs,
                nsamp=args.nsamp,
            )

            print(
                f"Posterior sample shape: {samples.shape}"
            )

            if used_mcmc:
                print("Sampling method: MCMC")
            else:
                print("Sampling method: rejection sampling")

            # ----------------------------------------------------
            # Select parameters 0, 2, 3
            # ----------------------------------------------------

            samples = samples[:, PARAM_INDICES]

            # ----------------------------------------------------
            # Check dimensions
            # ----------------------------------------------------

            if samples.shape[1] != len(param_names):
                raise ValueError(
                    f"{label}: posterior has "
                    f"{samples.shape[1]} selected parameters, "
                    f"but config defines {len(param_names)} "
                    f"parameters to plot."
                )

            # ----------------------------------------------------
            # Create GetDist object
            # ----------------------------------------------------

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

                print(
                    f"  {name:20s} = {value}"
                )

        # --------------------------------------------------------
        # Check whether all cases have the same truth
        # --------------------------------------------------------

        same_truth = all(
            np.allclose(truths[0], truth)
            for truth in truths[1:]
        )

        if same_truth:

            print(
                "\nAll cases have the same true parameters."
            )

            truth = truths[0]

        else:

            print(
                "\nWARNING: true parameters differ "
                "between cases."
            )

        # --------------------------------------------------------
        # Create triangle plot
        # --------------------------------------------------------

        print(
            f"\nCreating comparison plot for "
            f"test index {test_idx}..."
        )

        g = getdist_plots.get_subplot_plotter(
            width_inch=8,
        )

        g.settings.alpha_filled_add = 0.35
        g.settings.linewidth = 1.5
        g.settings.axes_fontsize = 12
        g.settings.lab_fontsize = 14
        g.settings.legend_fontsize = 12

        g.triangle_plot(
            posterior_samples,
            filled=True,
            legend_loc="upper right",
            contour_colors=case_colors,
            line_args=(
                [{"color": c} for c in case_colors]
                if case_colors is not None
                else None
            ),
        )

        # --------------------------------------------------------
        # Add true values
        # --------------------------------------------------------

        if same_truth:

            # ----------------------------------------------------
            # 1D panels: vertical truth line
            # ----------------------------------------------------

            for i, value in enumerate(truth):

                ax = g.subplots[i, i]

                ax.axvline(
                    value,
                    linestyle="--",
                    linewidth=1,
                    color="gray",
                )

            # ----------------------------------------------------
            # 2D panels: vertical + horizontal truth lines
            # ----------------------------------------------------

            nparam = len(param_names)

            for i in range(nparam):

                for j in range(i):

                    # GetDist convention:
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

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------

        output_path = make_output_path(
            args.output,
            test_idx,
        )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        print(
            f"\nSaving plot to:\n"
            f"{output_path}"
        )

        g.export(
            output_path,
            dpi=300,
        )

        plt.close(g.fig)

        print(
            f"Done: test index {test_idx}"
        )

    # ------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ALL POSTERIOR PLOTS FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()