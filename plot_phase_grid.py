from __future__ import print_function

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot NNGP Figure 4-style accuracy heatmaps."
    )

    parser.add_argument(
        "--input_files",
        required=True,
        help=(
            "Comma-separated CSV files produced by phase_grid.py. "
            "Example: file1.csv,file2.csv"
        )
    )

    parser.add_argument(
        "--nonlinearities",
        default="tanh,relu,erf",
        help="Comma-separated nonlinearities to plot."
    )

    parser.add_argument(
        "--output_file",
        default="/nngp/output/phase_diagram.png",
        help="Path for the output figure."
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=0.1,
        help="Minimum value of the heatmap color scale."
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=0.96,
        help="Maximum value of the heatmap color scale."
    )

    return parser.parse_args()


def load_results(input_files):
    rows = []

    for filename in input_files:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row["status"] != "success":
                    continue

                # Ignore incomplete rows.
                if row["test_acc"] == "":
                    continue

                rows.append({
                    "nonlinearity": row["nonlinearity"],
                    "depth": int(row["depth"]),
                    "weight_var": float(row["weight_var"]),
                    "bias_var": float(row["bias_var"]),
                    "test_acc": float(row["test_acc"])
                })

    return rows

# Arrange test accuracies by sigma_b^2 and sigma_w^2 so each matrix
# entry corresponds to one evaluated point in the Figure 4 phase grid.
def make_accuracy_grid(rows, nonlinearity):
    selected = [
        row for row in rows
        if row["nonlinearity"] == nonlinearity
    ]

    if len(selected) == 0:
        return None, None, None

    weight_values = sorted(
        set(row["weight_var"] for row in selected)
    )

    bias_values = sorted(
        set(row["bias_var"] for row in selected)
    )

    accuracy_grid = np.full(
        (len(bias_values), len(weight_values)),
        np.nan
    )

    weight_index = {
        value: i
        for i, value in enumerate(weight_values)
    }

    bias_index = {
        value: i
        for i, value in enumerate(bias_values)
    }

    for row in selected:
        i = bias_index[row["bias_var"]]
        j = weight_index[row["weight_var"]]

        accuracy_grid[i, j] = row["test_acc"]

    return (
        np.array(weight_values),
        np.array(bias_values),
        accuracy_grid
    )


def main():
    args = parse_args()

    input_files = [
        x.strip()
        for x in args.input_files.split(",")
        if x.strip()
    ]

    nonlinearities = [
        x.strip()
        for x in args.nonlinearities.split(",")
        if x.strip()
    ]

    rows = load_results(input_files)

    # Only plot nonlinearities that currently have data.
    available = []

    for nonlinearity in nonlinearities:
        weight_values, bias_values, accuracy_grid = (
            make_accuracy_grid(rows, nonlinearity)
        )

        if accuracy_grid is not None:
            available.append(
                (
                    nonlinearity,
                    weight_values,
                    bias_values,
                    accuracy_grid
                )
            )

    if len(available) == 0:
        raise RuntimeError("No successful results were found.")

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5 * len(available), 4.5),
        squeeze=False
    )

    axes = axes[0]

    image = None

    for ax, data in zip(axes, available):
        (
            nonlinearity,
            weight_values,
            bias_values,
            accuracy_grid
        ) = data

        # Plot the empirical accuracy phase diagram in the same form as
        # Figure 4: sigma_w^2 on the horizontal axis, sigma_b^2 on the
        # vertical axis, and classification accuracy encoded by color.
        image = ax.imshow(
            accuracy_grid,
            origin="lower",
            aspect="auto",
            extent=[
                weight_values.min(),
                weight_values.max(),
                bias_values.min(),
                bias_values.max()
            ],
            vmin=args.vmin,
            vmax=args.vmax,
            interpolation="nearest"
        )

        display_names = {
            "tanh": "Tanh",
            "relu": "ReLU",
            "erf": "Erf"
        }

        ax.set_title(
            "MNIST-3k, d=50, {0}".format(
                display_names.get(nonlinearity, nonlinearity)
            )
        )

        ax.set_xlabel(r"$\sigma_w^2$")
        ax.set_ylabel(r"$\sigma_b^2$")      

        ax.set_xticks(weight_values)
        ax.set_yticks(bias_values)

        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7
        )

        ax.tick_params(
            axis="y",
            labelsize=7
        )

        # Mark the highest-accuracy grid point.
        if not np.all(np.isnan(accuracy_grid)):
            best_index = np.nanargmax(accuracy_grid)

            best_i, best_j = np.unravel_index(
                best_index,
                accuracy_grid.shape
            )

            best_weight = weight_values[best_j]
            best_bias = bias_values[best_i]
            best_acc = accuracy_grid[best_i, best_j]

            ax.plot(
                best_weight,
                best_bias,
                marker="*",
                markersize=12,
                markeredgecolor="black",
                markerfacecolor="white"
            )

            print(
                "{0}: best test accuracy = {1:.4f}, "
                "weight_var = {2:.4f}, bias_var = {3:.4f}"
                .format(
                    nonlinearity,
                    best_acc,
                    best_weight,
                    best_bias
                )
            )

    # One shared colorbar is important because it ensures that
    # all nonlinearities are being viewed on exactly the same scale.
    colorbar = fig.colorbar(
        image,
        ax=axes.tolist(),
        fraction=0.03,
        pad=0.04
    )

    colorbar.set_label("Test accuracy")

    fig.suptitle(
        "NNGP Test Accuracy Across Weight and Bias Variance",
        fontsize=14
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.18,
        top=0.84,
        wspace=0.30
    )

    output_dir = os.path.dirname(args.output_file)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        args.output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print("")
    print("Figure saved to:")
    print(args.output_file)


if __name__ == "__main__":
    main()