from __future__ import print_function

import csv
import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
#
# No command-line arguments are required.
# Running this file executes the entire Docker demonstration.
# ---------------------------------------------------------------------

DEMO_DEPTH = 3
DEMO_NUM_TRAIN = 100
DEMO_NUM_EVAL = 100

DEMO_WEIGHT_VALUES = [1.0, 3.0]
DEMO_BIAS_VALUES = [0.0, 1.0]

NONLINEARITIES = ["tanh", "relu", "erf"]

OUTPUT_DIR = "/nngp/output"

DEMO_CSV = os.path.join(
    OUTPUT_DIR,
    "demo_phase_grid.csv"
)

DEMO_FIGURE = os.path.join(
    OUTPUT_DIR,
    "demo_phase_diagram.png"
)

FINAL_FIGURE = os.path.join(
    OUTPUT_DIR,
    "phase_diagram_three_panel.png"
)

FULL_BASELINE_CSV = (
    "/nngp/data/phase_grid_mnist3k_10x10.csv"
)

FULL_ERF_CSV = (
    "/nngp/data/phase_grid_erf_mnist3k_10x10.csv"
)


CSV_FIELDS = [
    "nonlinearity",
    "depth",
    "weight_var",
    "bias_var",
    "train_acc",
    "valid_acc",
    "test_acc",
    "train_mse",
    "valid_mse",
    "test_mse",
    "stability_eps",
    "status"
]


# ---------------------------------------------------------------------
# Run one NNGP experiment
# ---------------------------------------------------------------------

def run_single_experiment(
        nonlinearity,
        depth,
        weight_var,
        bias_var):

    temp_dir = tempfile.mkdtemp(
        prefix="nngp_docker_demo_"
    )

    hparams = (
        "nonlinearity={0},depth={1},"
        "weight_var={2},bias_var={3}"
    ).format(
        nonlinearity,
        depth,
        weight_var,
        bias_var
    )

    command = [
        "python",
        "run_experiments.py",
        "--dataset=mnist",
        "--num_train={0}".format(
            DEMO_NUM_TRAIN
        ),
        "--num_eval={0}".format(
            DEMO_NUM_EVAL
        ),
        "--experiment_dir={0}".format(
            temp_dir
        ),
    ]

    # Tanh and ReLU use the repository's original
    # precomputed lookup grids.
    #
    # Erf is generated numerically at lower resolution
    # so the clean Docker test stays within memory.
    if nonlinearity == "erf":
        command.extend([
            "--n_gauss=101",
            "--n_var=101",
            "--n_corr=100"
        ])

    command.append(
        "--hparams={0}".format(hparams)
    )

    print("")
    print("=" * 70)

    print(
        "Running {0}: depth={1}, "
        "weight_var={2}, bias_var={3}"
        .format(
            nonlinearity,
            depth,
            weight_var,
            bias_var
        )
    )

    print("=" * 70)

    try:
        subprocess.check_call(command)

        result_file = os.path.join(
            temp_dir,
            "results.csv"
        )

        with open(result_file, "r") as f:
            rows = list(csv.reader(f))

        if len(rows) == 0:
            raise RuntimeError(
                "run_experiments.py produced no results."
            )

        row = rows[-1]

        result = {
            "nonlinearity": row[1],
            "depth": int(row[4]),
            "weight_var": float(row[2]),
            "bias_var": float(row[3]),
            "train_acc": float(row[5]),
            "valid_acc": float(row[6]),
            "test_acc": float(row[7]),
            "train_mse": float(row[8]),
            "valid_mse": float(row[9]),
            "test_mse": float(row[10]),
            "stability_eps": float(row[11]),
            "status": "success"
        }

    except subprocess.CalledProcessError:
        result = {
            "nonlinearity": nonlinearity,
            "depth": depth,
            "weight_var": weight_var,
            "bias_var": bias_var,
            "train_acc": "",
            "valid_acc": "",
            "test_acc": "",
            "train_mse": "",
            "valid_mse": "",
            "test_mse": "",
            "stability_eps": "",
            "status": "failed"
        }

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

    return result


# ---------------------------------------------------------------------
# Generate small demonstration dataset
# ---------------------------------------------------------------------

def generate_demo_data():

    print("")
    print("Generating small reproducibility dataset...")
    print("")

    results = []

    total = (
        len(NONLINEARITIES)
        * len(DEMO_WEIGHT_VALUES)
        * len(DEMO_BIAS_VALUES)
    )

    count = 0

    # Reduced-scale demonstration of the same sigma_w^2 / sigma_b^2
    # sweep used for Figure 4. The full project uses a 10 x 10 grid;
    # this smaller grid is run during Docker testing to verify the complete
    # experiment-to-CSV-to-heatmap pipeline on a clean clone.
    for nonlinearity in NONLINEARITIES:
        for bias_var in DEMO_BIAS_VALUES:
            for weight_var in DEMO_WEIGHT_VALUES:

                count += 1

                print("")
                print(
                    "Demo experiment {0}/{1}"
                    .format(count, total)
                )

                result = run_single_experiment(
                    nonlinearity,
                    DEMO_DEPTH,
                    weight_var,
                    bias_var
                )

                results.append(result)

                if result["status"] != "success":
                    raise RuntimeError(
                        "Demo experiment failed: "
                        "{0}, weight_var={1}, "
                        "bias_var={2}"
                        .format(
                            nonlinearity,
                            weight_var,
                            bias_var
                        )
                    )

    return results


# ---------------------------------------------------------------------
# Write demonstration CSV
# ---------------------------------------------------------------------

def write_demo_csv(results):

    with open(DEMO_CSV, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print("")
    print(
        "Demo CSV written to:"
    )
    print(DEMO_CSV)


# ---------------------------------------------------------------------
# Load CSV data
# ---------------------------------------------------------------------

def load_results(filenames):

    rows = []

    for filename in filenames:

        if not os.path.exists(filename):
            raise RuntimeError(
                "Required data file not found: "
                + filename
            )

        with open(filename, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                if row["status"] != "success":
                    continue

                if row["test_acc"] == "":
                    continue

                rows.append({
                    "nonlinearity":
                        row["nonlinearity"],
                    "depth":
                        int(row["depth"]),
                    "weight_var":
                        float(row["weight_var"]),
                    "bias_var":
                        float(row["bias_var"]),
                    "test_acc":
                        float(row["test_acc"])
                })

    return rows


# ---------------------------------------------------------------------
# Convert rows to heatmap matrix
# ---------------------------------------------------------------------

def make_accuracy_grid(
        rows,
        nonlinearity):

    selected = [
        row
        for row in rows
        if row["nonlinearity"]
        == nonlinearity
    ]

    if len(selected) == 0:
        raise RuntimeError(
            "No data found for "
            + nonlinearity
        )

    weight_values = sorted(
        set(
            row["weight_var"]
            for row in selected
        )
    )

    bias_values = sorted(
        set(
            row["bias_var"]
            for row in selected
        )
    )

    accuracy_grid = np.full(
        (
            len(bias_values),
            len(weight_values)
        ),
        np.nan
    )

    weight_index = {
        value: index
        for index, value
        in enumerate(weight_values)
    }

    bias_index = {
        value: index
        for index, value
        in enumerate(bias_values)
    }

    for row in selected:

        i = bias_index[
            row["bias_var"]
        ]

        j = weight_index[
            row["weight_var"]
        ]

        accuracy_grid[i, j] = (
            row["test_acc"]
        )

    return (
        np.array(weight_values),
        np.array(bias_values),
        accuracy_grid
    )


# ---------------------------------------------------------------------
# Plot three-panel heatmap
# ---------------------------------------------------------------------

def make_three_panel_figure(
        rows,
        output_file,
        title):

    nonlinearities = [
        "tanh",
        "relu",
        "erf"
    ]

    display_names = {
        "tanh": "Tanh",
        "relu": "ReLU",
        "erf": "Erf"
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        squeeze=False
    )

    axes = axes[0]

    image = None

    for ax, nonlinearity in zip(
            axes,
            nonlinearities):

        (
            weight_values,
            bias_values,
            accuracy_grid
        ) = make_accuracy_grid(
            rows,
            nonlinearity
        )

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
            vmin=0.10,
            vmax=0.96,
            interpolation="nearest"
        )

        ax.set_title(
            "MNIST-3k, d=50, {0}"
            .format(
                display_names[
                    nonlinearity
                ]
            )
        )

        ax.set_xlabel(
            r"$\sigma_w^2$"
        )

        ax.set_ylabel(
            r"$\sigma_b^2$"
        )

        ax.set_xticks(
            weight_values
        )

        ax.set_yticks(
            bias_values
        )

        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7
        )

        ax.tick_params(
            axis="y",
            labelsize=7
        )

        best_index = np.nanargmax(
            accuracy_grid
        )

        best_i, best_j = (
            np.unravel_index(
                best_index,
                accuracy_grid.shape
            )
        )

        best_weight = (
            weight_values[best_j]
        )

        best_bias = (
            bias_values[best_i]
        )

        ax.plot(
            best_weight,
            best_bias,
            marker="*",
            markersize=12,
            markeredgecolor="black",
            markerfacecolor="white"
        )

    colorbar = fig.colorbar(
        image,
        ax=axes.tolist(),
        fraction=0.03,
        pad=0.04
    )

    colorbar.set_label(
        "Test accuracy"
    )

    fig.suptitle(
        title,
        fontsize=14
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.18,
        top=0.84,
        wspace=0.30
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("")
    print(
        "Figure written to:"
    )
    print(output_file)


# ---------------------------------------------------------------------
# Main Docker workflow
# ---------------------------------------------------------------------

def main():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("")
    print("=" * 70)
    print("NNGP Docker reproducibility demonstration")
    print("=" * 70)

    # -------------------------------------------------------------
    # Part 1:
    # Generate a new, small dataset from the actual NNGP code.
    # -------------------------------------------------------------

    demo_results = generate_demo_data()

    write_demo_csv(
        demo_results
    )

    # -------------------------------------------------------------
    # Part 2:
    # Generate a small three-panel heatmap from freshly computed
    # demonstration data.
    # -------------------------------------------------------------

    make_three_panel_figure(
        demo_results,
        DEMO_FIGURE,
        "NNGP Docker Demonstration"
    )

    # -------------------------------------------------------------
    # Part 3:
    # Reproduce the final Figure 4-style three-panel heatmap from the
    # committed full MNIST-3k experiment data. Tanh and ReLU reproduce
    # the paper's nonlinearities; Erf is the project extension.
    # -------------------------------------------------------------

    full_results = load_results([
        FULL_BASELINE_CSV,
        FULL_ERF_CSV
    ])

    make_three_panel_figure(
        full_results,
        FINAL_FIGURE,
        "NNGP Test Accuracy Across Weight and Bias Variance"
    )

    print("")
    print("=" * 70)
    print("Docker demonstration completed successfully.")
    print("")
    print("Generated:")
    print("  " + DEMO_CSV)
    print("  " + DEMO_FIGURE)
    print("  " + FINAL_FIGURE)
    print("=" * 70)


if __name__ == "__main__":
    main()