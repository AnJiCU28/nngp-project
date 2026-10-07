from __future__ import print_function

import argparse
import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NNGP accuracy-grid data for Figure 4."
    )

    parser.add_argument(
        "--nonlinearities",
        default="tanh,relu",
        help="Comma-separated nonlinearities to evaluate."
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=50,
        help="NNGP kernel depth."
    )

    parser.add_argument(
        "--num_train",
        type=int,
        default=5000,
        help="Number of MNIST training examples."
    )

    parser.add_argument(
        "--num_eval",
        type=int,
        default=1000,
        help="Number of validation/test examples to evaluate."
    )

    parser.add_argument(
        "--grid_size",
        type=int,
        default=5,
        help="Number of values along each weight/bias axis."
    )

    parser.add_argument(
        "--weight_min",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--weight_max",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--bias_min",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--bias_max",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--output_file",
        default="/nngp/output/phase_grid_results.csv",
        help="CSV file in which to save results."
    )

    parser.add_argument(
    "--n_gauss",
    type=int,
    default=501,
    help="Number of Gaussian integration points."
)

    parser.add_argument(
    "--n_var",
    type=int,
    default=501,
    help="Number of variance-grid points."
)

    parser.add_argument(
    "--n_corr",
    type=int,
    default=500,
    help="Number of correlation-grid points."
)

    return parser.parse_args()


def run_single_experiment(
        nonlinearity,
        depth,
        weight_var,
        bias_var,
        num_train,
        num_eval,
        n_gauss,
        n_var,
        n_corr):

    # Use a temporary experiment directory because run_experiments.py
    # writes its own results.csv file.
    temp_dir = tempfile.mkdtemp(prefix="nngp_phase_")

    hparams = (
        "nonlinearity={0},depth={1},weight_var={2},bias_var={3}"
        .format(
            nonlinearity,
            depth,
            weight_var,
            bias_var
        )
    )

    command = [
        "python",
        "run_experiments.py",
        "--dataset=mnist",
        "--num_train={0}".format(num_train),
        "--num_eval={0}".format(num_eval),
        "--n_gauss={0}".format(n_gauss),
        "--n_var={0}".format(n_var),
        "--n_corr={0}".format(n_corr),
        "--experiment_dir={0}".format(temp_dir),
        "--hparams={0}".format(hparams)
    ]

    print("")
    print("=" * 70)
    print(
        "Running: nonlinearity={0}, depth={1}, "
        "weight_var={2:.4f}, bias_var={3:.4f}"
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

        result_file = os.path.join(temp_dir, "results.csv")

        with open(result_file, "r") as f:
            rows = list(csv.reader(f))

        if len(rows) == 0:
            raise RuntimeError("run_experiments.py produced no result row.")

        row = rows[-1]

        # run_experiments.py stores:
        #
        # 0  num_train
        # 1  nonlinearity
        # 2  weight_var
        # 3  bias_var
        # 4  depth
        # 5  train_acc
        # 6  valid_acc
        # 7  test_acc
        # 8  train_mse
        # 9  valid_mse
        # 10 test_mse
        # 11 stability_eps

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
        print("Experiment FAILED.")

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
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result


def append_result(output_file, result):
    output_dir = os.path.dirname(output_file)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_exists = os.path.exists(output_file)

    fieldnames = [
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

    with open(output_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

def load_completed_runs(output_file):
    completed = set()

    if not os.path.exists(output_file):
        return completed

    with open(output_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["status"] == "success":
                key = (
                    row["nonlinearity"],
                    int(row["depth"]),
                    round(float(row["weight_var"]), 10),
                    round(float(row["bias_var"]), 10)
                )
                completed.add(key)

    return completed


def main():
    args = parse_args()

    nonlinearities = [
        x.strip()
        for x in args.nonlinearities.split(",")
        if x.strip()
    ]

    # Appendix D of the paper used 30 evenly spaced values from
    # 0.1 to 5.0 for sigma_w^2 and 30 values from 0 to 2.0
    # for sigma_b^2. grid_size allows us to test a smaller version first.
    weight_values = np.linspace(
        args.weight_min,
        args.weight_max,
        args.grid_size
    )

    bias_values = np.linspace(
        args.bias_min,
        args.bias_max,
        args.grid_size
    )

    total_runs = (
        len(nonlinearities)
        * len(weight_values)
        * len(bias_values)
    )

    print("NNGP Figure 4 data generation")
    print("Nonlinearities:", nonlinearities)
    print("Depth:", args.depth)
    print("Training examples:", args.num_train)
    print("Evaluation examples:", args.num_eval)
    print("Grid size:", args.grid_size, "x", args.grid_size)
    print("Total experiments:", total_runs)
    print("Output:", args.output_file)

    run_number = 0

    completed = load_completed_runs(args.output_file)

    print("Previously completed experiments:", len(completed))

    for nonlinearity in nonlinearities:
        for bias_var in bias_values:
            for weight_var in weight_values:

                key = (
                    nonlinearity,
                    args.depth,
                    round(float(weight_var), 10),
                    round(float(bias_var), 10)
                )

                if key in completed:
                    print(
                        "Skipping completed run: "
                        "nonlinearity={0}, weight_var={1:.4f}, bias_var={2:.4f}"
                        .format(
                            nonlinearity,
                            weight_var,
                            bias_var
                        )
                    )
                    continue

                run_number += 1

                print("")
                print(
                    "Experiment {0}/{1}"
                    .format(run_number, total_runs)
                )

                result = run_single_experiment(
                    nonlinearity=nonlinearity,
                    depth=args.depth,
                    weight_var=float(weight_var),
                    bias_var=float(bias_var),
                    num_train=args.num_train,
                    num_eval=args.num_eval,
                    n_gauss=args.n_gauss,
                    n_var=args.n_var,
                    n_corr=args.n_corr
                )

                append_result(args.output_file, result)

                print(
                    "Result: test_acc={0}, status={1}"
                    .format(
                        result["test_acc"],
                        result["status"]
                    )
                )

    print("")
    print("Finished.")
    print("Results saved to:")
    print(args.output_file)


if __name__ == "__main__":
    main()