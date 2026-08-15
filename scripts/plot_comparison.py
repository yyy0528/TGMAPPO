"""Plot comparable CSV metrics produced by TG-MAPPO and baselines."""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("delivery_rate", "Delivery Rate"),
    ("average_hops", "Average Hops"),
    ("average_delay", "Average Delay"),
)


def read_metrics(path):
    values = {"episode": []}
    for key, _ in METRICS:
        values[key] = []

    with open(path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(values)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "{} is missing columns: {}".format(
                    path, ", ".join(sorted(missing))
                )
            )
        for row in reader:
            values["episode"].append(int(row["episode"]))
            for key, _ in METRICS:
                values[key].append(float(row[key]))
    return values


def algorithm_name(path):
    suffix = "_metrics.csv"
    name = os.path.basename(path)
    return name[:-len(suffix)] if name.endswith(suffix) else name


def plot_all(input_dir, output_dir):
    paths = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith("_metrics.csv")
    )
    if not paths:
        raise ValueError("No *_metrics.csv files found in {}".format(input_dir))

    datasets = [(algorithm_name(path), read_metrics(path)) for path in paths]
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    for key, label in METRICS:
        plt.figure(figsize=(10, 6))
        for name, data in datasets:
            plt.plot(data["episode"], data[key], label=name)
        plt.xlabel("Episode")
        plt.ylabel(label)
        plt.title("{} Comparison".format(label))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, "{}_comparison.png".format(key))
        plt.savefig(output_path, dpi=300)
        plt.close()
        output_paths.append(output_path)
    return output_paths


def build_parser():
    parser = argparse.ArgumentParser(description="Plot all unified metric CSV files")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--num-moving-buoys", "--num_moving_buoys", type=int, default=25
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    move_dir = "moves_{}".format(args.num_moving_buoys)
    input_dir = os.path.join(args.output_root, move_dir, "logs")
    output_dir = os.path.join(args.output_root, move_dir, "comparison_plots")
    try:
        outputs = plot_all(input_dir, output_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc))
    for path in outputs:
        print("Saved: " + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
