"""Render descriptive physical-metric plots and CSV, never claims significance."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    args = parser.parse_args()
    root = Path(args.suite).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if manifest["state"] != "completed":
        parser.error("Only completed suites can be plotted")
    labels = {"A": "TG", "B": "TG+WM", "C": "TG+PM", "D": "TG+WM+PM", "E": "Actor+2257", "F": "History"}
    groups = [row["group"] for row in summary["groups"]]
    means = [row["metrics"]["delivery_rate"]["mean"] for row in summary["groups"]]
    stds = [row["metrics"]["delivery_rate"]["sample_std"] or 0 for row in summary["groups"]]
    figures = root/"figures"
    figures.mkdir(exist_ok=True)
    title = "{}: {} training episodes/seed; n={} independent policies/group".format(summary["purpose"].upper(),
        manifest["config"]["train_episodes"], len(manifest["config"]["train_seeds"]))
    fig, ax = plt.subplots(figsize=(9,4.5))
    positions = list(range(len(groups)))
    ax.bar(positions, means, yerr=stds, capsize=4, color="#89b5d7", alpha=.8, label="Mean +/- sample SD (not CI)")
    for index, group in enumerate(groups):
        values = [row["delivery_rate"] for row in summary["by_training_seed"] if row["group"] == group]
        offsets = [(i-(len(values)-1)/2)*.10 for i in range(len(values))]
        ax.scatter([index+offset for offset in offsets], values, color="#223b52", s=24, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[g] for g in groups])
    ax.set_ylabel("Actual delivered / generated")
    # Sample SD bars can extend outside [0,1]; display their full extent, not a clipped cap.
    ax.set_ylim(min(-.05,min(m-s for m,s in zip(means,stds))-.05), max(1.05,max(m+s for m,s in zip(means,stds))+.05))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(str(figures/"evaluation_delivery.png"), dpi=180)
    fig.savefig(str(figures/"evaluation_delivery.svg"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4.5))
    for group in groups:
        series = []
        for job in manifest["jobs"]:
            if job["kind"] == "training" and job["group"] == group:
                rows = [json.loads(line) for line in (Path(job["output"])/"logs"/"TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                series.append([row["delivery_rate"] for row in rows])
        means = [statistics.mean(values) for values in zip(*series)]
        stds = [statistics.stdev(values) for values in zip(*series)]
        x = list(range(1,len(means)+1))
        line, = ax.plot(x, means, marker="o", label=labels[group])
        ax.fill_between(x, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], color=line.get_color(), alpha=.08)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Completed training episode (1-based)")
    ax.set_ylabel("Physical delivery rate; mean +/- sample SD")
    ax.set_ylim(-.05,1.05)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(str(figures/"training_delivery.png"), dpi=180)
    plt.close(fig)
    with (root/"group_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group", "metric", "mean", "sample_std", "n_defined", "independent_training_n"))
        writer.writeheader()
        for row in summary["groups"]:
            for metric, values in row["metrics"].items():
                writer.writerow(dict(group=row["group"], metric=metric, independent_training_n=row["independent_training_n"], **values))
    provenance = {"plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "summary_sha256": hashlib.sha256((root/"summary.json").read_bytes()).hexdigest(), "scope": "descriptive pilot, no significance test"}
    (figures/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Plots and CSV: "+str(figures))


if __name__ == "__main__":
    main()
