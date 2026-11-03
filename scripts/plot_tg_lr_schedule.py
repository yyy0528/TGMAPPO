"""Descriptive LR comparison: independent policies, no significance inference."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True)
    parser.add_argument("--output-root",default="results/tg_lr_comparison")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(2,1,figsize=(12,9))
    colors = {"fixed":"#246b86","linear":"#bd7933"}
    for kind,offset in (("fixed",-.18),("linear",.18)):
        summary = report["suites"][kind]["summary"]
        groups = summary["groups"]
        x = np.arange(len(groups))+offset
        means = [g["metrics"]["delivery_rate"]["mean"]*100 for g in groups]
        sd = [g["metrics"]["delivery_rate"]["sample_std"]*100 for g in groups]
        axes[0].bar(x,means,width=.32,color=colors[kind],alpha=.45,label=kind.capitalize(),yerr=sd,capsize=4)
        for index,g in enumerate(groups):
            values = [r["delivery_rate"]*100 for r in summary["by_training_seed"] if r["group"] == g["group"]]
            axes[0].scatter(x[index]+np.linspace(-.05,.05,len(values)),values,color=colors[kind],s=25,zorder=5)
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels(["A: TG","B: WM","C: PM","D: WM+PM","E: Capacity","F: History"])
    axes[0].set_ylim(-8,40)
    axes[0].set_ylabel("Delivery rate (%)")
    axes[0].set_title("2 training episodes: 3 independent policies/group; error bars = sample SD, not CI")
    axes[0].legend()
    rows = report["baseline_20_episode_comparison"]
    for kind,field in (("fixed","fixed_delivered"),("linear","linear_delivered")):
        axes[1].plot(range(len(rows)),[r[field]/r["generated"]*100 for r in rows],marker="o",color=colors[kind],label=kind.capitalize())
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels(["Training seed "+str(r["seed"]) for r in rows])
    axes[1].set_ylabel("Delivery rate (%)")
    axes[1].set_ylim(0,100)
    axes[1].set_title("20 training episodes: BASELINE ONLY; frozen greedy evaluation on development scene 7001")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for extension in ("png","svg"):
        fig.savefig(str(root/("comparison."+extension)),dpi=180)
    plt.close(fig)
    provenance = {"receipt_sha256":hashlib.sha256(Path(args.report).read_bytes()).hexdigest(),
        "plotter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"scope":"descriptive development comparison; no significance test"}
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print("Figures: "+str(root))


if __name__ == "__main__":
    main()
