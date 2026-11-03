"""Descriptive baseline PPO-budget figures and per-update CSV, no significance test."""
import argparse
import csv
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
    parser.add_argument("--output-root",default="results/tg_ppo_budget_figures")
    args = parser.parse_args()
    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    rows = data["by_training_seed"]
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    colors = {1:"#397a9b",4:"#b97938"}
    for index,epochs in enumerate((1,4)):
        selected = sorted((r for r in rows if r["ppo_epoch"] == epochs),key=lambda r:r["train_seed"])
        rates = np.array([r["delivery_rate"] for r in selected])*100
        axes[0,0].bar(index,rates.mean(),yerr=rates.std(ddof=1),capsize=5,color=colors[epochs],alpha=.4)
        axes[0,0].scatter(index+np.linspace(-.07,.07,3),rates,color=colors[epochs],s=35,zorder=4)
        entropy = np.array([r["normalized_entropy"] for r in selected])
        axes[0,1].bar(index,entropy.mean(),color=colors[epochs],alpha=.4)
        axes[0,1].scatter(index+np.linspace(-.07,.07,3),entropy,color=colors[epochs],s=35,zorder=4)
        for ax,field,multiplier in ((axes[1,0],"approx_kl",1),(axes[1,1],"clip_fraction",100)):
            series = np.array([[record[field] for record in r["training_updates"]] for r in selected])*multiplier
            for values in series:
                ax.plot(range(1,21),values,color=colors[epochs],alpha=.2,linewidth=.8)
            ax.plot(range(1,21),series.mean(axis=0),color=colors[epochs],label=str(epochs)+" PPO epochs",linewidth=2)
    axes[0,0].set_title("Frozen greedy delivery: n=3 policies; mean +/- sample SD")
    axes[0,0].set_ylabel("Delivery rate (%)")
    axes[0,0].set_ylim(0,100)
    axes[0,1].set_title("Frozen policy normalized entropy")
    axes[0,1].set_ylabel("Entropy / log(number of legal actions)")
    axes[0,1].set_ylim(0,1.05)
    for ax in axes[0]:
        ax.set_xticks([0,1])
        ax.set_xticklabels(["1 PPO epoch\n40 optimizer steps/network","4 PPO epochs\n160 optimizer steps/network"])
    axes[1,0].set_title("Approximate KL: rollout means of pre-step minibatches")
    axes[1,1].set_title("PPO clip fraction: rollout means")
    axes[1,0].set_ylabel("Approximate KL")
    axes[1,1].set_ylabel("Clipped fraction (%)")
    for ax in axes[1]:
        ax.set_xlabel("Training episode")
        ax.legend()
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("BASELINE ONLY: 20 x 80 environment slots per seed; task_v2, linear LR, evaluation scene 7001",fontsize=12)
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(str(root/("budget_comparison."+ext)),dpi=180)
    plt.close(fig)
    fields = ["ppo_epoch","train_seed","update","actor_optimizer_steps","critic_head_optimizer_steps",
              "actor_lr","critic_lr","value_loss","policy_loss","dist_entropy","actor_grad_norm","critic_grad_norm","ratio","approx_kl","clip_fraction"]
    with (root/"training_updates.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for r in rows:
            for update in r["training_updates"]:
                writer.writerow(dict({key:update[key] for key in fields if key not in ("ppo_epoch","train_seed")},
                                     ppo_epoch=r["ppo_epoch"],train_seed=r["train_seed"]))
    provenance = {"receipt_sha256":hashlib.sha256(Path(args.report).read_bytes()).hexdigest(),
                  "plotter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"scope":"descriptive development diagnostic; SD is not CI"}
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print("Figures/CSV: "+str(root))


if __name__ == "__main__":
    main()
