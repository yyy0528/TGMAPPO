"""Observed dominant greedy decisions at initial node positions; not physical topology."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    args = parser.parse_args()
    root = Path(args.batch)
    figures = root/"figures"
    figures.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(12,5.8))
    for ax,seed in zip(axes,(22,33)):
        result = read(root/("A_{}_trained_greedy".format(seed))/"report.json")
        analysis = result["analysis"]
        first = json.loads((root/("A_{}_trained_greedy".format(seed))/"slots.jsonl").read_text().splitlines()[0])
        positions = first["positions"]
        loop_nodes = {node for group in analysis["observed_closed_sccs"] for node in group}
        wait_nodes = {node for node,row in enumerate(analysis["action_counts_by_sender"]) if sum(row) and max(range(26),key=lambda i:row[i]) == 25}
        choices = {node:max(range(26),key=lambda i:row[i]) for node,row in enumerate(analysis["action_counts_by_sender"]) if sum(row)}
        for source,target in choices.items():
            if target == 25:
                continue
            color = "#c63838" if source in loop_nodes and target in loop_nodes else "#71889e"
            a,b = positions[source],positions[target]
            ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="-|>",color=color,
                lw=2.1 if color == "#c63838" else 1.1,shrinkA=12,shrinkB=12,connectionstyle="arc3,rad=0.13"))
        for node,(x,y) in enumerate(positions):
            color = "#3b9c71" if node == 24 else "#efb354" if node in wait_nodes else "#e7a0a0" if node in loop_nodes else "#e0e7ef"
            ax.scatter([x],[y],s=390 if node == 24 else 300,c=color,edgecolors="#33465a",zorder=3)
            ax.text(x,y,str(node),ha="center",va="center",fontsize=9,zorder=4)
            if node in wait_nodes:
                ax.text(x,y+.24,"WAIT",ha="center",va="bottom",fontsize=8,color="#9a5e13")
        ax.set_title("Seed {}: {} / 48 delivered".format(seed,analysis["delivered"]))
        ax.set_xlim(-.55,4.55)
        ax.set_ylim(-.45,4.55)
        ax.set_aspect("equal")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.grid(alpha=.15)
        ax.set_xlabel("Node x at first observation")
        ax.set_ylabel("Node y at first observation")
    fig.legend(handles=[Line2D([0],[0],color="#c63838",lw=2,label="Observed closed loop"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor="#efb354",label="Dominant active WAIT",markersize=9),
        Line2D([0],[0],marker="o",color="w",markerfacecolor="#3b9c71",label="Sink = 24",markersize=9)],loc="lower center",ncol=3)
    fig.suptitle("Frozen TG policy: dominant selected next hop over 2 x 80 slots\nArrows show decisions, not all physically legal exits",fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.90))
    fig.savefig(str(figures/"zero_delivery_actions.png"),dpi=180)
    fig.savefig(str(figures/"zero_delivery_actions.svg"))
    plt.close(fig)
    print("Plot: "+str((figures/"zero_delivery_actions.png").resolve()))


if __name__ == "__main__":
    main()
