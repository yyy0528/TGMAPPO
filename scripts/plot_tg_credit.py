"""Plot recorded credit changes and same-input frozen routing probabilities."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True)
    p.add_argument('--output-root',required=True)
    a = p.parse_args()
    report = json.loads(Path(a.report).read_text(encoding='utf-8'))
    out = Path(a.output_root)
    out.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    cases = report['same_input_sink_cases']
    chosen = [next(r for r in cases if r['sender']==s and r['action']==17) for s in (18,23)]
    x,width = np.arange(2),.19
    for offset,(field,action,color,hatch,label) in enumerate([
        ('control_probabilities',17,'#287fa3','//','1 epoch: node 17'),
        ('control_probabilities',24,'#287fa3','','1 epoch: Sink 24'),
        ('probabilities',17,'#bf7737','//','4 epochs: node 17'),
        ('probabilities',24,'#bf7737','','4 epochs: Sink 24')]):
        axes[0].bar(x+(offset-1.5)*width,[r[field][action]*100 for r in chosen],width,
            color=color,hatch=hatch,label=label,alpha=.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Sender 18','Sender 23'])
    axes[0].set_ylabel('Action probability (%)')
    axes[0].set_title('Same observations and masks, first frozen decision')
    axes[0].legend(fontsize=8)
    first=report['first_rollout_comparison']
    means=[first[e]['positive_revisits']['p_selected_change_mean']*100 for e in ('1','4')]
    axes[1].bar([0,1],means,color=['#287fa3','#bf7737'],alpha=.8)
    axes[1].set_xticks([0,1])
    axes[1].set_xticklabels(['1 PPO epoch','4 PPO epochs'])
    axes[1].set_ylabel('Mean chosen-action probability change (percentage points)')
    n=first['1']['positive_revisits']['count']
    axes[1].set_title('Same initial rollout: {} revisits with positive centered A'.format(n),fontsize=10)
    for i,y in enumerate(means):
        axes[1].text(i,y,'{:.3f} pp'.format(y),ha='center',va='bottom')
    axes[1].set_ylim(0,max(means)*1.2)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Seed 33 diagnosis: frozen comparisons and exact training replays; no policy correction')
    fig.savefig(out/'credit_diagnosis.png',dpi=180)
    fig.savefig(out/'credit_diagnosis.svg')
    (out/'provenance.json').write_text(json.dumps({'report_sha256':hashlib.sha256(Path(a.report).read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(str(out.resolve()))


if __name__=='__main__':
    main()
