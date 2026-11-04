"""Descriptive paired fixed20/fixed100 charts and raw training curve CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True)
    p.add_argument('--output-root',required=True)
    a=p.parse_args()
    r=json.loads(Path(a.report).read_text(encoding='utf-8'))
    out=Path(a.output_root)
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    mean=[r['aggregate'][str(e)]['delivery_rate']['mean']*100 for e in (20,100)]
    sd=[r['aggregate'][str(e)]['delivery_rate']['sample_sd']*100 for e in (20,100)]
    axes[0].bar([0,1],mean,yerr=sd,capsize=5,color=['#5c8c9f','#be813b'],alpha=.5)
    colors=['#23698a','#b65d2d','#5e8545']
    for index,seed in enumerate((11,22,33)):
        pair=[next(x for x in r['by_training_seed'] if x['episodes']==e and x['seed']==seed) for e in (20,100)]
        axes[0].plot(np.array([0,1])+(index-1)*.055,[x['delivery_rate']*100 for x in pair],
            '-o',color=colors[index],label='seed '+str(seed),alpha=.9)
    axes[0].set_xticks([0,1])
    axes[0].set_xticklabels(['20 episodes\n1600 collected slots','100 episodes\n8000 collected slots'])
    axes[0].set_ylabel('Frozen greedy delivery (%)')
    axes[0].set_title('Three trained policies: mean +/- sample SD')
    axes[0].set_ylim(min(0,min(m-s for m,s in zip(mean,sd))-3),max(105,max(m+s for m,s in zip(mean,sd))+3))
    axes[0].legend(fontsize=8)
    curves=[]
    for index,seed in enumerate((11,22,33)):
        row=next(x for x in r['by_training_seed'] if x['episodes']==100 and x['seed']==seed)
        raw=row['training_curve']
        moving=[sum(x['delivery_rate'] for x in raw[end-20:end])/20 for end in range(20,101)]
        axes[1].plot(range(20,101),moving,color=colors[index],label='seed '+str(seed))
        for episode in raw:
            curves.append({'seed':seed,'episode':episode['episode'],'environment_steps':(episode['episode']+1)*80,
                'delivery_rate':episode['delivery_rate'],'loop_transfers':episode['loop_transfers'],
                'delivered_packets':episode['delivered_packets'],'generated_packets':episode['generated_packets']})
    axes[1].axhline(.8,ls='--',color='#555555',lw=1,label='predefined training target .8')
    axes[1].axvline(20,ls=':',color='#555555',lw=1)
    axes[1].set_ylim(0,1.05)
    axes[1].set_xlim(18,100)
    axes[1].set_xlabel('Training episode (80 joint slots each)')
    axes[1].set_ylabel('20-episode moving mean delivery')
    axes[1].set_title('Training trajectories; prefix20 matches preserved baseline')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Original TG baseline, fixed LR, PPO 1 x 2, no WM/PM/clock; development scene 7001',fontsize=11)
    fig.savefig(out/'interaction_budget.png',dpi=180)
    fig.savefig(out/'interaction_budget.svg')
    with (out/'training_curves.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (out/'provenance.json').write_text(json.dumps({'report_sha256':hashlib.sha256(Path(a.report).read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(str(out.resolve()))


if __name__=='__main__':
    main()
