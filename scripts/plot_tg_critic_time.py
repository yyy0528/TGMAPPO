"""Descriptive single-seed clock probe plots; no confidence interval or significance claim."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True)
    p.add_argument('--output-root',required=True)
    a=p.parse_args()
    report=json.loads(Path(a.report).read_text(encoding='utf-8'))
    out=Path(a.output_root)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mode,r in report['medium_seed33'].items():
        delivered=sum(x['delivered_packets'] for x in r['greedy_physical'])
        generated=sum(x['generated_packets'] for x in r['greedy_physical'])
        rows.append({'mode':mode,'greedy_delivered':delivered,'generated':generated,
            'greedy_rate':delivered/generated,'value_mc_bias':r['value_mc_bias'],'value_mc_rmse':r['value_mc_rmse'],
            'sample_delivered':sum(x['delivered_packets'] for x in r['sample_physical'])})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    colors=['#467b98','#999999','#bd803f']
    labels=['Disabled','Constant control','Remaining time']
    for ax,field,title,ylab in ((axes[0],'greedy_rate','Frozen greedy delivery: 2 episodes','Delivery rate (%)'),
        (axes[1],'value_mc_rmse','Frozen sampled policy: value vs MC return','RMSE (task reward units)')):
        values=[r[field]*(100 if field=='greedy_rate' else 1) for r in rows]
        ax.bar(range(3),values,color=colors)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylab)
        ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0,max(values)*1.2)
        for i,value in enumerate(values):
            ax.text(i,value,'{:.3f}'.format(value),ha='center',va='bottom')
    fig.suptitle('Critic clock probe: 20 x 80 slots, train seed 33, development scene 7001; n=1 policy/mode',fontsize=11)
    fig.savefig(out/'critic_clock_probe.png',dpi=180)
    fig.savefig(out/'critic_clock_probe.svg')
    with (out/'critic_clock_probe.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'provenance.json').write_text(json.dumps({'report_sha256':hashlib.sha256(Path(a.report).read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(str(out.resolve()))


if __name__=='__main__':
    main()
