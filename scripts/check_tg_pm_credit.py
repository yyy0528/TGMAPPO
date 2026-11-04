"""Three-seed fixed2 compatibility/control for versioned completion credit."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read,atomic,digest,snapshot,require


def main():
    previous=read(ROOT/'results/tg_ablation_lr_fixed_pilot/20261103_211528_024882/manifest.json')
    root=ROOT/'results/tg_pm_credit_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    source=snapshot(root)
    result={'root':str(root),'source':source,'jobs':[], 'comparisons':[],
            'scope':'fixed2 compatibility and exploratory comparison; not fixed100 efficacy'}
    atomic(root/'report.json',result)
    for seed in (11,22,33):
        old={j['kind']:j for j in previous['jobs'] if j['group']=='C' and j['train_seed']==seed}
        for mode in ('remaining','completion_credit'):
            for kind in (('training',) if mode=='remaining' else ('training','evaluation')):
                tag='{}_{}_{}'.format(mode,seed,kind)
                command=list(old[kind]['base_command'])
                command[1]=str(root/'source_snapshot/run_algorithm.py')
                command+=['--pm_potential_mode',mode,'--output_root',str(root/tag),'--run-name','case']
                output=root/tag/'runs/case/moves_5'
                if kind=='evaluation':
                    command+=['--tg_policy_checkpoint',str(root/('{}_{}_training'.format(mode,seed))/'runs/case/moves_5/checkpoints/tg_policy_latest.pt')]
                print('START '+tag,flush=True)
                with (root/(tag+'.log')).open('xb') as stream:
                    process=subprocess.run(command,cwd=str(root/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                record={'tag':tag,'command':command,'exit_code':process.returncode,'output':str(output)}
                result['jobs'].append(record);atomic(root/'report.json',result)
                require(process.returncode==0,'Probe failed: '+tag)
                report=read(output/'logs/TG_policy_report.json')
                require(report['pm']['frozen'] and report['pm']['potential_mode']==mode,'Wrong PM mode/freeze')
                if kind=='training':
                    require(len(report['training_updates'])==2 and report['training_updates'][-1]['actor_optimizer_steps']==4,'Wrong PPO budget')
                if mode=='remaining':
                    a,b=[torch.load(str(p),map_location='cpu') for p in
                         (output/'checkpoints/tg_policy_latest.pt',Path(old[kind]['output'])/'checkpoints/tg_policy_latest.pt')]
                    require(all(set(a[n])==set(b[n]) and all(torch.equal(a[n][k],b[n][k]) for k in a[n]) for n in ('actor','critic')),'Legacy parameters changed')
                    for filename in ('TG_physical_metrics.jsonl','TG_reward_components.csv'):
                        require(digest(output/'logs'/filename)==digest(Path(old[kind]['output'])/'logs'/filename),'Legacy trajectory/reward changed')
                    record['old_model_physics_rewards_exact']=True
                if kind=='evaluation':
                    require(report['optimizer_updates']==0 and report['actor_unchanged'] and report['critic_unchanged'],'Frozen evaluation changed weights')
                    def physical(p):return [json.loads(x) for x in (p/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
                    before,after=physical(Path(old[kind]['output'])),physical(output)
                    result['comparisons'].append({'seed':seed,'generated':sum(r['generated_packets'] for r in after),
                        'old_delivered':sum(r['delivered_packets'] for r in before),'new_delivered':sum(r['delivered_packets'] for r in after),
                        'old_loops':sum(r['loop_transfers'] for r in before),'new_loops':sum(r['loop_transfers'] for r in after)})
                atomic(root/'report.json',result)
                print('DONE '+tag,flush=True)
    result['success']=True;atomic(root/'report.json',result)
    print('PASS: '+str(root/'report.json'),flush=True)


if __name__=='__main__':main()
