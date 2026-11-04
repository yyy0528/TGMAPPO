"""Three-seed value-coordinate controls, exact legacy replay and frozen tests."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.buoy_network.ablation_suite import read,atomic,digest,snapshot,require


def equal(a,b):
    return all(set(a[n])==set(b[n]) and all(torch.equal(a[n][k],b[n][k]) for k in a[n]) for n in ('actor','critic'))


def main():
    old=read(ROOT/'results/tg_ablation_lr_fixed_pilot/20261103_211528_024882/manifest.json')
    root=ROOT/'results/tg_pm_value_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    result={'root':str(root),'source':snapshot(root),'jobs':[],'comparisons':[],
            'scope':'fixed2 invariance/compatibility control; not PM efficacy or fixed100 result'}
    atomic(root/'report.json',result)
    for seed in (11,22,33):
        refs={(j['group'],j['kind']):j for j in old['jobs'] if j['train_seed']==seed}
        models={}
        for mode,group in (('baseline','A'),('learned_shaped','C'),('base_residual','C')):
            for kind in (('training','evaluation') if mode=='base_residual' else ('training',)):
                tag='{}_{}_{}'.format(mode,seed,kind);name='case'
                command=list(refs[(group,kind)]['base_command'])
                command[1]=str(root/'source_snapshot/scripts/pm_value_worker.py')
                command+=['--snapshot',str(root/'source_snapshot'),'--output_root',str(root/tag),'--run-name',name]
                if mode!='baseline':command+=['--pm_value_mode',mode]
                if kind=='evaluation':command+=['--tg_policy_checkpoint',str(models[mode][0])]
                print('START '+tag,flush=True)
                with (root/(tag+'.log')).open('xb') as stream:
                    process=subprocess.run(command,cwd=str(root/'source_snapshot'),stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                output=root/tag/'runs/case/moves_5'
                entry={'tag':tag,'command':command,'exit_code':process.returncode,'output':str(output)}
                result['jobs'].append(entry);atomic(root/'report.json',result)
                require(process.returncode==0,'Probe failed: '+tag)
                report=read(output/'logs/TG_policy_report.json')
                if kind=='training':
                    require(len(report['training_updates'])==2 and report['training_updates'][-1]['actor_optimizer_steps']==4,'Wrong budget')
                    checkpoint=Path(report['checkpoint']);model=torch.load(str(checkpoint),map_location='cpu')
                    models[mode]=(checkpoint,model)
                    if mode in ('baseline','learned_shaped'):
                        ref=Path(refs[(group,kind)]['output'])
                        require(equal(model,torch.load(str(ref/'checkpoints/tg_policy_latest.pt'),map_location='cpu')),'Legacy model changed')
                        for f in ('TG_physical_metrics.jsonl','TG_reward_components.csv'):
                            require(digest(output/'logs'/f)==digest(ref/'logs'/f),'Legacy physical/reward changed')
                        entry['legacy_model_physics_rewards_exact']=True
                    else:
                        require(equal(model,models['baseline'][1]),'Residual PPO did not recover baseline')
                        entry['baseline_model_exact']=True
                    if mode!='baseline':
                        path=output.parent/'value_audit/summary.jsonl'
                        entry['value_audit']=[json.loads(x) for x in path.read_text().splitlines()]
                else:
                    require(report['optimizer_updates']==0 and report['actor_unchanged'] and report['critic_unchanged'],'Evaluation updated weights')
                    ref=Path(refs[('A',kind)]['output'])
                    require(digest(output/'logs/TG_physical_metrics.jsonl')==digest(ref/'logs/TG_physical_metrics.jsonl'),'Residual evaluation differs from baseline')
                    physical=[json.loads(x) for x in (output/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
                    result['comparisons'].append({'seed':seed,'baseline_and_residual_delivered':sum(x['delivered_packets'] for x in physical),
                        'generated':sum(x['generated_packets'] for x in physical),'physical_exact':True})
                atomic(root/'report.json',result);print('DONE '+tag,flush=True)
    result['success']=True;atomic(root/'report.json',result);print('PASS: '+str(root/'report.json'),flush=True)


if __name__=='__main__':main()
