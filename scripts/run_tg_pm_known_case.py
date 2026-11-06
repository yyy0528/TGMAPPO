"""Replay the previously failed seed 1042 with one frozen declared PM candidate."""
import argparse,json,statistics,subprocess,sys,time
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from buoy_network.ablation_suite import read,atomic,digest,require
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_action_credit import value_cache_audit,digest_bytes
from analyze_tg_pm_task_return import physical_return
from run_tg_pm_action_credit import setarg

def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--wait',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();require(ROOT/'results/tg_pm_action_credit' in root.parents,'Unexpected experiment path')
    plan=read(root/'known_case_regression_plan.json')
    outroot=root/'known_case1042';require(not outroot.exists(),'Existing regression is never retrained')
    outroot.mkdir()
    result={'status':'waiting','root':str(outroot),'controller_sha256':digest(Path(__file__)),
            'plan_sha256':digest(root/'known_case_regression_plan.json'),'jobs':[]}
    atomic(outroot/'manifest.json',result);print('OUTPUT '+str(outroot),flush=True)
    try:
        parent=read(root/'manifest.json')
        if a.wait:
            while parent['status']=='running':
                time.sleep(15);parent=read(root/'manifest.json')
        require(parent['status']=='completed','Parent trial is not complete')
        if not parent['development']['passed']:
            result.update(status='skipped',reason='Development gate failed; no regression training')
            atomic(outroot/'manifest.json',result);return
        require(parent['protocol']['value_mode']=='base_residual' and parent['protocol']['candidate']=='learned_regret','Unexpected candidate')
        for name,h in parent['source']['files_sha256'].items():require(digest(root/'source_snapshot'/name)==h,'Source archive changed')
        result['parent_manifest_sha256']=digest(root/'manifest.json')
        def run(template,tag,kind,scene=None,checkpoint=None):
            cmd=list(template['command']);episodes=20 if kind=='training' else 2
            for k,v in (('--seed',1042 if scene is None else scene),('--scenario_seed',1042 if scene is None else scene),
                        ('--output_root',outroot/tag),('--num_episodes',episodes),('--num_env_steps',episodes*80)):
                setarg(cmd,k,v)
            if checkpoint:setarg(cmd,'--tg_policy_checkpoint',checkpoint)
            j={'tag':tag,'kind':kind,'scene':scene,'command':cmd,'state':'running','started':datetime.now().isoformat(),
               'output':str(outroot/tag/'runs/case/moves_5')}
            result['jobs'].append(j);result['status']='running';atomic(outroot/'manifest.json',result);print('START '+tag,flush=True)
            with (outroot/(tag+'.log')).open('xb') as f:proc=subprocess.run(cmd,cwd=str(root/'source_snapshot'),stdout=f,stderr=subprocess.STDOUT,timeout=2400)
            j['exit_code']=proc.returncode;require(proc.returncode==0,'Regression worker failed: '+tag)
            out=Path(j['output']);r=read(out/'logs/TG_policy_report.json')
            require(r['pm_action_credit']==dict(mode='learned_regret',coefficient=.1,temperature=.2)
                and r['pm']['value_mode']=='base_residual' and r['pm']['frozen'],'Candidate changed')
            require(r['profile']['collect_calls']==episodes*80 and r['resources']['actor_trainable']==21956
                and r['resources']['critic_trainable']==18615,'Budget/capacity changed')
            audit_reward_csv(out/'logs',r['reward_contract'])
            if kind=='training':
                require(digest(r['checkpoint'])==r['checkpoint_sha256'] and len(r['training_updates'])==20,'Training artifact changed')
                for i,u in enumerate(r['training_updates'],1):
                    require(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*i and u['actor_lr']==.0005 and u['critic_lr']==.0003,'PPO budget changed')
                j['gae_audits']=value_cache_audit(out,'base_residual',20)
                require(len(r['pm_action_cache_audits'])==20 and all(x['unchanged'] for x in r['pm_action_cache_audits']),'Target cache mutated')
                for ep in range(20):
                    with np.load(str(out.parent/'value_audit'/('episode_'+str(ep)+'.npz'))) as cache:
                        target=cache['pm_action_targets'];mask=cache['pm_available_actions']
                        require(np.allclose(target.sum(-1),1.,atol=1e-5) and np.all(target[mask==0]==0)
                            and np.all(target[mask>0]>=1e-30),'Target support underflow')
                        require(digest_bytes(target.tobytes())==r['pm_action_cache_audits'][ep]['sha256'],'Target differs from actual PPO')
            else:
                require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged']
                    and digest(checkpoint)==r['checkpoint_sha256'],'Evaluation changed model')
            rows=[json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            j.update(state='completed',finished=datetime.now().isoformat(),delivered=sum(x['delivered_packets'] for x in rows),
                generated=sum(x['generated_packets'] for x in rows),base_return=physical_return(rows,r['reward_contract']),
                loop_transfers=sum(x['loop_transfers'] for x in rows),report_sha256=digest(out/'logs/TG_policy_report.json'),
                physical_sha256=digest(out/'logs/TG_physical_metrics.jsonl'))
            atomic(outroot/'manifest.json',result);print('DONE '+tag+' '+str(j['delivered'])+'/'+str(j['generated']),flush=True)
            return r.get('checkpoint')
        train=next(j for j in parent['jobs'] if j['tag']=='dev_learned_regret_11_train')
        evaluate=next(j for j in parent['jobs'] if j['tag']=='dev_learned_regret_11_eval_8001')
        ck=run(train,'train','training')
        for scene in plan['evaluation_scenes']:run(evaluate,'eval_'+str(scene),'evaluation',scene,ck)
        old=read(ROOT/plan['reference_manifest'])
        references={mode:[j for j in old['jobs'] if j['seed']==1042 and j['mode']==mode and j['kind']=='evaluation' and j['tag'].startswith('holdout_')] for mode in ('disabled','learned')}
        current=[j for j in result['jobs'] if j['kind']=='evaluation']
        summaries={}
        for label,jobs in list(references.items())+[('candidate',current)]:
            require(len(jobs)==3,'Missing scenes')
            summaries[label]={'delivery':statistics.mean(j['delivered']/j['generated'] for j in jobs),
                'base_return':statistics.mean(j['base_return'] for j in jobs),'delivered_per_scene':[j['delivered'] for j in jobs]}
        passed=all(j['delivered']/j['generated']>=next(x['delivered']/x['generated'] for x in references['disabled'] if x['scene']==j['scene']) for j in current)
        passed=passed and summaries['candidate']['base_return']>=summaries['disabled']['base_return']
        result.update(status='completed',engineering_passed=True,passed=passed,summaries=summaries,
            reference_manifest_sha256=digest(ROOT/plan['reference_manifest']),training_slots=1600,evaluation_slots=480)
    except Exception as exc:
        result.update(status='failed',error=repr(exc));atomic(outroot/'manifest.json',result);raise
    atomic(outroot/'manifest.json',result);print('REGRESSION_GATE '+str(result['passed']),flush=True)

if __name__=='__main__':main()
