"""Predeclared frozen-PM action credit trial: compatibility, development, then sealed holdout."""
import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from buoy_network.ablation_suite import read,atomic,digest,require,snapshot
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_task_return import physical_return

def setarg(cmd,key,value):
    if key in cmd:cmd[cmd.index(key)+1]=str(value)
    else:cmd.extend([key,str(value)])

def main():
    import numpy as np
    import torch
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume')
    p.add_argument('--development-only',action='store_true',help='Complete known-seed development and keep holdout sealed')
    p.add_argument('--continue-holdout',help='Open a previously sealed completed development run without retraining')
    p.add_argument('--value-mode',choices=('learned_shaped','base_residual'),default='learned_shaped')
    p.add_argument('--holdout-seed-base',type=int,default=1041)
    p.add_argument('--holdout-scene-base',type=int,default=9101)
    p.add_argument('--objective',choices=('cross_entropy','expected_regret','top_set'),default='cross_entropy')
    p.add_argument('--start-after',help='Wait for a preceding manifest to finish before creating this trial')
    a=p.parse_args()
    if a.start_after:
        import time
        prior=Path(a.start_after).resolve()
        require(ROOT/'results/tg_pm_action_credit' in prior.parents,'Wait path outside PM experiments')
        print('WAIT '+str(prior),flush=True)
        while read(prior/'manifest.json')['status'] in ('waiting','running'):time.sleep(15)
        require(read(prior/'manifest.json')['status']=='completed','Preceding trial failed; do not auto-launch another')
    refroot=ROOT/'results/tg_pm_task_return/pilot_20261105_140555_016580'
    lowroot=ROOT/'results/tg_pm_strength/20261105_154758_858806'
    reference,low=read(refroot/'manifest.json'),read(lowroot/'manifest.json')
    require(reference['status']==low['status']=='completed','References incomplete')
    require(not (a.resume and a.continue_holdout),'Choose resume or holdout continuation')
    if a.resume or a.continue_holdout:
        root=Path(a.resume or a.continue_holdout).resolve();require(ROOT/'results/tg_pm_action_credit' in root.parents,'Resume outside experiment root')
        m=read(root/'manifest.json')
        if a.continue_holdout:
            require(m['status']=='completed' and m['development']['passed'] and m['holdout'].get('status')=='not_started_budget','Require a passed, sealed development run')
            m['holdout_release']={'requested':datetime.now().isoformat(),'reuses_training_checkpoints':True}
        else:
            require(m['status']=='failed','Only resume a failed/incomplete audit')
            m.setdefault('audit_failures',[]).append(m.pop('error',None))
        m['status']='running';m.setdefault('resume_controllers',[]).append(digest(Path(__file__)))
    else:
        root=ROOT/'results/tg_pm_action_credit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');root.mkdir(parents=True,exist_ok=False)
        m={'status':'running','root':str(root),'source':snapshot(root),'jobs':[],
           'protocol':{'development_seeds':[11,22,33],'development_scenes':[8001,8002,8003],
             'holdout_seeds':list(range(a.holdout_seed_base,a.holdout_seed_base+5)),'holdout_scenes':list(range(a.holdout_scene_base,a.holdout_scene_base+3)),
             'value_mode':a.value_mode,
             'episodes':20,'eval_episodes':2,'alpha':.1,'coefficient':.1,'temperature':.2,
             'candidate':{'cross_entropy':'learned','expected_regret':'learned_regret','top_set':'learned_set'}[a.objective],
             'control':{'cross_entropy':'zero_model','expected_regret':'zero_regret','top_set':'zero_set'}[a.objective],
             'development_only':a.development_only,'top_set_log_gap':(.1 if a.objective=='top_set' else None),
             'objective':a.objective,'extra_trainable_parameters':0,
             'development_gate':'Learned exceeds A and old PM means and zero-model mean; >=5pp over A, all seeds >=max(A,old PM), >=2 wins over old PM.',
             'holdout_gate':'All 5 learned seeds strictly exceed matched A; learned no worse than zero_model per seed and strictly greater mean; >=5pp over A.',
             'statistical_unit':'training_seed; scenes averaged within seed, not treated as extra independent seeds',
             'conditional_holdout':'Run only once development passes; otherwise keep sealed.',
             'no_guarantee':'Auxiliary approximate packet objective is biased; not exact COMA or policy-invariant shaping.'},
           'reference_manifests':{str(x):digest(x) for x in (refroot/'manifest.json',lowroot/'manifest.json')}}
    for name,h in m['source']['files_sha256'].items():require(digest(root/'source_snapshot'/name)==h,'Runtime archive changed')
    value_mode=m['protocol'].get('value_mode','learned_shaped')
    holdout_seeds=tuple(m['protocol']['holdout_seeds']);holdout_scenes=tuple(m['protocol']['holdout_scenes'])
    candidate_mode=m['protocol']['candidate'];control_mode=m['protocol']['control']
    criteria_path=root/'extra_acceptance_criteria.json'
    if not criteria_path.exists():
        require(not any(j['tag'].startswith('holdout_') for j in m['jobs']), 'Cannot add return criteria after holdout started')
        atomic(criteria_path,{'published':datetime.now().isoformat(),'before_holdout_started':True,
            'criterion':'Learned mean base return exceeds A, old_PM and control in development; exceeds A and control in holdout.',
            'reason':'Higher delivery must not hide worse original task return.',
            'changes_training_or_hyperparameters':False,'does_not_relax_delivery_gates':True})
    atomic(root/'manifest.json',m);print('OUTPUT '+str(root),flush=True)

    def run(ref,tag,kind,mode,episodes,seed,scene=None,checkpoint=None,coef=.1,compat=False):
        cmd=list(ref['command']);cmd[0]=sys.executable;cmd[1]=str(root/'source_snapshot/scripts/pm_value_worker.py')
        for key,value in (('--snapshot',root/'source_snapshot'),('--output_root',root/tag),
             ('--pm_critic_context','disabled'),('--pm_action_credit',mode),('--pm_action_credit_coef',coef),
             ('--pm_action_temperature',.2),('--num_episodes',episodes),('--num_env_steps',episodes*80),
             ('--seed',seed if scene is None else scene),('--scenario_seed',seed if scene is None else scene)):
            setarg(cmd,key,value)
        if '--pm_reward_checkpoint' in cmd:setarg(cmd,'--pm_value_mode',value_mode)
        if checkpoint is not None:setarg(cmd,'--tg_policy_checkpoint',checkpoint)
        oldjob=next((j for j in m['jobs'] if j['tag']==tag),None)
        if oldjob:
            require(oldjob['command']==cmd,'Resume command changed')
            require(oldjob['exit_code']==0,'Cannot reuse failed training')
            job=oldjob
            if job['state']=='completed':
                require(digest(Path(job['output'])/'logs/TG_policy_report.json')==job['report_sha256'],'Completed artifact changed')
                r=read(Path(job['output'])/'logs/TG_policy_report.json')
                return r.get('checkpoint')
            job['process_reused_for_audit']=True
        else:
            job={'tag':tag,'kind':kind,'mode':mode,'seed':seed,'scene':scene,'episodes':episodes,'coefficient':coef,
                 'command':cmd,'output':str(root/tag/'runs/case/moves_5'),'state':'running','started':datetime.now().isoformat()}
            m['jobs'].append(job);atomic(root/'manifest.json',m);print('START '+tag,flush=True)
            with (root/(tag+'.log')).open('xb') as log:
                process=subprocess.run(cmd,cwd=str(root/'source_snapshot'),stdout=log,stderr=subprocess.STDOUT,timeout=2400)
            job['exit_code']=process.returncode;require(process.returncode==0,'Task failed: '+tag)
        out=Path(job['output']);r=read(out/'logs/TG_policy_report.json')
        config=read(out/'logs/TG_MAPPO_environment.json')['config']
        require(r['pm_action_credit']==dict(mode=mode,coefficient=float(coef),temperature=.2),'Action credit spec mismatch')
        require(r['profile']['collect_calls']==episodes*80 and r['resources']['actor_trainable']==21956
                and r['resources']['critic_trainable']==18615,'Budget/capacity changed')
        previous=read(Path(ref['output'])/'logs/TG_MAPPO_environment.json')['config']
        for key,value in dict(pm_reliability_report='',pm_critic_context='disabled',
                             pm_action_credit='disabled',pm_action_credit_coef=.1,pm_action_temperature=.2).items():previous.setdefault(key,value)
        ignored={'output_root','tg_policy_checkpoint','num_episodes','num_env_steps','requested_num_env_steps',
                 'requested_num_episodes','effective_num_env_steps','training_budget_source','seed','scenario_seed',
                 'pm_action_credit','pm_action_credit_coef','pm_value_mode'}
        require({k:v for k,v in config.items() if k not in ignored}==
                {k:v for k,v in previous.items() if k not in ignored},'Another setting changed')
        if r.get('pm'):
            original_pm=read(Path(ref['output'])/'logs/TG_policy_report.json')['pm']
            require(r['pm']['frozen'] and r['pm']['alpha']==.1 and r['pm']['value_mode']==value_mode
                    and r['pm']['checkpoint_sha256']==original_pm['checkpoint_sha256'],'Wrong PM')
        if kind=='training':
            require(len(r['training_updates'])==episodes and digest(r['checkpoint'])==r['checkpoint_sha256'],'Missing training')
            for i,u in enumerate(r['training_updates'],1):
                require(u['actor_optimizer_steps']==u['critic_head_optimizer_steps']==2*i and u['actor_lr']==.0005
                    and u['critic_lr']==.0003 and u['critic_head_in_optimizer'],'Optimizer budget changed')
                require(all(math.isfinite(v) for v in u.values() if isinstance(v,(int,float))),'Nonfinite PPO')
                if mode!='disabled' and coef>0:require(u['pm_action_target_loss']>0,'No auxiliary training')
            if mode!='disabled':
                require(len(r['pm_action_cache_audits'])==episodes and all(x['unchanged'] for x in r['pm_action_cache_audits']),'Target cache mutated')
                for ep in range(episodes):
                    with np.load(str(out.parent/'value_audit'/('episode_'+str(ep)+'.npz'))) as cache:
                        targets=cache['pm_action_targets'];mask=cache['pm_available_actions']
                        require(np.allclose(targets.sum(-1),1,atol=1e-5) and np.all(targets[mask==0]==0),'Target/mask mismatch')
        else:
            require(r['optimizer_updates']==0 and r['actor_unchanged'] and r['critic_unchanged'] and
                    r['checkpoint_sha256']==digest(checkpoint),'Frozen model changed')
        physical=[json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        audit_reward_csv(out/'logs',r['reward_contract'])
        if compat:
            residual = r.get('pm') is not None and r['pm']['value_mode']=='base_residual'
            old=Path(reference_job(seed,base=True)['output'] if residual else ref['output'])
            oldrows=read(old/'logs/TG_policy_report.json')['training_updates'][:episodes]
            actual=[{k:v for k,v in u.items() if k!='pm_action_target_loss'} for u in r['training_updates']]
            require(actual==oldrows,'Default/zero coefficient PPO differs')
            for name in ('TG_physical_metrics.jsonl','TG_reward_components.csv','TG_PM_rewards.csv'):
                q=old/'logs'/name
                if q.exists() and (not residual or name.endswith('.jsonl')):
                    n=episodes if name.endswith('.jsonl') else 80*episodes+1
                    require((out/'logs'/name).read_bytes()==b''.join(q.read_bytes().splitlines(keepends=True)[:n]),'Default behavior differs')
            if seed==22:
                group='A' if not config['use_pm_reward'] or residual else 'alpha01'
                path=ROOT/'results/tg_pm_update_replay/20261105_211844_626825'/('prefix_'+group)/'replay/moves_5/checkpoints/tg_policy_latest.pt'
                oldstate=torch.load(str(path),map_location='cpu');newstate=torch.load(r['checkpoint'],map_location='cpu')
                require(all(torch.equal(oldstate[n][k],newstate[n][k]) for n in ('actor','critic') for k in oldstate[n]),'Prefix tensors differ')
        job.update(state='completed',finished=datetime.now().isoformat(),delivered=sum(x['delivered_packets'] for x in physical),
            generated=sum(x['generated_packets'] for x in physical),base_return=physical_return(physical,r['reward_contract']),
            report_sha256=digest(out/'logs/TG_policy_report.json'),physical_sha256=digest(out/'logs/TG_physical_metrics.jsonl'),
            compatibility_exact=compat)
        atomic(root/'manifest.json',m);print('DONE {} {}/{}'.format(tag,job['delivered'],job['generated']),flush=True)
        return r.get('checkpoint')

    def reference_job(seed,kind='training',scene=None,base=False):
        return next(j for j in (reference['jobs'] if base else low['jobs'])
             if j['seed']==seed and j['kind']==kind and j['scene']==scene and (not base or j['group']=='A'))

    def rates(prefix,mode,seeds):
        result={}
        for seed in seeds:
            jobs=[j for j in m['jobs'] if j['tag'].startswith(prefix) and j['mode']==mode and j['kind']=='evaluation' and j['seed']==seed]
            require(len(jobs)==3,'Missing evaluation scenes')
            result[str(seed)]=dict(delivery=statistics.mean(j['delivered']/j['generated'] for j in jobs),
                                  base_return=statistics.mean(j['base_return'] for j in jobs))
        return result

    try:
        for seed in (11,22,33):
            for group,base in (('A',True),('PM',False)):
                ref=reference_job(seed,base=base)
                run(ref,'compat_'+group+'_'+str(seed),'training','disabled',2,seed,compat=True)
        run(reference_job(22),'compat_zero_coefficient','training',candidate_mode,2,22,coef=0,compat=True)
        for mode in (control_mode,candidate_mode):
            ck=run(reference_job(22),'smoke_'+mode+'_train','training',mode,2,22)
            run(reference_job(22,'evaluation',8001),'smoke_'+mode+'_eval','evaluation',mode,2,22,8001,ck)
        for mode in (control_mode,candidate_mode):
            for seed in (11,22,33):
                ck=run(reference_job(seed),'dev_'+mode+'_'+str(seed)+'_train','training',mode,20,seed)
                for scene in (8001,8002,8003):
                    run(reference_job(seed,'evaluation',scene),'dev_'+mode+'_'+str(seed)+'_eval_'+str(scene),
                        'evaluation',mode,2,seed,scene,ck)
        learned,control=rates('dev_',candidate_mode,(11,22,33)),rates('dev_',control_mode,(11,22,33))
        baserates={str(s):statistics.mean(j['delivered']/j['generated'] for j in reference['jobs'] if j['group']=='A' and j['kind']=='evaluation' and j['seed']==s) for s in (11,22,33)}
        oldrates={str(s):statistics.mean(j['delivered']/j['generated'] for j in low['jobs'] if j['kind']=='evaluation' and j['seed']==s) for s in (11,22,33)}
        mean=lambda values:statistics.mean(x['delivery'] for x in values.values())
        gate=mean(learned)>max(mean(control),statistics.mean(oldrates.values())) and mean(learned)>=statistics.mean(baserates.values())+.05
        gate=gate and all(learned[s]['delivery']>=max(baserates[s],oldrates[s]) for s in learned)
        gate=gate and sum(learned[s]['delivery']>oldrates[s] for s in learned)>=2
        m['development']={'learned':learned,'zero_model':control,'A':baserates,'old_PM':oldrates,'passed':gate}
        atomic(root/'manifest.json',m);print('DEVELOPMENT_GATE '+str(gate),flush=True)
        if gate and (not m['protocol'].get('development_only',False) or m.get('holdout_release')):
            for mode in ('disabled',control_mode,candidate_mode):
                for seed in holdout_seeds:
                    base=mode=='disabled'
                    ck=run(reference_job(11,base=base),'holdout_'+mode+'_'+str(seed)+'_train','training',mode,20,seed)
                    for scene in holdout_scenes:
                        run(reference_job(11,'evaluation',8001,base),'holdout_'+mode+'_'+str(seed)+'_eval_'+str(scene),
                            'evaluation',mode,2,seed,scene,ck)
            values={mode:rates('holdout_',mode,holdout_seeds) for mode in ('disabled',control_mode,candidate_mode)}
            held=values[candidate_mode];base=values['disabled'];ctrl=values[control_mode]
            passed=all(held[s]['delivery']>base[s]['delivery'] and held[s]['delivery']>=ctrl[s]['delivery'] for s in held)
            passed=passed and mean(held)>mean(ctrl) and mean(held)>=mean(base)+.05
            m['holdout']={'groups':values,'passed':passed,'scope':'5 predeclared new training seeds; 20-rollout budget only'}
        elif gate:m['holdout']={'status':'not_started_budget','reason':'Development-only budget; training checkpoints retained for later holdout continuation'}
        else:m['holdout']={'status':'not_started_sealed','reason':'Development gate failed'}
        m['status']='completed'
    except Exception as exc:
        m['status']='failed';m['error']=repr(exc)
        if m['jobs'] and m['jobs'][-1]['state']=='running':m['jobs'][-1]['state']='failed'
        atomic(root/'manifest.json',m);raise
    atomic(root/'manifest.json',m);print('COMPLETED '+str(root/'manifest.json'),flush=True)

if __name__=='__main__':main()
