"""Sample an INITIAL frozen original-TG policy, only for pre-PPO calibration."""
import csv
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def collect_initial(config, output, training_seed, scene):
    from .environment import buoy_environment
    from .agent import Agent,DeepQNetwork
    from .runner import Runner
    from .policy_artifacts import signature
    from .task_reward_audit import audit_reward_csv
    from .task_reward import reward_contract
    from .wm_runtime import tensor_digest
    from src.legacy_cli import resolve_training_budget
    if (scene==training_seed or config['use_wm_actor'] or config['export_wm_data']
            or config['use_recurrent_policy'] or config['use_naive_recurrent_policy']
            or config['tg_reward_version']!='task_v2' or config['pm_potential_mode']!='task_return'
            or config['pm_value_mode']!='learned_shaped' or not config['use_pm_reward']):
        raise ValueError('Calibration requires an independent scene and feedforward task-return PM')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    budget=resolve_training_budget(['--num_episodes','--num_env_steps'],config['num_episodes'],
        config['episode_length'],config['num_episodes']*config['episode_length'])
    calibrated_config=dict(config,**budget)
    args=SimpleNamespace(**dict(calibrated_config,seed=training_seed,scenario_seed=scene,output_root=str(output),
        tg_evaluate=True,tg_policy_checkpoint='',pm_reliability_report=''))
    random.seed(training_seed);np.random.seed(training_seed);torch.manual_seed(training_seed)
    env=buoy_environment([[10,[i,j]] for i in range(args.grid_size) for j in range(args.grid_size)],args)
    env.build_env()
    learner=DeepQNetwork(max(env.get_n_action(i) for i in range(env.buoy_number)),
        max(env.get_n_obs(i) for i in range(env.buoy_number)),learning_rate=.001,reward_decay=.9,
        e_greedy=1,replace_target_iter=200,memory_size=100000,env=env)
    before={n:tensor_digest(getattr(learner.policy,n).state_dict()) for n in ('actor','critic')}
    cache=[];collect=learner.collect;get_actions=learner.policy.get_actions
    def sampled(*a,**kw):
        kw['deterministic']=False
        return get_actions(*a,**kw)
    learner.policy.get_actions=sampled
    def observed(step):
        if step==0:cache.append({'scene':scene,'episode':len(cache),'values':[],'phi':[]})
        if len(cache[-1]['values'])!=step:raise ValueError('Calibration capture time mismatch')
        result=collect(step)
        values=(learner.trainer.value_normalizer.denormalize(result[0])
            if learner.buffer._use_popart or learner.buffer._use_valuenorm else result[0])
        cache[-1]['values'].append(np.asarray(values).reshape(env.buoy_number).astype(float).tolist())
        cache[-1]['phi'].append(float(learner.pm_runtime.phi))
        return result
    learner.collect=observed
    def forbidden(*a,**kw):raise AssertionError('Calibration attempted PPO or GAE update')
    learner.compute=learner.train=forbidden
    # This is sampled INITIAL-policy calibration, not a trained greedy evaluation.
    # Write the explicit phase report below instead of the production evaluation label.
    learner.save_policy=lambda run_dir: None
    random.seed(scene);np.random.seed(scene);torch.manual_seed(scene)
    Runner([Agent(b,learner) for b in env.buoys],env).run()
    after={n:tensor_digest(getattr(learner.policy,n).state_dict()) for n in ('actor','critic')}
    if before!=after or learner.profile['ppo_updates']!=0 or learner.policy.actor_optimizer.state or learner.policy.critic_optimizer.state:
        raise ValueError('Initial calibration policy/optimizer was modified')
    learner.pm_runtime.assert_frozen()
    log=output/('moves_'+str(args.num_moving_buoys))/'logs'
    audit_reward_csv(log,reward_contract(args))
    with (log/'TG_reward_components.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    with (log/'TG_PM_rewards.csv').open(encoding='utf-8',newline='') as stream:pm=list(csv.DictReader(stream))
    h=args.episode_length
    if len(cache)!=args.num_episodes or len(rows)!=len(cache)*h or len(pm)!=len(cache)*h:
        raise ValueError('Incomplete calibration tasks')
    for ep,row in enumerate(cache):
        if len(row['values'])!=h:raise ValueError('Incomplete calibration values')
        observed_rows=rows[ep*h:(ep+1)*h];pm_rows=pm[ep*h:(ep+1)*h]
        if (any(float(p['phi_t'])!=row['phi'][t] for t,p in enumerate(pm_rows))
                or float(pm_rows[-1]['phi_tp1'])!=0):
            raise ValueError('Calibration potential cache mismatch')
        row['phi'].append(0.)
        row['rewards']=[float(r['reward_base']) for r in observed_rows]
    result={'phase':'independent_initial_policy_calibration','training_seed':training_seed,'scene':scene,
        'initial_actor_sha256':before['actor'],'initial_critic_sha256':before['critic'],
        'policy_signature':signature(learner),'pm_checkpoint_sha256':learner.pm_runtime.file_sha256,
        'weights_unchanged':True,'optimizer_updates':0,'collect_calls':learner.profile['collect_calls'],
        'torch_threads':torch.get_num_threads(),'episodes':cache,
        'physical_sha256':hashlib.sha256((log/'TG_physical_metrics.jsonl').read_bytes()).hexdigest(),
        'interpretation':'sampled initial policy, independent calibration scenes, no PPO/GAE training'}
    (output/'calibration_part.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result
