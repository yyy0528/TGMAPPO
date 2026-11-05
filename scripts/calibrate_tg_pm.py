"""Standalone independent pre-PPO calibration of one original TG initialization."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from buoy_network.pm_calibration import collect_initial
from buoy_network.pm_reliability import SCHEMA,select_gate
from buoy_network.task_reward import reward_contract
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config',required=True)
    p.add_argument('--training-seed',required=True,type=int)
    p.add_argument('--scenes',nargs=2,type=int,required=True)
    p.add_argument('--pm-checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    config=json.loads(Path(a.environment_config).read_text(encoding='utf-8'))['config']
    config.update(num_episodes=2,num_env_steps=config['episode_length']*2,
        use_pm_reward=True,pm_reward_checkpoint=str(Path(a.pm_checkpoint).resolve()),pm_reward_alpha=1.,
        pm_potential_mode='task_return',pm_value_mode='learned_shaped',pm_reliability_report='')
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    parts=[collect_initial(config,output/('scene_'+str(scene)),a.training_seed,scene) for scene in a.scenes]
    for part in parts:
        if any(part[k]!=parts[0][k] for k in ('initial_actor_sha256','initial_critic_sha256','policy_signature','pm_checkpoint_sha256')):
            raise ValueError('Calibration initializations differ across scenes')
    rows=[row for part in parts for row in part['episodes']]
    h=config['episode_length']
    report={k:parts[0][k] for k in ('initial_actor_sha256','initial_critic_sha256','policy_signature','pm_checkpoint_sha256')}
    report.update(schema=SCHEMA,phase='independent_initial_policy_calibration',training_seed=a.training_seed,
        horizon=h,nodes=config['num_agents'],gamma=config['gamma'],maximum_alpha=1.,
        potential_mode='task_return',value_mode='learned_shaped',weights_unchanged=True,optimizer_updates=0,
        reward_contract=reward_contract(SimpleNamespace(**config)),episodes=rows,
        decision=select_gate(rows,h,config['num_agents'],config['gamma'],1.),
        source_parts=[str(output/('scene_'+str(s))/'calibration_part.json') for s in a.scenes])
    (output/'reliability.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('DECISION '+json.dumps(report['decision']),flush=True)
    print('REPORT '+str(output/'reliability.json'),flush=True)


if __name__=='__main__':main()
