"""One original-environment thesis job; evaluations run in isolated processes."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT)]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
TEMP = ROOT/'results/thesis_temp'
TEMP.mkdir(parents=True,exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(TEMP)
from tg_convergence import atomic, read, digest


def save_point(model, directory, save):
    """Native saver creates checkpoints but expects the logs folder to exist."""
    (Path(directory)/'logs').mkdir(parents=True,exist_ok=True)
    save(model,directory)


def inspect(output, algorithm, episodes, evaluation=False):
    import math
    output = Path(output)
    rows = [json.loads(line) for line in (output/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
    if len(rows) != episodes or any(r['horizon'] != 80 for r in rows):
        raise ValueError('Wrong physical episode budget')
    from buoy_network.task_reward_audit import audit_reward_csv
    config = read(output/'logs/TG_MAPPO_environment.json')
    audit_reward_csv(output/'logs', config['reward_contract'])
    if config['reward_contract']['version'] != 'task_v2':
        raise ValueError('Baseline must use original task_v2 reward')
    baseline = (output/'logs/baseline_report.json').is_file()
    report = read(output/('logs/baseline_report.json' if baseline else 'logs/TG_policy_report.json'))
    if baseline:
        if report['algorithm'] != algorithm or report['inference_calls'] != episodes*80:
            raise ValueError('Baseline identity/interaction mismatch')
        if evaluation:
            if not report['weights_unchanged'] or report['optimizer_updates'] != 0:
                raise ValueError('Evaluation changed baseline weights')
        elif algorithm != 'Greedy':
            if not report['actor_changed'] or report['optimizer_updates'] <= 0:
                raise ValueError('Baseline did not learn')
            if algorithm in ('MAPPO','IPPO','COMA') and not report['critic_changed']:
                raise ValueError('PPO critic did not learn')
            if algorithm == 'QMIX' and not report['mixer_changed']:
                raise ValueError('QMIX mixer did not learn')
        if not math.isfinite(report['update_seconds']):
            raise ValueError('Invalid compute timing')
    elif evaluation and (not report['actor_unchanged'] or not report['critic_unchanged'] or report['optimizer_updates']):
        raise ValueError('Evaluation changed TG weights')
    generated = sum(r['generated_packets'] for r in rows)
    hops = [v for r in rows for v in r['success_hops']]
    delays = [v for r in rows for v in r['success_delays']]
    return dict(delivery=sum(r['delivered_packets'] for r in rows)/generated,
                delivered=sum(r['delivered_packets'] for r in rows), generated=generated,
                success_hops_mean=sum(hops)/len(hops) if hops else None,
                success_delay_mean=sum(delays)/len(delays) if delays else None,
                loop_transfers=sum(r['loop_transfers'] for r in rows),
                report_sha256=digest(output/('logs/baseline_report.json' if baseline else 'logs/TG_policy_report.json')))


def run(job_file, evaluate=None):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('Registered GPU is unavailable')
    job = read(job_file)
    folder = Path(job['work'])
    if evaluate:
        folder = Path(evaluate['output'])
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/'moves_5'
    episodes = evaluate['episodes'] if evaluate else job['episodes']
    seed = evaluate['scene'] if evaluate else job['seed']
    import run_algorithm
    import buoy_network.agent as native
    from buoy_network.main import main
    from buoy_network.wm_runtime import tensor_digest
    from buoy_network.thesis_baselines import ALGORITHMS, BaselineLearner, tensor_hash
    baseline = job['algorithm'] in ALGORITHMS
    cls = type('SelectedBaseline', (BaselineLearner,), {'algorithm':job['algorithm']}) if baseline else native.DeepQNetwork
    progress = dict(status='running', algorithm=job['algorithm'], seed=job['seed'], trained_episodes=0, points=[])
    seen = set()
    start = time.perf_counter()
    def checkpoint(model, n):
        if evaluate or n in seen or n not in job['points']:
            return
        cp = folder/'convergence'/('e%04d'%n)
        save_point(model, cp, save)
        weights = cp/'checkpoints'/('baseline.pt' if baseline else 'tg_policy_latest.pt')
        before = (tensor_hash(model.actor),tensor_hash(model.critic)) if baseline else (tensor_digest(model.policy.actor.state_dict()),tensor_digest(model.policy.critic.state_dict()))
        point = dict(episode=n, training_steps=n*80, checkpoint=str(weights), checkpoint_sha256=digest(weights), evaluations=[])
        for scene in job['scenes']:
            spec = dict(output=str(cp/('scene_'+str(scene))), scene=scene, episodes=job['eval_episodes'], checkpoint=str(weights))
            cmd = [sys.executable,'-s','-u',str(Path(__file__).resolve()),'--job',str(Path(job_file).resolve()),'--evaluate',json.dumps(spec)]
            with (cp/('eval_'+str(scene)+'.log')).open('xb') as stream:
                proc = subprocess.run(cmd, stdout=stream,stderr=subprocess.STDOUT,timeout=900,cwd=str(ROOT))
            if proc.returncode:
                raise ValueError('Frozen evaluation failed: '+str(cp/('eval_'+str(scene)+'.log')))
            metrics = inspect(Path(spec['output'])/'moves_5',job['algorithm'],job['eval_episodes'],True)
            point['evaluations'].append(dict(scene=scene,**metrics))
        after = (tensor_hash(model.actor),tensor_hash(model.critic)) if baseline else (tensor_digest(model.policy.actor.state_dict()),tensor_digest(model.policy.critic.state_dict()))
        if before != after:
            raise ValueError('Frozen evaluation changed parent weights')
        seen.add(n)
        progress['points'].append(point)
        progress['trained_episodes'] = n
        atomic(folder/'progress.json',progress)
        print('POINT {} seed{} e{} delivery {:.4f}'.format(job['algorithm'],job['seed'],n,
              sum(r['delivery'] for r in point['evaluations'])/len(point['evaluations'])),flush=True)
    warmup, save = cls.warmup, cls.save_policy
    def observed_warmup(model):
        warmup(model)
        checkpoint(model, model.episodes if baseline else model.profile['ppo_updates'])
    def observed_save(model,directory):
        save(model,directory)
        checkpoint(model,model.episodes if baseline else model.profile['ppo_updates'])
    args = ['--num_episodes',str(episodes),'--num_env_steps',str(episodes*80),'--episode_length','80',
            '--seed',str(seed),'--scenario_seed',str(seed),'--tg_device','cuda','--tg_reward_version','task_v2',
            '--num_moving_buoys','5','--ppo_epoch','1','--num_mini_batch','2','--output_root',str(folder)] + job.get('extra_args',[])
    if evaluate:
        args += ['--tg_evaluate','true','--tg_policy_checkpoint',evaluate['checkpoint']]
    atomic(folder/'progress.json',progress)
    try:
        with patch.object(native,'DeepQNetwork',cls), patch.object(cls,'warmup',observed_warmup), patch.object(cls,'save_policy',observed_save):
            main(run_algorithm.with_tg_mappo_defaults(args))
        metrics = inspect(output,job['algorithm'],episodes,bool(evaluate))
        if not evaluate and seen != set(job['points']):
            raise ValueError('Missing convergence checkpoints')
        progress.update(status='completed',trained_episodes=0 if evaluate else episodes,wall_seconds=time.perf_counter()-start,metrics=metrics)
        atomic(folder/'progress.json',progress)
    except Exception as error:
        progress.update(status='failed',error=repr(error))
        atomic(folder/'progress.json',progress)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--evaluate')
    args = parser.parse_args()
    run(args.job, json.loads(args.evaluate) if args.evaluate else None)
