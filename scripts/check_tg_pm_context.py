"""One predeclared PM-Critic context candidate and equal-capacity constant control."""
import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from buoy_network.ablation_suite import read, atomic, digest, require, snapshot
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_task_return import physical_return


def setarg(command, key, value):
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend([key, str(value)])


def main():
    import numpy as np
    import torch
    reference = ROOT/'results/tg_pm_task_return/pilot_20261105_140555_016580'
    low = ROOT/'results/tg_pm_strength/20261105_154758_858806'
    old, small = read(reference/'manifest.json'), read(low/'manifest.json')
    require(old['status'] == small['status'] == 'completed', 'Reference tasks incomplete')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume')
    args = parser.parse_args()
    root = Path(args.resume).resolve() if args.resume else ROOT/'results/tg_pm_context'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.resume:
        require(ROOT/'results/tg_pm_context' in root.parents, 'Resume path must stay in the context experiment directory')
        m = read(root/'manifest.json')
        require(m['status'] == 'failed', 'Resume expects a failed audit, not an active or completed run')
        m.setdefault('audit_failures', []).append({'error': m.pop('error'), 'resumed': datetime.now().isoformat()})
        m['status'] = 'running'
        m.setdefault('resume_controllers', []).append({'sha256': digest(Path(__file__)), 'date': datetime.now().isoformat()})
    else:
        root.mkdir(parents=True, exist_ok=False)
        m = {'status': 'running', 'root': str(root), 'source': snapshot(root), 'jobs': [],
         'reference_manifests': {str(p): digest(p) for p in (reference/'manifest.json', low/'manifest.json')},
         'protocol': {'seeds': [11, 22, 33], 'scenes': [8001, 8002, 8003], 'episodes': 20,
            'alpha': .1, 'candidate': 'potential', 'capacity_control': 'constant', 'extra_critic_parameters': 1,
            'gate': 'Mean delivery/base return exceed A, old alpha0.1 and constant; no seed worse than A; >=2 seed wins vs A.',
            'budget': '6 default prefix checks + 4 enabled smoke jobs + 24 full train/eval jobs; 14080 slots',
            'no_new_blind_or_independent_seeds': True}}
    for path, fingerprint in m['source']['files_sha256'].items():
        require(digest(root/'source_snapshot'/path) == fingerprint, 'Runtime snapshot changed')
    atomic(root/'manifest.json', m)
    print('OUTPUT '+str(root), flush=True)

    def run(ref, tag, mode, episodes, kind, checkpoint=None, compat=False):
        command = list(ref['command'])
        command[0] = sys.executable
        command[1] = str(root/'source_snapshot/scripts/pm_value_worker.py')
        for key, value in (('--snapshot', root/'source_snapshot'), ('--output_root', root/tag),
            ('--pm_critic_context', mode), ('--num_episodes', episodes), ('--num_env_steps', 80*episodes)):
            setarg(command, key, value)
        if checkpoint is not None:
            setarg(command, '--tg_policy_checkpoint', checkpoint)
        job = next((j for j in m['jobs'] if j['tag'] == tag), None)
        if job is not None:
            require(job['command'] == command, 'Resume command changed')
            require(job['exit_code'] == 0 and (Path(job['output'])/'logs/TG_policy_report.json').is_file(),
                    'Cannot reuse a failed training process')
            if job['state'] == 'completed':
                require(digest(Path(job['output'])/'logs/TG_policy_report.json') == job['report_sha256'], 'Completed report changed')
                report = read(Path(job['output'])/'logs/TG_policy_report.json')
                return Path(report['checkpoint']) if kind == 'training' else None
            job['audit_reused_completed_process'] = True
            job['state'] = 'running'
        else:
            job = {'tag': tag, 'seed': ref['seed'], 'scene': ref['scene'], 'kind': kind,
            'mode': mode, 'episodes': episodes, 'command': command, 'state': 'running',
            'output': str(root/tag/'runs/case/moves_5'), 'started': datetime.now().isoformat()}
            m['jobs'].append(job); atomic(root/'manifest.json', m)
            print('START '+tag, flush=True)
            with (root/(tag+'.log')).open('xb') as log:
                result = subprocess.run(command, cwd=str(root/'source_snapshot'), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            job['exit_code'] = result.returncode
            require(result.returncode == 0, 'Task failed: '+tag)
        out = Path(job['output']); report = read(out/'logs/TG_policy_report.json')
        actual = read(out/'logs/TG_MAPPO_environment.json')['config']
        previous = read(Path(ref['output'])/'logs/TG_MAPPO_environment.json')['config']
        previous.setdefault('pm_reliability_report', '')
        ignored = {'output_root', 'tg_policy_checkpoint', 'num_episodes', 'num_env_steps',
                   'effective_num_env_steps', 'training_budget_source', 'pm_critic_context',
                   'requested_num_env_steps', 'requested_num_episodes'}
        # Names of old budget annotations are version-specific; all non-budget options must match.
        ignored |= {k for k in actual if 'budget' in k}
        ignored |= {k for k in previous if 'budget' in k}
        require({k: v for k, v in actual.items() if k not in ignored} ==
                {k: v for k, v in previous.items() if k not in ignored}, 'Another configuration variable changed')
        require(actual['pm_critic_context'] == report['pm_critic_context'] == mode, 'Wrong context mode')
        require(report['profile']['collect_calls'] == 80*episodes, 'Wrong actual environment budget')
        if report.get('pm') is not None:
            require(report['pm']['frozen'] and report['pm']['checkpoint_sha256'] ==
                    read(Path(ref['output'])/'logs/TG_policy_report.json')['pm']['checkpoint_sha256'], 'PM changed')
        if kind == 'training':
            require(len(report['training_updates']) == episodes, 'Wrong rollout count')
            for i, u in enumerate(report['training_updates'], 1):
                require(u['actor_optimizer_steps'] == u['critic_head_optimizer_steps'] == 2*i and
                        u['critic_head_in_optimizer'], 'Wrong PPO budget/optimizer registration')
                require(all(math.isfinite(v) for v in u.values() if isinstance(v, (int, float))), 'Nonfinite training')
            require(digest(report['checkpoint']) == report['checkpoint_sha256'], 'Checkpoint fingerprint mismatch')
            if mode != 'disabled':
                payload = torch.load(report['checkpoint'], map_location='cpu')
                job['learned_context_weight'] = float(payload['critic']['v_out.weight'][0, -1])
                require(job['learned_context_weight'] != 0., 'Context weight did not learn')
                for ep in range(episodes):
                    with np.load(str(out.parent/'value_audit'/('episode_'+str(ep)+'.npz'))) as cache:
                        require(np.array_equal(cache['critic_progress'], cache['phi'].astype(np.float32)), 'Wrong time in PPO context')
        else:
            require(report['optimizer_updates'] == 0 and report['actor_unchanged'] and report['critic_unchanged']
                    and report['checkpoint_sha256'] == digest(checkpoint), 'Frozen evaluation changed/wrong weights')
        physics = [json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        audit_reward_csv(out/'logs', report['reward_contract'])
        if compat:
            original = Path(ref['output'])
            for name in ('TG_physical_metrics.jsonl', 'TG_reward_components.csv', 'TG_PM_rewards.csv'):
                if not (original/'logs'/name).is_file():
                    continue
                n = episodes if name.endswith('.jsonl') else episodes*80+1
                expected = (original/'logs'/name).read_bytes().splitlines(keepends=True)[:n]
                require((out/'logs'/name).read_bytes() == b''.join(expected), 'Default behavior changed: '+name)
            require(report['training_updates'] == read(original/'logs/TG_policy_report.json')['training_updates'][:episodes],
                    'Default PPO losses changed')
        job.update(state='completed', finished=datetime.now().isoformat(), report_sha256=digest(out/'logs/TG_policy_report.json'),
            delivered=sum(p['delivered_packets'] for p in physics), generated=sum(p['generated_packets'] for p in physics),
            base_return=physical_return(physics, report['reward_contract']), compatibility_exact=compat,
            physical_sha256=digest(out/'logs/TG_physical_metrics.jsonl'))
        atomic(root/'manifest.json', m)
        print('DONE {} {}/{}'.format(tag, job['delivered'], job['generated']), flush=True)
        return Path(report['checkpoint']) if kind == 'training' else None

    try:
        for seed in (11, 22, 33):
            for group, jobs in (('A', old['jobs']), ('alpha01', small['jobs'])):
                ref = next(j for j in jobs if j['seed'] == seed and j['kind'] == 'training' and
                           (group != 'A' or j['group'] == 'A'))
                run(ref, 'compat_{}_{}'.format(group, seed), 'disabled', 2, 'training', compat=True)
        for mode in ('constant', 'potential'):
            train = next(j for j in small['jobs'] if j['seed'] == 22 and j['kind'] == 'training')
            checkpoint = run(train, 'smoke_'+mode+'_train', mode, 2, 'training')
            evaluation = next(j for j in small['jobs'] if j['seed'] == 22 and j['kind'] == 'evaluation' and j['scene'] == 8001)
            run(evaluation, 'smoke_'+mode+'_eval', mode, 2, 'evaluation', checkpoint)
        for mode in ('constant', 'potential'):
            for seed in (11, 22, 33):
                train = next(j for j in small['jobs'] if j['seed'] == seed and j['kind'] == 'training')
                checkpoint = run(train, '{}_{}_train'.format(mode, seed), mode, 20, 'training')
                for scene in (8001, 8002, 8003):
                    ref = next(j for j in small['jobs'] if j['seed'] == seed and j['kind'] == 'evaluation' and j['scene'] == scene)
                    run(ref, '{}_{}_eval_{}'.format(mode, seed, scene), mode, 2, 'evaluation', checkpoint)
        require(all(digest(root/'source_snapshot'/p) == h for p, h in m['source']['files_sha256'].items()), 'Source snapshot changed')
        m['status'] = 'completed'
    except Exception as exc:
        m['status'], m['error'] = 'failed', repr(exc)
        if m['jobs'] and m['jobs'][-1]['state'] == 'running':
            m['jobs'][-1]['state'] = 'failed'
        atomic(root/'manifest.json', m); raise
    atomic(root/'manifest.json', m)
    print('COMPLETED '+str(root/'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
