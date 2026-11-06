"""Independent PM-Critic context acceptance and development-only comparison."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from buoy_network.ablation_suite import read, atomic, digest, require
from buoy_network.task_reward_audit import audit_reward_csv
from analyze_tg_pm_strength import cache_audit
from analyze_tg_pm_task_return import physical_return


def audit(root):
    import numpy as np
    import torch
    m = read(root/'manifest.json')
    require(m['status'] == 'completed' and len(m['jobs']) == 34, 'Context suite incomplete')
    for p, h in m['reference_manifests'].items():
        require(digest(p) == h, 'Historical manifest changed')
    for p, h in m['source']['files_sha256'].items():
        require(digest(root/'source_snapshot'/p) == h, 'Runtime archive changed')
    artifacts, cache_reports = [], []
    for job in m['jobs']:
        out = Path(job['output']); r = read(out/'logs/TG_policy_report.json')
        require(job['state'] == 'completed' and job['exit_code'] == 0 and
                digest(out/'logs/TG_policy_report.json') == job['report_sha256'] and
                digest(out/'logs/TG_physical_metrics.jsonl') == job['physical_sha256'], 'Artifact or process failed')
        require(r['pm_critic_context'] == job['mode'] and r['profile']['collect_calls'] == job['episodes']*80, 'Context/budget mismatch')
        audit_reward_csv(out/'logs', r['reward_contract'])
        if job['kind'] == 'training':
            require(digest(r['checkpoint']) == r['checkpoint_sha256'], 'Model changed')
            require(len(r['training_updates']) == job['episodes'], 'Missing updates')
            for i, u in enumerate(r['training_updates'], 1):
                require(u['actor_optimizer_steps'] == u['critic_head_optimizer_steps'] == 2*i and
                        u['actor_lr'] == .0005 and u['critic_lr'] == .0003, 'Training budget/LR mismatch')
            if job['mode'] != 'disabled':
                require(r['resources']['actor_trainable'] == 21956 and r['resources']['critic_trainable'] == 18616,
                        'Capacity control mismatch')
                for ep in range(job['episodes']):
                    with np.load(str(out.parent/'value_audit'/('episode_'+str(ep)+'.npz'))) as data:
                        require(np.array_equal(data['critic_progress'], data['phi'].astype(np.float32)), 'Decision-time cache mismatch')
                cache_reports.append({'tag': job['tag'], 'rollouts': cache_audit(out, .1, job['episodes'])})
        else:
            require(r['optimizer_updates'] == 0 and r['actor_unchanged'] and r['critic_unchanged'], 'Evaluation updated weights')
            require(digest(r['source']) == r['checkpoint_sha256'], 'Evaluation model changed')
        artifacts.append({'tag': job['tag'], 'report_sha256': job['report_sha256'], 'physical_sha256': job['physical_sha256']})
    # Stronger default compatibility: the previously saved seed22 two-rollout parameters.
    for group in ('A', 'alpha01'):
        job = next(j for j in m['jobs'] if j['tag'] == 'compat_'+group+'_22')
        prior = ROOT/'results/tg_pm_update_replay/20261105_211844_626825'/('prefix_'+group)/'replay/moves_5'
        a = torch.load(str(Path(job['output'])/'checkpoints/tg_policy_latest.pt'), map_location='cpu')
        b = torch.load(str(prior/'checkpoints/tg_policy_latest.pt'), map_location='cpu')
        require(all(torch.equal(a[n][k], b[n][k]) for n in ('actor', 'critic') for k in b[n]), 'Default prefix parameters changed')
    original = read(ROOT/'results/tg_pm_task_return/pilot_20261105_140555_016580/manifest.json')
    low = read(ROOT/'results/tg_pm_strength/20261105_154758_858806/manifest.json')
    groups = {'A': [j for j in original['jobs'] if j['group'] == 'A' and j['kind'] == 'evaluation'],
              'alpha01': [j for j in low['jobs'] if j['kind'] == 'evaluation']}
    for mode in ('constant', 'potential'):
        groups[mode] = [j for j in m['jobs'] if j['mode'] == mode and j['kind'] == 'evaluation' and not j['tag'].startswith('smoke')]
    rows, per_seed = [], {}
    for group, jobs in groups.items():
        require(len(jobs) == 9, 'Missing scenes/seeds')
        for j in jobs:
            out = Path(j['output']); r = read(out/'logs/TG_policy_report.json')
            if 'report_sha256' in j:
                require(digest(out/'logs/TG_policy_report.json') == j['report_sha256'], 'Reference report changed')
            require(r['optimizer_updates'] == 0 and r['actor_unchanged'] and r['critic_unchanged'], 'Reference evaluation is not frozen')
            metrics = [json.loads(x) for x in (out/'logs/TG_physical_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            audit_reward_csv(out/'logs', r['reward_contract'])
            delivered, generated = sum(x['delivered_packets'] for x in metrics), sum(x['generated_packets'] for x in metrics)
            rows.append({'group': group, 'seed': j['seed'], 'scene': j['scene'], 'delivered': delivered, 'generated': generated,
                         'delivery': delivered/generated, 'base_return': physical_return(metrics, r['reward_contract'])})
        for seed in (11, 22, 33):
            selected = [x for x in rows if x['group'] == group and x['seed'] == seed]
            per_seed[(group, seed)] = {'group': group, 'seed': seed,
                'delivery': statistics.mean(x['delivery'] for x in selected),
                'base_return': statistics.mean(x['base_return'] for x in selected)}
    summary = {}
    for group in groups:
        values = [per_seed[(group, s)] for s in (11, 22, 33)]
        summary[group] = {'delivery_mean': statistics.mean(x['delivery'] for x in values),
            'delivery_sample_SD': statistics.stdev(x['delivery'] for x in values),
            'base_return_mean': statistics.mean(x['base_return'] for x in values), 'per_seed': values}
    comparisons = ('A', 'alpha01', 'constant')
    higher_mean = all(summary['potential']['delivery_mean'] > summary[g]['delivery_mean'] for g in comparisons)
    higher_return = all(summary['potential']['base_return_mean'] > summary[g]['base_return_mean'] for g in comparisons)
    no_regression = all(per_seed[('potential', s)]['delivery'] >= per_seed[('A', s)]['delivery'] for s in (11, 22, 33))
    wins = sum(per_seed[('potential', s)]['delivery'] > per_seed[('A', s)]['delivery'] for s in (11, 22, 33))
    return {'engineering_passed': True, 'root': str(root), 'manifest_sha256': digest(root/'manifest.json'),
        'summary': summary, 'scene_rows': rows, 'artifacts': artifacts, 'GAE_cache_audit': cache_reports,
        'development_gate': {'passed': higher_mean and higher_return and no_regression and wins >= 2,
            'higher_mean_delivery': higher_mean, 'higher_mean_base_return': higher_return,
            'no_seed_worse_than_A': no_regression, 'wins_vs_A': wins},
        'new_training_steps': sum(j['episodes']*80 for j in m['jobs'] if j['kind'] == 'training'),
        'new_evaluation_steps': sum(j['episodes']*80 for j in m['jobs'] if j['kind'] == 'evaluation'),
        'formal_efficacy_proven': False, 'default_changed': False,
        'limitations': ['Three known development training seeds; scenes already reused; no significance or blind generalization.',
                        'Scalar Phi is partial PM context; it does not make the full history Markov or provide causal per-agent credit.']}


def plot(report, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['A', 'alpha01', 'constant', 'potential']
    labels = ['TG', 'PM 0.1', '+ constant', '+ Phi']
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for i, seed in enumerate((11, 22, 33)):
        y = [100*next(s['delivery'] for s in report['summary'][g]['per_seed'] if s['seed'] == seed) for g in groups]
        ax.plot(range(4), y, marker='o', label='Train seed '+str(seed))
    means = [100*report['summary'][g]['delivery_mean'] for g in groups]
    ax.plot(range(4), means, color='black', ls='--', marker='s', label='Three-seed mean')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('Frozen delivery rate (%)'); ax.grid(alpha=.2); ax.legend()
    ax.set_title('PM Critic context: same budget; known development seeds')
    fig.tight_layout(); folder = root/'figures'; folder.mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(str(folder/('pm_critic_context.'+ext)), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root'); args = p.parse_args()
    root = Path(args.root).resolve(); result = audit(root); plot(result, root)
    result['auditor_sha256'] = digest(Path(__file__))
    result['figures'] = {str(p.relative_to(root)): digest(p) for p in (root/'figures').iterdir()}
    atomic(root/'acceptance.json', result)
    print(json.dumps(result['summary'], indent=2)); print(result['development_gate'])
    print('REPORT '+str(root/'acceptance.json'))


if __name__ == '__main__':
    main()
