"""Audit and summarize exact seed22 optimizer observations; no new training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def summarize(rows):
    out = {'count': len(rows)}
    if rows:
        for name in ('advantage_normalized', 'advantage_raw', 'reward_progress', 'packet_progress', 'other_deliveries'):
            out[name+'_mean'] = float(np.mean([x[name] for x in rows]))
        out['positive_advantage'] = sum(x['advantage_normalized'] > 0 for x in rows)
        out['negative_own_PM'] = sum(x['packet_progress'] < 0 for x in rows)
        out['negative_own_PM_positive_advantage'] = sum(x['packet_progress'] < 0 and x['advantage_normalized'] > 0 for x in rows)
        changed = [x for x in rows if 'p_selected_after' in x]
        out['probability_observed_rows'] = len(changed)
        out['selected_probability_increased'] = sum(x['p_selected_after'] > x['p_selected_before'] for x in changed)
        out['selected_probability_change_mean'] = float(np.mean([x['p_selected_after']-x['p_selected_before'] for x in changed])) if changed else None
    return out


def audit(root):
    import torch
    m = read(root/'manifest.json')
    require(m['status'] == 'completed' and len(m['jobs']) == 6, 'Replay suite incomplete')
    project = Path(__file__).resolve().parents[1]
    for name, digest in m['tool_sha256'].items():
        require(sha(project/name) == digest, 'Observer source changed')
    all_reports = []
    for job in m['jobs']:
        p = Path(job['output'])/'report.json'
        require(job['state'] == 'completed' and job['exit_code'] == 0 and sha(p) == job['report_sha256'],
                'Replay job or report audit failed')
        all_reports.append(read(p))
    new_steps = sum(x['environment_steps'] for x in all_reports)
    new_updates = sum(x['optimizer_steps'] for x in all_reports)
    require(new_steps == 5280 and new_updates == 132, 'Suite total budget mismatch')
    results = {}
    for group in ('A', 'task_return', 'alpha01'):
        out, prefix = root/('full_'+group), root/('prefix_'+group)
        report = read(out/'report.json')
        job = next(x for x in m['jobs'] if x['tag'] == 'full_'+group)
        require(job['state'] == 'completed' and sha(out/'report.json') == job['report_sha256'], 'Report fingerprint mismatch')
        require(report['exact_final_actor_critic'] and report['exact_training_updates'] and report['full_replay'], 'Exact replay gate failed')
        require(report['environment_steps'] == 1600 and report['optimizer_steps'] == 40, 'Training budget mismatch')
        for name, fingerprint in (('training_credit.jsonl', 'training_credit_sha256'), ('updates.jsonl', 'updates_sha256')):
            require(sha(out/name) == report[fingerprint], 'Observer artifact changed')
            require((out/name).read_bytes().startswith((prefix/name).read_bytes()), 'Prefix/full observer rows differ')
        reference = Path(report['reference'])
        require(sha(reference/'checkpoints/tg_policy_latest.pt') == report['reference_checkpoint_sha256'], 'Original model changed')
        original = torch.load(str(reference/'checkpoints/tg_policy_latest.pt'), map_location='cpu')
        repeated = torch.load(str(out/'replay/moves_5/checkpoints/tg_policy_latest.pt'), map_location='cpu')
        for network in ('actor', 'critic'):
            require(set(original[network]) == set(repeated[network]) and
                    all(torch.equal(original[network][k], repeated[network][k]) for k in original[network]),
                    'Independent final tensor audit failed')
        for name, fingerprint in report['exact_logs'].items():
            require(sha(out/'replay/moves_5/logs'/name) == fingerprint == sha(reference/'logs'/name), 'Original log audit failed')
        updates = lines(out/'updates.jsonl')
        rows = lines(out/'training_credit.jsonl')
        require(len(updates) == 40 and len(rows) == report['active_rows'], 'Observer row count mismatch')
        trajectory = [dict(optimizer_step=0, episode=-1, **report['initial_probe'])]
        flips, argmax_changes = [], []
        previous = report['initial_probe']
        for u in updates:
            require(u['probe_before'] == previous, 'Fixed probe update chain discontinuity')
            after = u['probe_after']
            trajectory.append(dict(optimizer_step=u['optimizer_step'], episode=u['episode'], **after))
            if (previous['gap17_minus19'] > 0) != (after['gap17_minus19'] > 0):
                flips.append(u)
            if previous['argmax'] != after['argmax']:
                argmax_changes.append(u)
            previous = after
        with np.load(str(Path(m['diagnosis'])/'same_inputs/probabilities.npz')) as frozen:
            expected = frozen[group][0, 13]
            error = max(abs(previous['p17']-float(expected[17])), abs(previous['p19']-float(expected[19])))
            require(error < 1e-6 and previous['argmax'] == int(expected.argmax()), 'Final ranking differs from frozen diagnosis')
        target = [x for x in rows if x['sender'] == 13 and x['legal17_and19']]
        packet_at_13 = [x for x in rows if x['packet_id'] == 13 and x['sender'] == 13]
        last = argmax_changes[-1] if argmax_changes else None
        results[group] = {'exact_actor_critic_loss_rewards_physics': True,
            'active_rows': len(rows), 'final_probe': trajectory[-1], 'trajectory': trajectory,
            'frozen_probe_probability_error': error,
            'pair_flip_steps': [x['optimizer_step'] for x in flips],
            'argmax_changes': [{'optimizer_step': x['optimizer_step'], 'episode': x['episode'],
                               'before': x['probe_before']['argmax'], 'after': x['probe_after']['argmax']} for x in argmax_changes],
            'last_pair_flip': flips[-1] if flips else None, 'last_argmax_change': last,
            'last_argmax_rollout_node13': [x for x in target if last is not None and x['episode'] == last['episode']],
            'node13_both_legal': {str(a): summarize([x for x in target if x['action'] == a]) for a in (17, 19)},
            'packet13_at_node13': {str(a): summarize([x for x in packet_at_13 if x['action'] == a]) for a in (17, 19, 25)},
            'node13_revisits': summarize([x for x in rows if x['sender'] == 13 and x['ack'] is True and x['revisit']]),
            'per_rollout': [{'episode': e, 'node13_17': summarize([x for x in target if x['episode'] == e and x['action'] == 17]),
                             'node13_19': summarize([x for x in target if x['episode'] == e and x['action'] == 19])} for e in range(20)],
            'fingerprints': {name: sha(out/name) for name in ('report.json', 'training_credit.jsonl', 'updates.jsonl')}}
    initial = [read(root/('full_'+g)/'report.json')['initial_probe'] for g in results]
    require(all(x == initial[0] for x in initial), 'Initial Actor probes differ')
    return {'success': True, 'root': str(root), 'manifest_sha256': sha(root/'manifest.json'),
        'groups': results, 'new_environment_steps': new_steps, 'new_optimizer_steps': new_updates,
        'budget_scope': '480 prefix slots + 4800 exact repeated training slots; 12 prefix + 120 full optimizer steps.',
        'algorithm_changed': False, 'new_efficacy_evidence': False,
        'limitations': ['Fixed development observation: identifies timing of learned ranking changes, not their sole cause.',
            'Shared Actor/Adam updates use all active nodes, entropy, clipping, and optimizer history.',
            'Local PM changes are observational model scores, not counterfactual causal action values.']}


def plot(report, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    choices = sorted({d['argmax'] for g in report['groups'].values() for d in g['trajectory']})
    for group, label, color in (('A', 'TG', '#547b9a'), ('task_return', 'PM alpha=1', '#4d9871'),
                                 ('alpha01', 'PM alpha=0.1', '#c37245')):
        data = report['groups'][group]['trajectory']
        x = [d['optimizer_step'] for d in data]
        axes[0].plot(x, [100*d['gap17_minus19'] for d in data], label=label, color=color, marker='.', ms=4)
        axes[1].step(x, [choices.index(d['argmax']) for d in data], where='post', label=label, color=color, marker='.', ms=4)
    axes[0].axhline(0, color='gray', lw=.8)
    axes[0].set_ylabel('P(17) - P(19), percentage points')
    axes[0].set_title('Seed22: identical node13 evaluation input after each actual Adam step')
    axes[0].legend()
    axes[1].set_ylabel('Greedy next hop'); axes[1].set_xlabel('Actor optimizer step (2 steps / rollout)')
    axes[1].set_yticks(range(len(choices))); axes[1].set_yticklabels(choices)
    axes[1].set_ylim(-.3, len(choices)-.7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    folder = root/'figures'; folder.mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(str(folder/('seed22_optimizer_rankings.'+ext)), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root'); args = p.parse_args()
    root = Path(args.root).resolve()
    result = audit(root); plot(result, root)
    result['auditor_sha256'] = sha(Path(__file__))
    result['figures'] = {str(p.relative_to(root)): sha(p) for p in (root/'figures').iterdir()}
    (root/'acceptance.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for name, r in result['groups'].items():
        print(name, 'final', r['final_probe'], 'last argmax', r['argmax_changes'][-1:] )
    print('PASS '+str(root/'acceptance.json'))


if __name__ == '__main__':
    main()
