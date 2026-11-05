"""Observe original seed-22 PPO updates; require exact archived-training replay.

No losses, actions, targets, or training hyperparameters are changed. The optional
short prefix is a compatibility check, not a newly trained efficacy candidate.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('obs', 'node_obs', 'adj', 'agent_id', 'rnn_states', 'masks', 'available_actions')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ranking(probabilities, first=17, second=19):
    import numpy as np
    p = np.asarray(probabilities)
    require(p.ndim == 1 and np.isfinite(p).all() and min(first, second) >= 0
            and max(first, second) < len(p), 'Invalid ranking probabilities')
    return {'p17': float(p[first]), 'p19': float(p[second]),
            'gap17_minus19': float(p[first]-p[second]), 'argmax': int(p.argmax())}


def probe(actor, capture, arrays):
    """No-grad deterministic scores, preserving mode, RNG, and training hook data."""
    import numpy as np
    import torch
    state, last = actor.training, capture.last
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    pieces = []
    try:
        actor.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for start in range(0, len(arrays[0]), 100):
                x = [v[start:start+100] for v in arrays]
                actor(*x[:6], available_actions=x[6], deterministic=True)
                pieces.append(capture.last.copy())
    finally:
        actor.train(state)
        capture.last = last
    after = np.random.get_state()
    require(random.getstate() == python_rng and numpy_rng[0] == after[0]
            and np.array_equal(numpy_rng[1], after[1]) and numpy_rng[2:] == after[2:]
            and torch.equal(torch_rng, torch.get_rng_state()), 'Probability observation consumed RNG')
    return np.concatenate(pieces)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis', required=True)
    p.add_argument('--group', choices=('A', 'task_return', 'alpha01'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', type=int, help='Optional reference prefix compatibility check')
    args = p.parse_args()
    diagnosis, output = Path(args.diagnosis).resolve(), Path(args.output).resolve()
    manifest = read(diagnosis/'manifest.json')
    view = read(diagnosis/'view/manifest.json')
    snapshot = Path(manifest['source_snapshot'])
    for name, checksum in view['source']['files_sha256'].items():
        require(sha(snapshot/name) == checksum, 'Archived source changed: '+name)
    for path, checksum in manifest['reference_manifests'].items():
        require(sha(path) == checksum, 'Reference manifest changed')
    job = next(j for j in view['jobs'] if j['group'] == args.group and j['kind'] == 'training')
    reference = Path(job['output'])
    report = read(reference/'logs/TG_policy_report.json')
    checkpoint = reference/'checkpoints/tg_policy_latest.pt'
    reference_sha = sha(checkpoint)
    require(reference_sha == report['checkpoint_sha256'], 'Reference checkpoint changed')
    config = read(reference/'logs/TG_MAPPO_environment.json')['config']
    require(config['seed'] == 22 and config['num_episodes'] == 20 and config['episode_length'] == 80
            and not config['use_wm_actor'] and not config['use_recurrent_policy']
            and not config['use_linear_lr_decay'] and not config['tg_evaluate']
            and config['tg_critic_time_mode'] == 'disabled', 'Unsupported replay configuration')
    episodes = args.episodes if args.episodes is not None else config['num_episodes']
    require(1 <= episodes <= config['num_episodes'], 'Invalid prefix budget')
    full = episodes == config['num_episodes']
    config = dict(config, num_episodes=episodes, num_env_steps=episodes*config['episode_length'])
    # Load the safe training hook before selecting the immutable runtime archive.
    from diagnose_tg_credit import RolloutCapture
    sys.path[:0] = [str(snapshot), str(snapshot/'src'), str(snapshot/'scripts')]
    import numpy as np
    import torch
    import diagnose_tg_policy as d
    from buoy_network.progress_runtime import FrozenPMRuntime
    from buoy_network.pm_return import expected_task_return
    helper_path = ROOT/'src/buoy_network/credit_diagnostics.py'
    spec = importlib.util.spec_from_file_location('credit_helper', str(helper_path))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    output.mkdir(parents=True, exist_ok=False)
    env, learner = d.build(config, output/'replay')
    capture = RolloutCapture(learner.policy.actor)
    fixed_path = diagnosis/'alpha01_greedy/actor_inputs.npz'
    with np.load(str(fixed_path)) as cached:
        fixed = [cached[k][0].copy() for k in FIELDS]
    initial_probe = ranking(probe(learner.policy.actor, capture, fixed)[13])
    context = {'episode': -1, 'rows': {}, 'episode_rows': [], 'updates': 0}
    packet_values, rollouts, updates = {}, [], []
    row_count = [0]
    stream = (output/'training_credit.jsonl').open('x', encoding='utf-8')
    update_stream = (output/'updates.jsonl').open('x', encoding='utf-8')
    potential = FrozenPMRuntime.potential

    def observed_potential(runtime, e, step):
        keys = sorted(p.id for p in e.packets if not p.if_sink)
        captured, forward = [], runtime.model.step
        def model_step(*a, **kw):
            logits, following = forward(*a, **kw)
            captured.append(logits.detach().clone())
            return logits, following
        with patch.object(runtime.model, 'step', model_step):
            phi = potential(runtime, e, step)
        values = {}
        if captured:
            with torch.no_grad():
                result = expected_task_return(captured[0], torch.full((len(keys),), runtime.horizon-step,
                                              dtype=torch.long), runtime.horizon, runtime.gamma, runtime.task_contract)
            values = dict(zip(keys, result['task_return'].tolist()))
        require(abs(sum(values.values())/max(runtime.initial_packets, 1)-phi) < 1e-7,
                'Per-packet PM observation differs from team Phi')
        packet_values[(runtime.episode, step)] = values
        return phi

    class Metrics(d.PacketMetrics):
        def __init__(self, packets):
            super().__init__(packets)
            context['episode'] += 1
            context['episode_rows'] = []
            context['metrics'] = self

        def attempt(self, packet_id, sender, receiver, success):
            context['rows'][sender].update(ack=bool(success),
                revisit=receiver in self.packets[packet_id]['route'])
            super().attempt(packet_id, sender, receiver, success)

    collect = learner.collect
    def observed_collect(step):
        result = collect(step)
        b, rows, service = learner.buffer, {}, {}
        for packet in sorted(env.packets, key=lambda x: x.queue_order):
            if not packet.if_sink:
                service.setdefault(packet.agent_id, packet)
        for node in np.flatnonzero(b.active_masks[step, 0, :, 0] > 0):
            node = int(node)
            packet = service[node]
            action = int(result[1][0, node, 0])
            probabilities = capture.last[node]
            rows[node] = {'episode': context['episode'], 'step': step, 'sender': node,
                'packet_id': packet.id, 'action': action, 'sink': env.sink_id, 'wait': env.buoy_number,
                'sink_legal': bool(b.available_actions[step, 0, node, env.sink_id]),
                'legal17_and19': bool(b.available_actions[step, 0, node, 17] and b.available_actions[step, 0, node, 19]),
                'route_before': list(context['metrics'].packets[packet.id]['route']),
                'p_selected_before': float(probabilities[action]), 'p_sink_before': float(probabilities[env.sink_id]),
                'p17_before': float(probabilities[17]), 'p19_before': float(probabilities[19]),
                'ack': None, 'revisit': False}
        context['rows'] = rows
        context['episode_rows'].extend(rows.values())
        return result
    learner.collect = observed_collect

    train, update = learner.trainer.train, learner.trainer.ppo_update
    def observed_update(sample, *a, **kw):
        before = ranking(probe(learner.policy.actor, capture, fixed)[13])
        result = update(sample, *a, **kw)
        after = ranking(probe(learner.policy.actor, capture, fixed)[13])
        context['updates'] += 1
        active = np.asarray(sample[12]).reshape(-1) > 0
        node = np.asarray(sample[4]).reshape(-1).astype(int)
        action = np.asarray(sample[8]).reshape(-1).astype(int)
        z = np.asarray(sample[14]).reshape(-1)
        ratio = result[5].detach().cpu().numpy().reshape(-1)
        target = active & (node == 13)
        samples = [{'action': int(action[i]), 'advantage': float(z[i]), 'ratio': float(ratio[i])}
                   for i in np.flatnonzero(target)]
        row = {'episode': context['episode'], 'optimizer_step': context['updates'],
               'probe_before': before, 'probe_after': after,
               'probe_gap_change': after['gap17_minus19']-before['gap17_minus19'],
               'node13_samples': samples, 'active_batch_size': int(active.sum()),
               'policy_loss': float(result[2]), 'entropy': float(result[3]),
               'actor_grad_norm': float(result[4]),
               'limitation': 'Observed shared-parameter Adam update; local samples alone are not a causal gradient decomposition.'}
        updates.append(row)
        update_stream.write(json.dumps(row, allow_nan=False)+'\n')
        update_stream.flush()
        return result
    learner.trainer.ppo_update = observed_update

    def observed_train(buffer, *a, **kw):
        ep = context['episode']
        values = learner.trainer.value_normalizer.denormalize(buffer.value_preds)
        raw, z = helper.advantages(buffer.returns[:-1], values[:-1], buffer.active_masks[:-1])
        runtime = learner.pm_runtime
        if runtime is not None:
            # Verify every original historical rollout, not only the final checkpoint.
            audit_path = reference.parent/'value_audit'/('episode_'+str(ep)+'.npz')
            require(audit_path.is_file(), 'Missing original value audit: '+str(audit_path))
            with np.load(str(audit_path)) as original:
                require(np.array_equal(values, original['raw_values']) and np.array_equal(raw, original['actual_gae']),
                        'Original value/GAE trajectory changed')
                phi = np.broadcast_to(np.asarray([r['phi_t'] for r in runtime.episode_records]+[0.], dtype=np.float64)
                    .reshape(-1, 1, 1, 1), original['phi'].shape)
                require(np.array_equal(phi, original['phi']), 'Original PM state trajectory changed')
                require(np.array_equal(buffer.masks, original['masks']), 'Original bootstrap masks changed')
                # Buffer rewards are float32; historical diagnostic coordinates are float64.
                shaped = np.asarray([r['reward_train'] for r in runtime.episode_records], dtype=np.float64)
                require(np.array_equal(shaped, original['shaped_rewards'][:, 0, 0, 0]), 'Original shaping rewards changed')
        rows = context['episode_rows']
        deliveries = {t: sum(r['ack'] is True and r['action'] == env.sink_id for r in rows if r['step'] == t)
                      for t in range(buffer.episode_length)}
        for row in rows:
            t, node = row['step'], row['sender']
            pm = runtime.episode_records[t] if runtime is not None else None
            own = 0.
            if pm is not None:
                before = packet_values[(ep, t)].get(row['packet_id'], 0.)
                after = packet_values[(ep, t+1)].get(row['packet_id'], 0.)
                own = runtime.alpha*(buffer.gamma*after-before)/max(runtime.initial_packets, 1)
            row.update(advantage_raw=float(raw[t, 0, node, 0]), advantage_normalized=float(z[t, 0, node, 0]),
                value=float(values[t, 0, node, 0]), **{'return': float(buffer.returns[t, 0, node, 0])},
                reward=float(buffer.rewards[t, 0, node, 0]),
                reward_env=pm['reward_env'] if pm is not None else float(buffer.rewards[t, 0, node, 0]),
                reward_progress=pm['reward_progress'] if pm is not None else 0., packet_progress=own,
                deliveries=deliveries[t], other_deliveries=deliveries[t]-int(row['ack'] is True and row['action'] == env.sink_id))
        target_rows = [r for r in rows if r['sender'] == 13 or r['packet_id'] == 13]
        arrays = [np.asarray([getattr(buffer, k)[r['step'], 0, r['sender']] for r in target_rows]) for k in FIELDS]
        pre = probe(learner.policy.actor, capture, arrays)
        error = max(abs(float(p[r['action']])-r['p_selected_before']) for p, r in zip(pre, target_rows)) if target_rows else 0.
        require(error < 1e-6, 'Behavior-policy probability replay changed')
        result = train(buffer, *a, **kw)
        post = probe(learner.policy.actor, capture, arrays)
        for row, probabilities in zip(target_rows, post):
            row.update(p17_after=float(probabilities[17]), p19_after=float(probabilities[19]),
                p_selected_after=float(probabilities[row['action']]))
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()
        row_count[0] += len(rows)
        rollouts.append({'episode': ep, 'active_rows': len(rows), 'behavior_replay_error': error,
                         'fixed_probe': ranking(probe(learner.policy.actor, capture, fixed)[13])})
        print('ROLLOUT {} step={} gap={:+.8f}'.format(ep, context['updates'], rollouts[-1]['fixed_probe']['gap17_minus19']), flush=True)
        return result
    learner.trainer.train = observed_train
    try:
        with patch('buoy_network.runner.PacketMetrics', Metrics), patch.object(FrozenPMRuntime, 'potential', observed_potential):
            d.Runner([d.Agent(b, learner) for b in env.buoys], env).run()
    finally:
        stream.close()
        update_stream.close()
        capture.close()
    current = output/'replay/moves_5'
    require(learner.training_records == report['training_updates'][:episodes], 'Original losses or optimizer steps differ')
    exact_files = {}
    for name in ('TG_physical_metrics.jsonl', 'TG_reward_components.csv', 'TG_PM_rewards.csv'):
        original = reference/'logs'/name
        if not original.is_file():
            continue
        actual = current/'logs'/name
        if full:
            require(actual.read_bytes() == original.read_bytes(), 'Original log differs: '+name)
        else:
            expected_lines = original.read_bytes().splitlines(keepends=True)
            n = episodes if name.endswith('.jsonl') else episodes*config['episode_length']+1
            require(actual.read_bytes() == b''.join(expected_lines[:n]), 'Original log prefix differs: '+name)
        exact_files[name] = sha(actual)
    model_match = None
    if full:
        original = torch.load(str(checkpoint), map_location='cpu')
        for name in ('actor', 'critic'):
            require(d.tensor_digest(getattr(learner.policy, name).state_dict()) == d.tensor_digest(original[name]),
                    'Original final '+name+' differs')
        model_match = True
    require(sha(checkpoint) == reference_sha, 'Original checkpoint modified')
    if learner.pm_runtime is not None:
        learner.pm_runtime.assert_frozen()
    final = {'success': True, 'group': args.group, 'seed': 22, 'episodes': episodes, 'full_replay': full,
        'scope': 'Exact repeated training for observation; not an independent efficacy seed.',
        'reference': str(reference), 'reference_checkpoint_sha256': reference_sha,
        'exact_final_actor_critic': model_match, 'exact_training_updates': True, 'exact_logs': exact_files,
        'exact_original_PM_value_GAE_rollouts': episodes if learner.pm_runtime is not None else 0,
        'environment_steps': learner.profile['collect_calls'], 'rollout_updates': learner.profile['ppo_updates'],
        'optimizer_steps': context['updates'], 'active_rows': row_count[0],
        'initial_probe': initial_probe, 'rollouts': rollouts,
        'fixed_probe_inputs_sha256': sha(fixed_path), 'script_sha256': sha(Path(__file__)),
        'training_credit_sha256': sha(output/'training_credit.jsonl'), 'updates_sha256': sha(output/'updates.jsonl'),
        'limitations': ['Same-input probability changes identify optimizer timing, not individual causal credit.',
                        'Training seed and development inputs were already inspected; no new generalization evidence.']}
    (output/'report.json').write_text(json.dumps(final, indent=2, allow_nan=False), encoding='utf-8')
    print('PASS '+str(output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
