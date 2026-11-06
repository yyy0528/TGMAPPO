"""Capture seed22 step4 exactly, then ablate gradient components on copies."""
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    import torch
    from diagnose_tg_pm_updates import read, sha, require, FIELDS, probe, ranking
    from diagnose_tg_credit import RolloutCapture
    diagnosis = ROOT/'results/tg_seed22_signals/20261105_165019_896744'
    snapshot = Path(read(diagnosis/'manifest.json')['source_snapshot'])
    sys.path[:0] = [str(snapshot), str(snapshot/'src'), str(snapshot/'scripts')]
    import diagnose_tg_policy as d
    from onpolicy.algorithms.graph_mappo import GR_MAPPO
    # Explicitly load the current observer; runtime imports remain in the archive.
    import importlib.util
    spec = importlib.util.spec_from_file_location('update_observer', str(ROOT/'scripts/diagnose_tg_pm_updates.py'))
    observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
    root = ROOT/'results/tg_pm_gradient'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    print('OUTPUT '+str(root), flush=True)
    original = GR_MAPPO.ppo_update
    count, saved = [0], {}

    def recorded(self, sample, *a, **kw):
        count[0] += 1
        if count[0] == 4:
            saved.update(actor=copy.deepcopy(self.policy.actor.state_dict()),
                optimizer=copy.deepcopy(self.policy.actor_optimizer.state_dict()),
                sample=tuple(x.copy() if isinstance(x, np.ndarray) else copy.deepcopy(x) for x in sample),
                config=dict(vars(self.policy.actor.args)), mode=self.policy.actor.training,
                clip_param=self.clip_param, entropy_coef=self.entropy_coef,
                max_grad_norm=self.max_grad_norm, use_max_grad_norm=self._use_max_grad_norm)
        result = original(self, sample, *a, **kw)
        if count[0] == 4:
            saved.update(after=copy.deepcopy(self.policy.actor.state_dict()),
                clipped_gradients=[p.grad.detach().clone() if p.grad is not None else None for p in self.policy.actor.parameters()])
        return result

    argv = [sys.argv[0], '--diagnosis', str(diagnosis), '--group', 'alpha01',
            '--output', str(root/'prefix'), '--episodes', '2']
    with patch.object(sys, 'argv', argv), patch.object(GR_MAPPO, 'ppo_update', recorded):
        observer.main()
    previous = ROOT/'results/tg_pm_update_replay/20261105_211844_626825/prefix_alpha01'
    for name in ('training_credit.jsonl', 'updates.jsonl'):
        require((root/'prefix'/name).read_bytes() == (previous/name).read_bytes(), 'Capture perturbed original prefix')
    torch.save(saved, str(root/'step4.pt'))
    _, learner = d.build(saved['config'], root/'constructor')
    actor = learner.policy.actor
    actor.load_state_dict(saved['actor']); actor.train(saved['mode'])
    parameters = list(actor.parameters())
    sample = saved['sample']
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32)
    logp, entropy = actor.evaluate_actions(sample[1], sample[2], sample[3], sample[4], sample[6], sample[8], sample[11],
                                           available_actions=sample[15], active_masks=sample[12])
    mask, z, old = tensor(sample[12]), tensor(sample[14]), tensor(sample[13])
    ratio = (logp-old).exp()
    losses = -torch.min(ratio*z, ratio.clamp(1-saved['clip_param'], 1+saved['clip_param'])*z)
    denominator = mask.sum().clamp(min=1)
    node = torch.as_tensor(sample[4]).reshape(-1, 1) == 13
    parts = {'node13': (losses*mask*node).sum()/denominator,
             'other_nodes': (losses*mask*~node).sum()/denominator,
             'entropy': -saved['entropy_coef']*entropy}
    total = (losses*mask).sum()/denominator-saved['entropy_coef']*entropy
    gradients = {k: torch.autograd.grad(v, parameters, retain_graph=True, allow_unused=True) for k, v in parts.items()}
    full = torch.autograd.grad(total, parameters, allow_unused=True)
    def vector(values):
        return torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p, v in zip(parameters, values)])
    summed = sum(vector(g) for g in gradients.values())
    decomposition_error = float((summed-vector(full)).abs().max())
    require(decomposition_error < 1e-6, 'Gradient components do not reconstruct full loss')
    with np.load(str(diagnosis/'alpha01_greedy/actor_inputs.npz')) as cache:
        fixed = [cache[k][0].copy() for k in FIELDS]
    capture = RolloutCapture(actor)
    before = ranking(probe(actor, capture, fixed)[13])
    capture.close()
    results = {}
    component_sets = {'full': full, 'node13': gradients['node13'], 'other_nodes': gradients['other_nodes'],
        'entropy': gradients['entropy'], 'no_entropy': tuple(None if g is None else
            g-(torch.zeros_like(g) if e is None else e) for g, e in zip(full, gradients['entropy'])),
        'zero_current': tuple(None if g is None else torch.zeros_like(g) for g in full)}
    for key, grads in component_sets.items():
        for history in (True, False):
            clone = copy.deepcopy(actor)
            optimizer = torch.optim.Adam(clone.parameters())
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            if not history:
                optimizer.state.clear()
            for p, g, original_grad in zip(clone.parameters(), grads, full):
                p.grad = None if original_grad is None else (torch.zeros_like(p) if g is None else g.detach().clone())
            if saved['use_max_grad_norm']:
                torch.nn.utils.clip_grad_norm_(clone.parameters(), saved['max_grad_norm'])
            grad_error = max(float((p.grad-g).abs().max()) for p, g in zip(clone.parameters(), saved['clipped_gradients']) if g is not None)
            optimizer.step()
            c = RolloutCapture(clone)
            after = ranking(probe(clone, c, fixed)[13]); c.close()
            actual_error = max(float((v-saved['after'][k]).abs().max()) for k, v in clone.state_dict().items())
            label = key+('_history' if history else '_fresh')
            results[label] = {'before': before, 'after': after, 'gap_change': after['gap17_minus19']-before['gap17_minus19'],
                              'actual_parameter_error': actual_error, 'actual_clipped_gradient_error': grad_error}
    require(results['full_history']['actual_parameter_error'] < 1e-7 and
            results['full_history']['actual_clipped_gradient_error'] < 1e-6, 'Clone failed to reproduce actual optimizer step')
    report = {'success': True, 'step': 4, 'seed': 22, 'alpha': .1, 'environment_steps': 160,
        'optimizer_steps': 4, 'exact_prior_observer_rows': True, 'gradient_sum_error': decomposition_error,
        'component_gradient_norms': {k: float(vector(g).norm()) for k, g in gradients.items()},
        'cases': results, 'snapshot_sha256': sha(root/'step4.pt'), 'script_sha256': sha(Path(__file__)),
        'scope': 'Counterfactual optimizer steps on copies of one real minibatch; no alternate actions executed.',
        'limitations': ['Removing loss components changes gradient clipping and Adam denominator; effects are nonlinear and not additive.',
            'Fresh Adam resets moments and bias correction together; this is not an independent efficacy trial.']}
    (root/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for key, row in results.items():
        print(key, row['after']['argmax'], '{:+.8f}'.format(row['gap_change']))
    print('PASS '+str(root/'report.json'), flush=True)


if __name__ == '__main__':
    main()
