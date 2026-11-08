"""Protocol and audit rules for the five-group original-TG CUDA study."""
from pathlib import Path
from tg_convergence import require, digest, nested_digest, rng_digest

SCHEMA = 'tg_five_convergence_v1'
GROUPS = ['TG', 'Immediate', 'PM', 'WM', 'PM_WM']
VARIANTS = {
    'TG': dict(wm=False, pm=False, mode='disabled'),
    'Immediate': dict(wm=False, pm=True, mode='zero_set'),
    'PM': dict(wm=False, pm=True, mode='learned_set'),
    'WM': dict(wm=True, pm=False, mode='disabled'),
    'PM_WM': dict(wm=True, pm=True, mode='learned_set'),
}


def validate(spec):
    require(spec['schema'] == SCHEMA and spec['groups'] == GROUPS, 'Require all five groups in registered order')
    require(spec['device'] == 'cuda' and spec['torch_threads'] == 1 and spec['deterministic'] is True, 'Require unified deterministic CUDA/one-thread protocol')
    require(spec['horizon'] == 80 and spec['moving_buoys'] == 5, 'Original environment/horizon changed')
    full = spec['episodes'] == 300
    require(full or spec['episodes'] == 2, 'Only full300 or bounded two-episode probe supported')
    require(spec['seeds'] == ([11, 22, 33] if full else [22]), 'Unexpected training seeds')
    require(spec['scenes'] == ([8001, 8002, 8003] if full else [8001]), 'Unexpected evaluation scenes')
    require(spec['interval'] == (20 if full else 1) and spec['eval_episodes'] == (2 if full else 1), 'Evaluation budget changed')
    require(Path(spec['python']).is_file(), 'Python interpreter missing')
    for name in ('wm', 'pm'):
        require(digest(spec[name+'_checkpoint']) == spec[name+'_sha256'], name+' checkpoint hash mismatch')
    return spec


def capacity(group):
    return 21956 + (2209 if VARIANTS[group]['wm'] else 0), 18615


def comparable_initial(model):
    """Compare shared weights, optimizer settings and RNG, excluding WM's extra head."""
    actor = {k: v for k, v in model.policy.actor.state_dict().items() if not k.startswith('wm_fusion.')}
    opt = model.policy.actor_optimizer.state_dict()
    require(not opt['state'], 'Initial Actor optimizer is not fresh')
    if model.policy.actor.use_wm:
        head = model.policy.actor.wm_fusion[-1]
        require(not head.weight.detach().count_nonzero() and not head.bias.detach().count_nonzero(), 'WM residual must start at zero')
    return dict(actor_backbone=nested_digest(actor), critic=nested_digest(model.policy.critic.state_dict()),
                actor_optimizer_settings=nested_digest([{k:v for k,v in g.items() if k != 'params'} for g in opt['param_groups']]),
                critic_optimizer=nested_digest(model.policy.critic_optimizer.state_dict()),
                normalizer=nested_digest(model.trainer.value_normalizer.state_dict() if model.trainer.value_normalizer is not None else None),
                rng=rng_digest())


def audit_components(report, config, spec, group):
    variant = VARIANTS[group]
    actor, critic = capacity(group)
    require(report['resources']['actor_trainable'] == actor and report['resources']['critic_trainable'] == critic, 'Unexpected group capacity')
    require(report['resources']['device'] == 'cuda' and config['tg_device'] == 'cuda', 'Policy must run on CUDA')
    require(config['use_wm_actor'] == variant['wm'] and config['use_pm_reward'] == variant['pm'], 'Wrong WM/PM switches')
    require(report['pm_action_credit'] == dict(mode=variant['mode'], coefficient=.1, temperature=.2), 'Wrong auxiliary objective')
    for name in ('wm', 'pm'):
        actual = report.get(name)
        require((actual is not None) == variant[name], 'Wrong frozen component: '+name)
        if actual is not None:
            require(actual['frozen'] and actual['checkpoint_sha256'] == spec['frozen_'+name+'_sha256']
                    and actual['inference_device'].startswith('cuda'), 'Frozen '+name+' weights/device changed')
    if variant['pm']:
        require(report['pm']['value_mode'] == 'base_residual' and report['pm']['potential_mode'] == 'task_return'
                and report['pm']['alpha'] == .1, 'PM coordinates/settings changed')
    if variant['wm']:
        require(report['wm']['feature_source'] == 'learned' and report['wm']['learned_forward_used'], 'Wrong WM features')


def budget(spec):
    from tg_convergence import checkpoint_grid
    models = len(spec['groups'])*len(spec['seeds'])
    return dict(training_models=models, training_steps=models*spec['episodes']*80,
                evaluation_steps=models*len(checkpoint_grid(spec['episodes'],spec['interval']))*len(spec['scenes'])*spec['eval_episodes']*80)
