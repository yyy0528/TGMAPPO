"""Initial value-prior reliability from independent, pre-PPO calibration tasks.

This gate is selected once, before fresh training RNGs/trajectories are started.
It is not an uncertainty head or a guarantee of better closed-loop routing.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCHEMA = 'tg_pm_initial_value_gate_v1'
RULE = {'minimum_episodes': 4, 'minimum_scenes': 2,
        'relative_mse_margin': .05, 'every_episode_improves': True}


def select_gate(episodes, horizon, nodes, gamma, maximum_alpha):
    if (type(horizon) is not int or horizon < 1 or type(nodes) is not int or nodes < 1
            or not 0 < gamma <= 1 or not math.isfinite(maximum_alpha) or maximum_alpha <= 0
            or len(episodes) < RULE['minimum_episodes']):
        raise ValueError('Invalid/insufficient PM calibration tasks')
    scores=[];keys=set()
    for row in episodes:
        key=(row['scene'],row['episode'])
        if (any(type(x) is not int or x < 0 for x in key) or key in keys):
            raise ValueError('Duplicate/invalid calibration task identity')
        keys.add(key)
        r,v,phi=[np.asarray(row[k],dtype=np.float64) for k in ('rewards','values','phi')]
        if (r.shape != (horizon,) or v.shape != (horizon,nodes) or phi.shape != (horizon+1,)
                or phi[-1] != 0 or not all(np.isfinite(x).all() for x in (r,v,phi))):
            raise ValueError('Invalid PM calibration values/rewards/terminal potential')
        mc=np.zeros(horizon,dtype=np.float64)
        for t in reversed(range(horizon)):mc[t]=r[t]+(gamma*mc[t+1] if t+1<horizon else 0)
        base=float(np.mean((v-mc[:,None])**2))
        prior=float(np.mean((v+maximum_alpha*phi[:-1,None]-mc[:,None])**2))
        scores.append({'scene':key[0],'episode':key[1],'baseline_mse':base,'pm_prior_mse':prior})
    if len({k[0] for k in keys}) < RULE['minimum_scenes']:
        raise ValueError('PM calibration requires independent scenes')
    base=sum(x['baseline_mse'] for x in scores)/len(scores)
    prior=sum(x['pm_prior_mse'] for x in scores)/len(scores)
    enabled=(base > 0 and prior <= (1-RULE['relative_mse_margin'])*base
             and all(x['pm_prior_mse'] < x['baseline_mse'] for x in scores))
    return {'rule':dict(RULE),'enabled':bool(enabled),'effective_alpha':maximum_alpha if enabled else 0.,
            'baseline_mse':base,'pm_prior_mse':prior,'episodes':scores,
            'interpretation':'initial value-prior calibration only; not routing efficacy or uncertainty'}


def load_gate(path, pm_sha256, horizon, nodes, gamma, maximum_alpha, contract):
    raw=Path(path).read_bytes()
    report=json.loads(raw.decode('utf-8'))
    expected={'schema':SCHEMA,'phase':'independent_initial_policy_calibration',
        'horizon':horizon,'nodes':nodes,'gamma':gamma,'maximum_alpha':maximum_alpha,
        'pm_checkpoint_sha256':pm_sha256,'reward_contract':contract,
        'potential_mode':'task_return','value_mode':'learned_shaped',
        'weights_unchanged':True,'optimizer_updates':0}
    if any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError('PM reliability report contract/model/phase mismatch')
    if type(report.get('training_seed')) is not int or report['training_seed'] in {e['scene'] for e in report['episodes']}:
        raise ValueError('PM calibration must precede independent fresh training tasks')
    for k in ('initial_actor_sha256','initial_critic_sha256'):
        value=report.get(k,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing PM calibration initialization fingerprint')
    if not isinstance(report.get('policy_signature'),dict) or not report['policy_signature']:
        raise ValueError('Missing PM calibration policy signature')
    decision=select_gate(report['episodes'],horizon,nodes,gamma,maximum_alpha)
    if report.get('decision')!=decision:
        raise ValueError('PM reliability decision differs from cached calibration evidence')
    return {'report':str(Path(path).resolve()),'report_sha256':hashlib.sha256(raw).hexdigest(),
        'training_seed':report['training_seed'],'initial_actor_sha256':report['initial_actor_sha256'],
        'initial_critic_sha256':report['initial_critic_sha256'],'maximum_alpha':maximum_alpha,
        'policy_signature':report['policy_signature'],
        'effective_alpha':decision['effective_alpha'],'enabled':decision['enabled'],
        'rule':dict(RULE),'calibration_episodes':len(report['episodes']),
        'calibration_steps':len(report['episodes'])*horizon,
        'baseline_mse':decision['baseline_mse'],'pm_prior_mse':decision['pm_prior_mse']}
