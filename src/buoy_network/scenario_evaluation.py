"""Explicit evaluation-only interventions on the ORIGINAL buoy environment."""
import math
from contextlib import contextmanager
from unittest.mock import patch

BASELINE=dict(loss_scale=1.,walk_step=.1,packets_per_source=1)
PROFILES={
    'baseline':dict(BASELINE),
    'loss_low':dict(BASELINE,loss_scale=.5),
    'loss_high':dict(BASELINE,loss_scale=1.5),
    'walk_slow':dict(BASELINE,walk_step=.05),
    'walk_fast':dict(BASELINE,walk_step=.2),
    'load_double':dict(BASELINE,packets_per_source=2),
}


def validate(settings):
    if set(settings)!=set(BASELINE):raise ValueError('Incomplete/unknown scenario settings')
    x=dict(settings)
    if any(isinstance(x[k],bool) or not math.isfinite(float(x[k])) for k in ('loss_scale','walk_step')):
        raise ValueError('Nonfinite scenario parameter')
    if not 0<=x['loss_scale']<=2 or not 0<x['walk_step']<=1:
        raise ValueError('Scenario parameters outside registered range')
    if type(x['packets_per_source']) is not int or not 1<=x['packets_per_source']<=4:
        raise ValueError('Require integer initial packets per source')
    return x


def apply_environment(env,settings):
    settings=validate(settings)
    before=[];after=[]
    for node in env.buoys:
        before.extend(node.lost_rate_list)
        node.lost_rate_list=[min(1.,max(0.,float(x)*settings['loss_scale'])) for x in node.lost_rate_list]
        node.lost_rate=min(1.,max(0.,float(node.lost_rate)*settings['loss_scale']))
        node.max_speed=settings['walk_step']
        after.extend(node.lost_rate_list)
    return dict(loss_before=before,loss_after=after,walk_step=settings['walk_step'],
                displacement_limit_per_coordinate=.1)


def generate_packets(env,original,settings):
    """Repeat the SAME original initial-burst generator; retain IDs and FIFO."""
    settings=validate(settings)
    for _ in range(settings['packets_per_source']):original(env)
    packets=env.packets
    expected=(env.buoy_number-1)*settings['packets_per_source']
    if env.packet_id!=expected or len(packets)!=expected or len({p.id for p in packets})!=expected:
        raise ValueError('Initial packet conservation/unique-ID check failed')
    queues={node.id:sum(p.agent_id==node.id for p in packets) for node in env.buoys if not node.if_sink}
    if any(n!=settings['packets_per_source'] for n in queues.values()):
        raise ValueError('Initial per-source load mismatch')
    if any(p.queue_order!=p.id for p in packets):raise ValueError('Initial FIFO order changed')
    return dict(generated=expected,packets_per_source=settings['packets_per_source'],queues=queues)


@contextmanager
def scenario_context(env_class,settings):
    settings=validate(settings)
    build,generate=env_class.build_env,env_class.packet_generation
    audit=dict(settings=settings,environment_builds=[],initial_bursts=[])
    def built(env,*a,**kw):
        result=build(env,*a,**kw)
        if not getattr(env.all_args,'tg_evaluate',False):raise ValueError('Scenario intervention is evaluation-only')
        audit['environment_builds'].append(apply_environment(env,settings))
        return result
    def generated(env):
        audit['initial_bursts'].append(generate_packets(env,generate,settings))
    with patch.object(env_class,'build_env',built),patch.object(env_class,'packet_generation',generated):
        yield audit
