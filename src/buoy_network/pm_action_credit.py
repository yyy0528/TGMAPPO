"""Training-only packet action scores from a frozen PM; approximate one-step lookahead.

Future positions/links and other agents are held at their observed values.
This is not a simulator counterfactual, COMA Q-function, or invariance guarantee.
"""
import math
import numpy as np
import torch
from .progress_model import prefix_features
from .pm_return import expected_task_return


@torch.no_grad()
def packet_action_scores(runtime, actor, node, packet, legal, step, learned=True):
    if runtime.potential_mode != 'task_return' or step != runtime.step or not runtime.started:
        raise ValueError('Packet credit requires current task-return PM state')
    n, horizon = runtime.n, runtime.horizon
    legal = np.asarray(legal)
    if legal.shape != (n+1,) or not legal.any() or not legal[n]:
        raise ValueError('Invalid packet candidate mask')
    if packet.id not in runtime.hidden or packet.route[-1] != node:
        raise ValueError('Unknown/current packet PM history')
    choices = np.flatnonzero(legal > 0)
    rows, hidden, variants = [], [], {}
    following_step = step+1
    # There is no unseen t+1 input: positions, topology, and other queues are held.
    for action in choices:
        action = int(action)
        if action != n and (action == node or not 0 < actor['distances'][node][action] <= 1.5):
            raise ValueError('Packet candidate is not a physical neighbor')
        for success in (False, True):
            if success and action == n:
                continue
            delivered = success and action == packet.destination
            if delivered or following_step == horizon or not learned:
                variants[(action, success)] = 0.
                continue
            after_node = action if success else node
            route = list(packet.route)+([action] if success else [])
            nodes = np.asarray(actor['node_features']).copy()
            if success:
                nodes[node, 2] -= 1./max(runtime.initial_packets, 1)
                nodes[action, 2] += 1./max(runtime.initial_packets, 1)
            hypothetical = dict(actor, node_features=nodes)
            feature, _ = prefix_features(hypothetical, after_node, packet.destination, route,
                following_step, horizon, runtime.prefix_state[packet.id])
            rows.append(feature); hidden.append(runtime.hidden[packet.id])
            variants[(action, success)] = len(rows)-1
    values = []
    if rows:
        logits, _ = runtime.model.step(torch.tensor(np.asarray(rows)), torch.stack(hidden).unsqueeze(0))
        values = expected_task_return(logits, torch.full((len(rows),), horizon-following_step, dtype=torch.long),
                                      horizon, runtime.gamma, runtime.task_contract)['task_return'].tolist()
    # Replace index placeholders only for actually modeled variants.
    if rows:
        for key, value in list(variants.items()):
            if type(value) is int:
                variants[key] = float(values[value])
    result = np.zeros(n+1, dtype=np.float32)
    c = runtime.task_contract
    for action in choices:
        action = int(action)
        p = 0. if action == n else float(np.asarray(actor['obs'])[node, action])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('ACK observation must be an observed probability')
        failure = variants[(action, False)]
        success = variants.get((action, True), failure)
        sink = action == packet.destination
        immediate = p*c['delivery_weight']*int(sink)-c['delay_weight']/horizon
        if following_step == horizon:
            immediate -= c['unfinished_weight']*(1-p*int(sink))
        result[action] = immediate+runtime.gamma*((1-p)*failure+p*success)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite PM packet candidate scores')
    return result

def target_distribution(scores, legal, temperature):
    legal = np.asarray(legal) > 0
    q = np.asarray(scores,dtype=np.float64)
    if q.shape != legal.shape or not legal.any() or not np.isfinite(q[legal]).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid PM target inputs')
    result = np.zeros_like(q,dtype=np.float32)
    mass = np.exp((q[legal]-q[legal].max())/temperature)
    result[legal] = mass/mass.sum()
    return result


@torch.no_grad()
def collect_action_targets(runtime, env, buffer, step):
    if not runtime.started or runtime.step != step:
        raise ValueError('PM action target time mismatch')
    packet_at = {}
    for packet in sorted(env.packets,key=lambda p:p.queue_order):
        if not packet.if_sink:
            packet_at.setdefault(packet.agent_id,packet)
    actor = {'obs':buffer.obs[step,0], 'node_features':buffer.node_obs[step,0,0],
             'distances':buffer.adj[step,0,0]}
    masks = buffer.available_actions[step,0]
    targets = np.zeros_like(masks,dtype=np.float32)
    for node,legal in enumerate(masks):
        if buffer.active_masks[step,0,node,0] <= 0:
            targets[node,-1] = 1.
            continue
        scores = packet_action_scores(runtime,actor,node,packet_at[node],legal,step,
                                      buffer.pm_action_spec['mode'] in ('learned','learned_regret','learned_set'))
        targets[node] = target_distribution(scores,legal,buffer.pm_action_spec['temperature'])
    return targets
