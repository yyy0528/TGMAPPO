"""Auditable feed-forward baselines for the ORIGINAL buoy Runner.

This module never steps the environment. Simulation, FIFO service, masks,
task_v2 rewards and physical metrics remain in buoy_network.runner.Runner.
Actors receive the same observed global node positions/queues as TG (no true
loss probabilities, packet destinations/history or future outcomes). MLPs use
a different encoding, not additional privileged information.
"""
import copy
import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


ALGORITHMS = ('MAPPO', 'IPPO', 'QMIX', 'IQL', 'VDN', 'COMA', 'Greedy')


class SumMixer(nn.Module):
    def forward(self, qs, state):
        return qs.sum(-1)


class CounterfactualCritic(nn.Module):
    """Q_i(s,u_-i,a): the input explicitly excludes agent i's chosen action."""
    def __init__(self, agents, features, state_dim):
        super().__init__()
        self.agents=agents
        self.net=mlp(features+state_dim+agents*(agents+1),agents+1)

    def forward(self,x,state,actions):
        n=self.agents
        joint=torch.nn.functional.one_hot(actions,n+1).to(x.dtype)
        other=joint.unsqueeze(-3).expand(*joint.shape[:-2],n,n,n+1)
        mask=(1-torch.eye(n,device=x.device,dtype=x.dtype)).unsqueeze(-1)
        other=(other*mask).flatten(-2)
        expanded=state.unsqueeze(-2).expand(*x.shape[:-1],state.shape[-1])
        return self.net(torch.cat((x,expanded,other),-1))


def counterfactual_advantage(q, probabilities, actions):
    return q.gather(-1,actions.unsqueeze(-1)).squeeze(-1)-(q*probabilities).sum(-1)


def td_lambda_returns(rewards, target_q, gamma, lam):
    result=torch.zeros_like(target_q)
    for t in reversed(range(len(rewards))):
        result[t]=rewards[t]
        if t+1<len(rewards):
            result[t]+=gamma*((1-lam)*target_q[t+1]+lam*result[t+1])
    return result


def tensor_hash(network):
    h = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode(observations, nodes, distances):
    """Encode already-observable fields, with fixed physical scaling only."""
    obs = np.asarray(observations, dtype=np.float32)[0].copy()
    graph = np.asarray(nodes, dtype=np.float32)[0, 0].copy()
    adj = np.asarray(distances, dtype=np.float32)[0, 0].copy()
    n = len(obs)
    obs[:, -1] /= max(n-1, 1)
    graph[:, :2] /= 5.
    adj /= 5.
    features = np.concatenate((obs, np.tile(graph.reshape(1, -1), (n, 1)), adj,
                               np.eye(n, dtype=np.float32)), axis=1)
    state = np.concatenate((obs.reshape(-1), graph.reshape(-1), adj.reshape(-1)))
    if not np.isfinite(features).all() or not np.isfinite(state).all():
        raise ValueError('Nonfinite baseline observation')
    return features, state


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, outputs))


def masked_distribution(logits, legal):
    if not legal.any(dim=-1).all():
        raise ValueError('Action mask contains an empty row')
    return Categorical(logits=logits.masked_fill(~legal, -1e9))


def gae(rewards, values, gamma=.99, lam=.95):
    """Finite task horizon is terminal; bootstrap is zero, not a truncation."""
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else torch.zeros_like(carry)
        carry = rewards[t] + gamma*next_value - values[t] + gamma*lam*carry
        advantages[t] = carry
    return advantages, advantages+values


class Mixer(nn.Module):
    """QMIX: a joint mixer for ALL agents, nonnegative hypernetwork weights."""
    def __init__(self, agents, state_dim, hidden=32):
        super().__init__()
        self.agents, self.hidden = agents, hidden
        self.w1 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, agents*hidden))
        self.w2 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, hidden))
        self.b1 = nn.Linear(state_dim, hidden)
        self.b2 = mlp(state_dim, 1)

    def forward(self, qs, state):
        w1 = self.w1(state).abs().reshape(-1, self.agents, self.hidden)
        w2 = self.w2(state).abs().reshape(-1, self.hidden, 1)
        h = torch.nn.functional.elu(torch.bmm(qs.unsqueeze(1), w1) + self.b1(state).unsqueeze(1))
        return torch.bmm(h, w2).reshape(-1) + self.b2(state).reshape(-1)


class BaselineLearner:
    """Runner protocol adapter; parameter sharing, fresh on-policy PPO rollouts.

    MAPPO: centralized team critic on joint observations. IPPO: per-agent
    critic on that agent's observation encoding, no joint critic input.
    Both have access to the task's original global observation; IPPO does NOT
    mean a newly restricted local-sensing environment.
    IQL: shared independent masked TD. QMIX: shared utilities + joint mixer.
    """
    algorithm = None

    def __init__(self, *unused, env=None, **kwargs):
        if self.algorithm not in ALGORITHMS:
            raise ValueError('Unknown baseline')
        self.env, self.all_args = env, env.all_args
        self.n = env.buoy_number
        self.device = torch.device(self.all_args.tg_device)
        self.evaluating = self.all_args.tg_evaluate
        self.wm_runtime = self.pm_runtime = None
        self.buffer = SimpleNamespace(memory_list=[])
        self.policy = SimpleNamespace(lr_decay=lambda *args: None)
        self.steps = self.episodes = self.updates = self.inference_calls = 0
        self.inference_seconds = self.update_seconds = 0.
        self.records, self.rollout = [], []
        self.replay = deque(maxlen=3000)
        self.random = random.Random(self.all_args.seed + 177)
        obs, ids, nodes, adj = env.observe_graph()
        x, state = encode([obs], [nodes], [adj])
        self.actor = mlp(x.shape[-1], self.n+1).to(self.device)
        self.critic = (CounterfactualCritic(self.n,x.shape[-1],len(state)) if self.algorithm=='COMA'
                       else mlp(len(state) if self.algorithm == 'MAPPO' else x.shape[-1], 1)).to(self.device)
        self.mixer = (Mixer(self.n, len(state)).to(self.device) if self.algorithm == 'QMIX'
                      else SumMixer().to(self.device) if self.algorithm=='VDN' else None)
        self.target_critic=copy.deepcopy(self.critic) if self.algorithm=='COMA' else None
        self.target = copy.deepcopy(self.actor)
        self.target_mixer = copy.deepcopy(self.mixer)
        parameters = list(self.actor.parameters()) + (list(self.mixer.parameters()) if self.mixer else [])
        self.actor_optimizer = torch.optim.Adam(parameters, lr=self.all_args.lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.all_args.critic_lr, eps=1e-5)
        self.initial_actor = tensor_hash(self.actor)
        self.initial_critic = tensor_hash(self.critic)
        self.initial_mixer = tensor_hash(self.mixer) if self.mixer else None
        self.loaded = None
        self.on_checkpoint = None

    def t(self, value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

    def warmup(self):
        # Match native TG initialization exactly (including reset_graph).
        obs, ids, nodes, adj = self.env.reset_graph()
        self.x, self.state = encode([obs], [nodes], [adj])
        self.buffer.memory_list = []
        self.rollout = []
        if self.on_checkpoint:
            self.on_checkpoint(self, self.episodes)

    def values(self, x, state):
        if self.algorithm == 'MAPPO':
            v = self.critic(state)
            return v.expand(*x.shape[:-1])
        return self.critic(x).squeeze(-1)

    @torch.no_grad()
    def collect(self, step):
        start = time.perf_counter()
        legal = np.asarray(self.env.available_actions()[0], dtype=bool)
        active = np.asarray(self.env.active_masks()[0, :, 0], dtype=np.float32)
        x, state = self.t(self.x), self.t(self.state)
        dist = masked_distribution(self.actor(x), self.t(legal, torch.bool))
        if self.algorithm == 'Greedy':
            actions = np.full(self.n, self.n, dtype=np.int64)
            sink = self.env.buoys[self.env.sink_id].position
            for i in range(self.n):
                candidates = np.flatnonzero(legal[i, :self.n])
                if len(candidates):
                    actions[i] = min(candidates, key=lambda j: sum((self.env.buoys[j].position[k]-sink[k])**2 for k in (0,1)))
            actions = self.t(actions, torch.long)
        elif self.evaluating or self.algorithm in ('QMIX', 'IQL', 'VDN'):
            actions = dist.logits.argmax(-1)
            if not self.evaluating:
                epsilon = max(.05, 1. - .95*self.steps/max(1, .8*self.all_args.num_episodes*80))
                actions = self.t([self.random.choice(list(np.flatnonzero(legal[i])))
                    if self.random.random() < epsilon else int(a) for i,a in enumerate(actions.tolist())], torch.long)
        else:
            actions = dist.sample()
        value = self.values(x, state) if self.algorithm in ('MAPPO', 'IPPO') else torch.zeros(self.n, device=self.device)
        logp = dist.log_prob(actions)
        self.pending = dict(x=self.x.copy(), state=self.state.copy(), legal=legal.copy(), active=active,
                            action=actions.cpu().numpy(), value=value.cpu().numpy(), logp=logp.cpu().numpy())
        a = actions.cpu().numpy().reshape(1, self.n, 1)
        zeros = np.zeros((1,self.n,self.all_args.recurrent_N,self.all_args.hidden_size), dtype=np.float32)
        self.inference_calls += 1
        self.inference_seconds += time.perf_counter()-start
        return value.cpu().numpy().reshape(1,self.n,1), a, logp.cpu().numpy().reshape(1,self.n,1), zeros, zeros.copy(), np.eye(self.n+1)[a[:,:,0]]

    def insert(self, data):
        self.buffer.memory_list.append(data[-2])
        self.x, self.state = encode(data[0], data[2], data[3])
        row = dict(self.pending, reward=float(np.asarray(data[5])[0,0,0]), next_x=self.x.copy(),
                   next_state=self.state.copy(), next_legal=np.asarray(data[6][0], dtype=bool),
                   next_active=np.asarray(self.env.active_masks()[0,:,0], dtype=np.float32),
                   done=bool(np.asarray(data[7]).all()))
        if not self.evaluating:
            self.rollout.append(row)
            if self.algorithm in ('QMIX','IQL','VDN'):
                self.replay.append(row)
            self.steps += 1

    def compute(self):
        pass  # GAE is computed once in train, from this fresh task rollout.

    def ppo_update(self):
        rows = self.rollout
        x, state = self.t([r['x'] for r in rows]), self.t([r['state'] for r in rows])
        active = self.t([r['active'] for r in rows])
        values = self.t([r['value'] for r in rows])
        rewards = self.t([r['reward'] for r in rows]).unsqueeze(-1)
        advantages, returns = gae(rewards, values, self.all_args.gamma, self.all_args.gae_lambda)
        valid = active.bool()
        if not valid.any():
            raise ValueError('PPO rollout has no active routing decisions')
        advantages = (advantages-advantages[valid].mean())/(advantages[valid].std(unbiased=False)+1e-5)
        actions, old = self.t([r['action'] for r in rows], torch.long), self.t([r['logp'] for r in rows])
        legal = self.t([r['legal'] for r in rows], torch.bool)
        losses = []
        for _ in range(self.all_args.ppo_epoch):
            for idx in torch.randperm(len(rows), device=self.device).chunk(self.all_args.num_mini_batch):
                distribution = masked_distribution(self.actor(x[idx]), legal[idx])
                ratio = (distribution.log_prob(actions[idx])-old[idx]).exp()
                surrogate = torch.minimum(ratio*advantages[idx], ratio.clamp(.8,1.2)*advantages[idx])
                mask = active[idx]
                loss = -(surrogate*mask).sum()/mask.sum().clamp_min(1) - .01*(distribution.entropy()*mask).sum()/mask.sum().clamp_min(1)
                vloss = .5*(self.values(x[idx],state[idx])-returns[idx]).square().mean()
                self.optimize(loss, self.actor_optimizer, self.actor.parameters())
                self.optimize(vloss, self.critic_optimizer, self.critic.parameters())
                self.updates += 1
                losses.append((float(loss.detach()), float(vloss.detach())))
        return losses

    @staticmethod
    def optimize(loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite baseline loss')
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(list(parameters), 10.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite baseline gradient')
        optimizer.step()

    def q_update(self):
        if len(self.replay) < 64:
            return []
        losses = []
        for _ in range(2):
            rows = self.random.sample(list(self.replay), 32)
            x, nx = self.t([r['x'] for r in rows]), self.t([r['next_x'] for r in rows])
            action = self.t([r['action'] for r in rows], torch.long)
            active, next_active = self.t([r['active'] for r in rows]), self.t([r['next_active'] for r in rows])
            q = self.actor(x).gather(-1,action.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                nq = self.target(nx).masked_fill(~self.t([r['next_legal'] for r in rows],torch.bool), -1e9).max(-1).values
                reward = self.t([r['reward'] for r in rows])
                terminal = self.t([r['done'] for r in rows])
                if self.algorithm in ('QMIX','VDN'):
                    nq = self.target_mixer(nq*next_active, self.t([r['next_state'] for r in rows]))
                    target = reward + self.all_args.gamma*(1-terminal)*nq
                else:
                    target = reward[:,None] + self.all_args.gamma*(1-terminal[:,None])*nq*next_active
            if self.algorithm in ('QMIX','VDN'):
                prediction = self.mixer(q*active,self.t([r['state'] for r in rows]))
                loss = (prediction-target).square().mean()
            else:
                loss = ((q-target).square()*active).sum()/active.sum().clamp_min(1)
            params = list(self.actor.parameters()) + (list(self.mixer.parameters()) if self.mixer else [])
            self.optimize(loss,self.actor_optimizer,params)
            self.updates += 1
            if self.updates % 100 == 0:
                self.target.load_state_dict(self.actor.state_dict())
                if self.mixer:
                    self.target_mixer.load_state_dict(self.mixer.state_dict())
            losses.append(float(loss.detach()))
        return losses

    def train(self):
        start = time.perf_counter()
        losses = (self.ppo_update() if self.algorithm in ('MAPPO','IPPO') else self.q_update()
                  if self.algorithm in ('QMIX','IQL','VDN') else self.coma_update() if self.algorithm=='COMA' else [])
        self.update_seconds += time.perf_counter()-start
        self.episodes += 1
        self.records.append(dict(episode=self.episodes, environment_steps=self.steps, optimizer_updates=self.updates, losses=losses))
        return {}

    def coma_update(self):
        rows=self.rollout
        x,state=self.t([r['x'] for r in rows]),self.t([r['state'] for r in rows])
        actions=self.t([r['action'] for r in rows],torch.long)
        legal=self.t([r['legal'] for r in rows],torch.bool)
        active=self.t([r['active'] for r in rows])
        with torch.no_grad():
            tq=self.target_critic(x,state,actions).gather(-1,actions.unsqueeze(-1)).squeeze(-1)
            returns=td_lambda_returns(self.t([r['reward'] for r in rows]).unsqueeze(-1),tq,self.all_args.gamma,self.all_args.gae_lambda)
        critic_losses=[]
        for idx in torch.randperm(len(rows),device=self.device).chunk(2):
            prediction=self.critic(x[idx],state[idx],actions[idx]).gather(-1,actions[idx].unsqueeze(-1)).squeeze(-1)
            loss=((prediction-returns[idx]).square()*active[idx]).sum()/active[idx].sum().clamp_min(1)
            self.optimize(loss,self.critic_optimizer,self.critic.parameters())
            critic_losses.append(float(loss.detach()))
        distribution=masked_distribution(self.actor(x),legal)
        with torch.no_grad():
            q=self.critic(x,state,actions)
            advantage=counterfactual_advantage(q,distribution.probs,actions)
        loss=-(distribution.log_prob(actions)*advantage*active).sum()/active.sum().clamp_min(1)
        loss-=.01*(distribution.entropy()*active).sum()/active.sum().clamp_min(1)
        self.optimize(loss,self.actor_optimizer,self.actor.parameters())
        self.updates+=1
        if (self.episodes+1)%20==0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        return dict(actor_loss=float(loss.detach()),critic_losses=critic_losses,actor_steps=1,critic_steps=2)

    def save_policy(self, directory):
        directory = Path(directory)
        (directory/'checkpoints').mkdir(parents=True,exist_ok=True)
        (directory/'logs').mkdir(parents=True,exist_ok=True)
        payload = dict(format='original_env_thesis_baseline_v1', algorithm=self.algorithm, actor=self.actor.state_dict(),
                       critic=self.critic.state_dict(), mixer=self.mixer.state_dict() if self.mixer else None,
                       horizon=80, agents=self.n, encoding='observed_graph_mlp_v1')
        torch.save(payload,directory/'checkpoints/baseline.pt')
        unchanged = self.loaded is None or all(self.loaded[k]==tensor_hash(getattr(self,k)) for k in self.loaded)
        report = dict(algorithm=self.algorithm, implementation='original_env_thesis_baseline_v1',
                      information='original global graph observations; no hidden channel loss/future labels',
                      environment_steps=self.steps, optimizer_updates=self.updates, episodes=self.episodes,
                      actor_changed=tensor_hash(self.actor)!=self.initial_actor,
                      critic_changed=tensor_hash(self.critic)!=self.initial_critic,
                      mixer_changed=self.mixer is not None and tensor_hash(self.mixer)!=self.initial_mixer,
                      evaluation=self.evaluating, weights_unchanged=unchanged,
                      trainable_parameters=sum(p.numel() for m in (self.actor,self.critic,self.mixer) if m for p in m.parameters()),
                      device=str(self.device), inference_calls=self.inference_calls, inference_seconds=self.inference_seconds,
                      update_seconds=self.update_seconds, records=self.records)
        (directory/'logs/baseline_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')

    def load_policy(self, path):
        payload = torch.load(path,map_location=self.device,weights_only=True)
        if payload.get('format')!='original_env_thesis_baseline_v1' or payload['algorithm']!=self.algorithm or payload['agents']!=self.n:
            raise ValueError('Incompatible baseline checkpoint')
        for key in ('actor','critic','mixer'):
            module = getattr(self,key)
            if module is not None:
                module.load_state_dict(payload[key])
        self.loaded = {key:tensor_hash(getattr(self,key)) for key in ('actor','critic','mixer') if getattr(self,key) is not None}
