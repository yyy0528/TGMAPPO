"""On-policy clipped PPO using complete joint-observation time minibatches."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict

import torch

from ..contracts import TrainingReport
from ..runtime.checkpoints import CheckpointMetadata
from .base import Trainer, TrainingContext


class GraphMAPPOTrainer(Trainer):
    def __init__(self, config, device):
        if config.policy.backend != "graph_mappo":
            raise ValueError("The graph_mappo trainer requires --policy graph_mappo")
        if config.world_model.enabled or config.progress.enabled or config.policy.use_earp:
            raise ValueError("The baseline PPO trainer currently requires WM/PM/EARP disabled")
        if config.policy.checkpoint:
            raise ValueError("Policy weights are for evaluation; training resume/warm-start is not implemented")
        if config.policy.deterministic:
            raise ValueError("PPO training requires stochastic sampling; disable --policy-deterministic")
        self.config, self.device = config, torch.device(device)
        self.policy = None
        self.optimizer = None
        self.minibatch_rng = torch.Generator().manual_seed(config.runtime.seed)

    def _parameter_vector(self, actor):
        parameters = [parameter.detach().reshape(-1) for name, parameter in self.policy.network.named_parameters()
                      if (name.startswith("actor_") or name == "wait_embedding") == actor]
        return torch.cat(parameters).clone()

    def update(self, rollout):
        records = rollout.transitions
        if not records or any(record.action_batch.policy_version != self.policy.version for record in records):
            raise ValueError("PPO requires a fresh rollout from the current behavior policy")
        advantages, returns = rollout.compute_gae(self.config.policy.gamma, self.config.policy.gae_lambda)
        observations = tuple(record.obs_t for record in records)
        features = tuple(record.wm_features for record in records)
        actions = torch.tensor([record.action_batch.actions for record in records],
                               dtype=torch.long, device=self.device)
        old_logp = torch.tensor([record.action_batch.log_probs for record in records],
                                dtype=torch.float32, device=self.device)
        old_values = torch.tensor([record.action_batch.values for record in records],
                                  dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        active = torch.tensor([[local.active and any(local.available_actions[:-1]) for local in obs.agents]
                               for obs in observations], dtype=torch.bool, device=self.device)
        if active.any():
            selected = advantages[active]
            advantages = (advantages - selected.mean()) / selected.std(unbiased=False).clamp(min=1e-5)
        advantages = advantages.masked_fill(~active, 0.0)
        with torch.no_grad():
            behavior, _ = self.policy.distribution(observations, features)
            replay = behavior.log_prob(actions)
            ratio_error = (torch.exp(replay - old_logp) - 1.0).abs().max().item()
        if ratio_error > 1e-4:
            raise ValueError("PPO behavior probability replay mismatch")
        actor_before, critic_before = self._parameter_vector(True), self._parameter_vector(False)
        totals = {name: 0.0 for name in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
        batches = 0
        settings = self.config.policy
        self.policy.network.train()
        for _ in range(settings.ppo_epochs):
            permutation = torch.randperm(len(records), generator=self.minibatch_rng).tolist()
            for start in range(0, len(records), settings.minibatch_steps):
                indices = permutation[start:start + settings.minibatch_steps]
                distribution, values = self.policy.distribution(
                    tuple(observations[index] for index in indices), tuple(features[index] for index in indices))
                logp = distribution.log_prob(actions[indices])
                log_ratio = logp - old_logp[indices]
                ratio = torch.exp(log_ratio)
                mask = active[indices]
                if mask.any():
                    objective = torch.minimum(ratio * advantages[indices],
                        ratio.clamp(1.0 - settings.clip_param, 1.0 + settings.clip_param) * advantages[indices])
                    actor_loss = -objective[mask].mean()
                    entropy = distribution.entropy()[mask].mean()
                else:
                    # No actor graph: Adam must skip actor parameters, including momentum.
                    actor_loss = values.new_zeros(())
                    entropy = values.new_zeros(())
                clipped = old_values[indices] + (values - old_values[indices]).clamp(-settings.clip_param, settings.clip_param)
                value_loss = 0.5 * torch.maximum((values - returns[indices]).square(),
                                                 (clipped - returns[indices]).square()).mean()
                loss = actor_loss - settings.entropy_coef * entropy + settings.value_loss_coef * value_loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PPO loss; optimizer was not stepped")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), settings.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise ValueError("Nonfinite PPO gradients; optimizer was not stepped")
                self.optimizer.step()
                with torch.no_grad():
                    totals["actor_loss"] += actor_loss.item()
                    totals["value_loss"] += value_loss.item()
                    totals["entropy"] += entropy.item()
                    if mask.any():
                        totals["approx_kl"] += ((ratio - 1.0) - log_ratio)[mask].mean().item()
                        totals["clip_fraction"] += ((ratio - 1.0).abs() > settings.clip_param)[mask].float().mean().item()
                    totals["grad_norm"] += grad_norm.item()
                batches += 1
        self.policy.network.eval()
        self.policy.updates += 1
        self.policy.version = "graph_mappo_v1:{}".format(self.policy.updates)
        metrics = {name: value / batches for name, value in totals.items()}
        metrics.update(active_fraction=active.float().mean().item(),
                       initial_ratio_max_error=ratio_error, minibatches=batches,
                       actor_parameter_delta=torch.linalg.vector_norm(self._parameter_vector(True) - actor_before).item(),
                       critic_parameter_delta=torch.linalg.vector_norm(self._parameter_vector(False) - critic_before).item())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite PPO update metrics")
        return metrics

    def fit(self, context: TrainingContext) -> TrainingReport:
        from ..bootstrap import build_runner
        runner = build_runner(context.config, context.device, context.registry)
        self.policy = runner.policy
        self.optimizer = torch.optim.Adam(self.policy.network.parameters(), lr=context.config.policy.learning_rate)
        initial_actor, initial_critic = self._parameter_vector(True), self._parameter_vector(False)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        metrics_rows, update_rows = [], []
        started = time.perf_counter()
        try:
            for episode in range(context.config.runtime.episodes):
                runner.reset("train_{:06d}".format(episode), context.config.runtime.seed + episode)
                records = []
                while not runner.terminated:
                    rollout = runner.collect_rollout()
                    update = self.update(rollout)
                    records.extend(rollout.transitions)
                    update_rows.append(dict(update=self.policy.updates, episode=episode,
                                            environment_steps=len(records), **update))
                metrics_rows.append(runner.metrics(tuple(records), episode))
            context.artifacts.write_metrics(metrics_rows)
            with (context.artifacts.root / "model_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(update_rows[0]))
                writer.writeheader()
                writer.writerows(update_rows)
            path = context.artifacts.root / "checkpoints" / "policy_latest.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = CheckpointMetadata.for_config(context.config, "policy", self.policy.version)
            payload = {
                "metadata": asdict(metadata), "model_version": self.policy.version,
                "architecture_version": self.policy.architecture_version,
                "hidden_size": context.config.policy.hidden_size, "graph_layers": context.config.policy.graph_layers,
                "updates": self.policy.updates, "model_state": self.policy.network.state_dict(),
            }
            with path.open("xb") as stream:
                torch.save(payload, stream)
            metadata.save(path.with_suffix(".metadata.json"))
            metrics = dict(environment_steps=sum(row["environment_steps"] for row in metrics_rows),
                updates=len(update_rows), optimizer_steps=sum(row["minibatches"] for row in update_rows),
                actor_parameter_delta=torch.linalg.vector_norm(self._parameter_vector(True) - initial_actor).item(),
                critic_parameter_delta=torch.linalg.vector_norm(self._parameter_vector(False) - initial_critic).item(),
                parameter_count=sum(parameter.numel() for parameter in self.policy.network.parameters()),
                training_seconds=time.perf_counter() - started, torch_threads=1,
                critic_information="joint_observations", recurrent_policy=False)
            return TrainingReport("policy", self.policy.version, metrics, str(path))
        finally:
            runner.close()
            torch.set_num_threads(old_threads)
