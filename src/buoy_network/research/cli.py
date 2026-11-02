"""Strict CLI. Defaults live in config.py; execution lives in component classes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .bootstrap import build_runner, default_registry, load_extensions
from .config import ExperimentConfig
from .data.storage import DatasetStore, episode_split
from .runtime.artifacts import RunArtifacts
from .runtime.device import resolve_device
from .training.base import TrainingContext

# CLI spelling -> canonical config path. Both kebab-case and snake_case work.
OPTIONS = {
    "env-version": ("environment.version", str),
    "env-backend": ("environment.backend", str),
    "grid-size": ("environment.grid_size", int),
    "comm-range": ("environment.comm_range", float),
    "num-moving-buoys": ("environment.num_moving_buoys", int),
    "mobility-std": ("environment.mobility_std", float),
    "loss-probability": ("environment.loss_probability", float),
    "task-horizon": ("environment.task_horizon", int),
    "task-ttl": ("environment.task_ttl", int),
    "reward-mode": ("environment.reward_mode", str),
    "policy": ("policy.backend", str),
    "policy-trainer": ("policy.trainer", str),
    "policy-hidden-size": ("policy.hidden_size", int),
    "policy-graph-layers": ("policy.graph_layers", int),
    "policy-checkpoint": ("policy.checkpoint", str),
    "policy-learning-rate": ("policy.learning_rate", float),
    "ppo-epochs": ("policy.ppo_epochs", int),
    "minibatch-steps": ("policy.minibatch_steps", int),
    "clip-param": ("policy.clip_param", float),
    "entropy-coef": ("policy.entropy_coef", float),
    "value-loss-coef": ("policy.value_loss_coef", float),
    "max-grad-norm": ("policy.max_grad_norm", float),
    "rollout-length": ("policy.rollout_length", int),
    "gamma": ("policy.gamma", float),
    "gae-lambda": ("policy.gae_lambda", float),
    "wm-backend": ("world_model.backend", str),
    "wm-checkpoint": ("world_model.checkpoint", str),
    "wm-history-len": ("world_model.history_len", int),
    "wm-horizon": ("world_model.horizon", int),
    "wm-hidden-size": ("world_model.hidden_size", int),
    "wm-graph-hidden-size": ("world_model.graph_hidden_size", int),
    "wm-graph-layers": ("world_model.graph_layers", int),
    "wm-batch-size": ("world_model.batch_size", int),
    "wm-learning-rate": ("world_model.learning_rate", float),
    "wm-trainer": ("world_model.trainer", str),
    "pm-backend": ("progress.backend", str),
    "pm-checkpoint": ("progress.checkpoint", str),
    "pm-alpha": ("progress.alpha", float),
    "pm-heads": ("progress.heads", int),
    "pm-update-mode": ("progress.update_mode", str),
    "pm-hidden-size": ("progress.hidden_size", int),
    "pm-batch-size": ("progress.batch_size", int),
    "pm-learning-rate": ("progress.learning_rate", float),
    "pm-trainer": ("progress.trainer", str),
    "seed": ("runtime.seed", int),
    "episodes": ("runtime.episodes", int),
    "device": ("runtime.device", str),
    "output-root": ("runtime.output_root", str),
    "run-name": ("runtime.run_name", str),
    "dataset": ("runtime.dataset", str),
}


def boolean(text: str) -> bool:
    value = text.lower()
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("Expected true or false")
    return value == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buoy research architecture and experiment entry")
    parser.add_argument("command", choices=(
        "describe", "check-config", "smoke", "collect", "evaluate",
        "train-policy", "train-wm", "train-pm",
    ))
    parser.add_argument("--config", help="Nested JSON config; relative paths use the working directory")
    parser.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                        help="Import MODULE.register_components(registry); repeatable")
    options = dict(OPTIONS)
    options.update({
        "policy-deterministic": ("policy.deterministic", boolean),
        "use-world-model": ("world_model.enabled", boolean),
        "use-progress-reward": ("progress.enabled", boolean),
        "use-earp": ("policy.use_earp", boolean),
    })
    for name, (destination, value_type) in options.items():
        spellings = ["--" + name]
        if "-" in name:
            spellings.append("--" + name.replace("-", "_"))
        parser.add_argument(*spellings, dest=destination, type=value_type,
                            default=argparse.SUPPRESS)
    return parser


def _train(command, config, device, registry):
    section = {"train-policy": config.policy, "train-wm": config.world_model,
               "train-pm": config.progress}[command]
    # Fail before making a run directory when the neural trainer is not installed.
    registry.require("trainer", section.trainer)
    dataset = None
    if command != "train-policy":
        if not config.runtime.dataset:
            raise ValueError("Auxiliary training requires --dataset PATH")
        dataset = DatasetStore(config.runtime.dataset)
        manifest = dataset.manifest()
        if manifest["env_version"] != config.environment.version:
            raise ValueError("Dataset environment version is incompatible")
        if manifest["num_nodes"] != config.environment.num_nodes:
            raise ValueError("Dataset node/action dimensions are incompatible with the configuration")
        if (manifest.get("environment_backend") is not None
                and manifest["environment_backend"] != config.environment.backend):
            raise ValueError("Dataset environment backend is incompatible")
        if any(not manifest["splits"].get(split) for split in ("train", "validation", "test")):
            raise ValueError("Collect at least 3 complete episodes before auxiliary training")
    trainer = registry.build("trainer", section.trainer, config, device)
    artifacts = RunArtifacts.create(config, device, command)
    report = trainer.fit(TrainingContext(config, device, artifacts, registry, dataset))
    artifacts.write_json("training_report.json", asdict(report))
    return {"command": command, "output": str(artifacts.root), "report": asdict(report)}


def execute(args) -> dict:
    overrides = {key: value for key, value in vars(args).items() if "." in key}
    # Validate the final merged configuration so CLI can override a file value.
    config = ExperimentConfig.load(args.config, validate=False).with_overrides(overrides)
    if (args.command == "evaluate" and config.policy.backend == "graph_mappo"
            and config.policy.deterministic is None):
        config = config.with_overrides({"policy.deterministic": True})
    registry = default_registry()
    load_extensions(registry, args.plugin)
    if args.command == "describe":
        return {
            "implemented": registry.describe(),
            "planned": {"world_model": "local_graph_gru",
                        "progress_model": "prefix_gru",
                        "trainers": ["supervised_wm", "supervised_pm"]},
            "extension_hook": "register_components(registry)",
            "documentation": "docs/RESEARCH_ARCHITECTURE.md",
        }
    device = resolve_device(config.runtime.device)
    if args.command == "check-config":
        return {"valid": True, "resolved_device": device, "config": config.to_dict()}
    if args.command.startswith("train-"):
        return _train(args.command, config, device, registry)

    if config.policy.checkpoint and config.policy.backend != "graph_mappo":
        raise ValueError("A policy checkpoint currently requires --policy graph_mappo")
    if args.command == "evaluate" and config.policy.backend == "graph_mappo" and not config.policy.checkpoint:
        raise ValueError("Learned policy evaluation requires --policy-checkpoint PATH")

    runner = build_runner(config, device, registry)
    try:
        artifacts = RunArtifacts.create(config, device, args.command)
        store = DatasetStore(artifacts.root / "data")
        rows, episode_ids = [], []
        episodes = 1 if args.command == "smoke" else config.runtime.episodes
        for episode in range(episodes):
            episode_id = "episode_{:06d}".format(episode)
            seed = config.runtime.seed + episode

            def verify_gae(rollout):
                advantages, returns = rollout.compute_gae(config.policy.gamma, config.policy.gae_lambda)
                if not advantages or not returns:
                    raise ValueError("Empty GAE results")

            records = runner.run_episode(episode_id, seed, on_rollout=verify_gae)
            if args.command != "evaluate":
                store.write_episode(records, runner.environment.packet_traces())
                episode_ids.append(episode_id)
            rows.append(runner.metrics(records, episode))
        if episode_ids:
            store.write_manifest(episode_split(
                episode_ids, config.runtime.seed, config.data.train_fraction,
                config.data.validation_fraction,
            ), config.environment.version, environment_backend=config.environment.backend)
        artifacts.write_metrics(rows)
        report = {"command": args.command, "device": device, "output": str(artifacts.root),
                  "environment": config.environment.backend, "policy": config.policy.backend,
                  "episodes": episodes, "metrics": rows,
                  "model_versions": {"policy": runner.policy.version,
                                     "world_model": runner.world_model.version,
                                     "progress_model": runner.progress_model.version}}
        artifacts.write_json("run_report.json", report)
        return report
    finally:
        runner.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except (ValueError, OSError, ImportError) as exc:
        print("Research command failed: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
