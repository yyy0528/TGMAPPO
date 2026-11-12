"""Original TG policy weights and frozen evaluation; no training resume."""

import hashlib
import json
from pathlib import Path

import torch
from .wm_runtime import tensor_digest
from .task_reward import reward_contract
from .learning_rate import schedule_spec, optimizer_rates
from onpolicy.algorithms.utils.critic_time import time_mode
from onpolicy.algorithms.utils.critic_progress import context_mode
from onpolicy.algorithms.utils.pm_action_loss import action_credit_settings
from onpolicy.algorithms.utils.wm_fusion import residual_scale

TRAINING_IMPLEMENTATION = "tg_ppo_v3_lr_switch"


CONFIG_KEYS = ("grid_size", "comm_range", "num_moving_buoys", "random_channel", "episode_length",
               "hidden_size", "gnn_hidden_size", "gnn_num_heads", "gnn_layer_N", "gnn_concat_heads",
               "actor_graph_aggr", "critic_graph_aggr", "global_aggr_type", "graph_feat_type",
               "use_cent_obs", "use_recurrent_policy", "use_naive_recurrent_policy", "use_popart",
               "num_agents", "obs_type", "layer_N", "use_ReLU", "use_feature_normalization",
               "recurrent_N", "use_valuenorm", "stacked_frames", "num_embeddings", "embedding_size",
               "embed_hidden_size", "embed_layer_N", "embed_use_ReLU", "embed_add_self_loop", "gnn_use_ReLU")


def signature(learner):
    return {key: getattr(learner.all_args, key) for key in CONFIG_KEYS}


def load_policy(learner, path):
    try:
        payload = (torch.load(path, map_location="cpu", weights_only=True)
                   if int(torch.__version__.split('.')[0]) >= 2 else torch.load(path, map_location="cpu"))
        if (payload["format"] != "original_tg_policy_v1" or payload["config"] != signature(learner)
                or payload["use_wm_actor"] != (learner.wm_runtime is not None)):
            raise ValueError("Incompatible TG environment/config/WM mode")
        if payload.get("reward_contract", {"version": "legacy_v1"}) != reward_contract(learner.all_args):
            raise ValueError("Policy task reward version/weights mismatch")
        if learner.wm_runtime is not None and payload["wm"]["checkpoint_sha256"] != learner.wm_runtime.file_sha256:
            raise ValueError("Policy expects a different frozen WM")
        if payload.get("actor_extra_width", 0) != getattr(learner.all_args, "actor_extra_width", 0):
            raise ValueError("Actor expansion configuration mismatch")
        if payload.get('wm_actor_input', 'raw') != getattr(learner.all_args, 'wm_actor_input', 'raw'):
            raise ValueError('WM Actor input normalization mismatch')
        if payload.get('wm_actor_fusion', 'shared') != getattr(learner.all_args, 'wm_actor_fusion', 'shared'):
            raise ValueError('WM Actor fusion configuration mismatch')
        scale = residual_scale(learner.all_args)
        if (payload.get('wm_actor_residual_scale', 1.0) != scale
                or getattr(learner.policy.actor, 'wm_residual_scale', 1.0) != scale):
            raise ValueError('WM Actor residual scale mismatch')
        from onpolicy.algorithms.utils.policy_temperature import temperature_from_args
        if getattr(learner.policy.actor, 'policy_temperature', 1.0) != temperature_from_args(learner.all_args):
            raise ValueError('Runtime policy temperature differs from configuration')
        if payload.get('policy_temperature', 1.0) != temperature_from_args(learner.all_args):
            raise ValueError('Policy sampling temperature mismatch')
        expected_calibration=getattr(learner,'wm_calibration_metadata',None)
        if payload.get('wm_actor_calibration_sha256') != (expected_calibration['sha256'] if expected_calibration else None):
            raise ValueError('WM Actor calibration artifact mismatch')
        if getattr(learner.all_args, 'wm_actor_input', 'raw') == 'residual_scaled':
            for key in ('wm_input_scale','wm_input_center'):
                if not torch.allclose(payload['actor'][key], getattr(learner.policy.actor,key).cpu(), atol=1e-7, rtol=1e-6):
                    raise ValueError('WM Actor statistics differ from frozen train-split normalization')
        if payload.get("critic_time_mode","disabled") != time_mode(learner.all_args):
            raise ValueError("Critic clock configuration mismatch")
        if payload.get('pm_critic_context','disabled') != context_mode(learner.all_args):
            raise ValueError('PM Critic context configuration mismatch')
        if learner.wm_runtime is not None and payload["wm"].get("feature_source", "learned") != learner.wm_runtime.feature_source:
            raise ValueError("Candidate feature source mismatch")
        if payload.get("use_pm_reward", False) != (learner.pm_runtime is not None):
            raise ValueError("Policy PM mode mismatch")
        if learner.pm_runtime is not None and payload["pm"].get("potential_mode","remaining") != learner.pm_runtime.potential_mode:
            raise ValueError("Policy PM potential mode mismatch")
        if learner.pm_runtime is not None and payload["pm"].get("value_mode","learned_shaped") != learner.pm_runtime.value_mode:
            raise ValueError("Policy PM value coordinate mode mismatch")
        if learner.pm_runtime is not None:
            old_gate=payload['pm'].get('reliability')
            new_gate=learner.pm_runtime.reliability
            if (old_gate is None)!=(new_gate is None) or (old_gate is not None and
                    old_gate['report_sha256']!=new_gate['report_sha256']):
                raise ValueError('Policy PM reliability calibration mismatch')
        if learner.pm_runtime is not None and learner.pm_runtime.potential_mode == "task_return" and payload["pm"].get("task_return_contract") != learner.pm_runtime.task_contract:
            raise ValueError("Policy PM task-return contract mismatch")
        if learner.pm_runtime is not None and any(payload["pm"][key] != learner.pm_runtime.metadata()[key]
                                                for key in ("checkpoint_sha256", "alpha", "gamma")):
            raise ValueError("Policy expects different PM weights/alpha/gamma")
        expected_action_spec = action_credit_settings(learner.all_args)
        previous_action_spec = payload.get('pm_action_credit',dict(mode='disabled',coefficient=.1,temperature=.2))
        if previous_action_spec != expected_action_spec:
            raise ValueError('PM action credit configuration mismatch')
        networks = [("actor", learner.policy.actor), ("critic", learner.policy.critic)]
        if learner.trainer._use_valuenorm:
            networks.append(("value_normalizer", learner.trainer.value_normalizer))
        for name, network in networks:
            expected = network.state_dict()
            state = payload[name]
            if set(state) != set(expected):
                raise ValueError("Policy parameter names differ")
            for key, value in state.items():
                if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not torch.isfinite(value).all():
                    raise ValueError("Invalid policy tensor: " + key)
        for name, network in networks:
            network.load_state_dict(payload[name])
        learner.evaluation_metadata = {"source": str(Path(path).resolve()),
            "source_training_implementation": payload.get("training_implementation", "legacy_popart_v1"),
            "source_learning_rate_schedule":payload.get("learning_rate_schedule"),
            "source_critic_time_mode":payload.get("critic_time_mode","disabled"),
            "source_pm_critic_context":payload.get('pm_critic_context','disabled'),
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "actor_sha256": tensor_digest(learner.policy.actor.state_dict()),
            "critic_sha256": tensor_digest(learner.policy.critic.state_dict())}
    except Exception as exc:
        raise ValueError("Cannot load TG policy {}: {}".format(path, exc)) from None


def save_policy(learner, run_dir):
    root = Path(run_dir)
    wm = learner.wm_runtime.metadata() if learner.wm_runtime is not None else None
    pm = learner.pm_runtime.metadata() if learner.pm_runtime is not None else None
    wm_calibration = getattr(learner,'wm_calibration_metadata',None)
    from onpolicy.algorithms.utils.policy_temperature import temperature_from_args
    policy_temperature = temperature_from_args(learner.all_args)
    wm_scale = residual_scale(learner.all_args)
    if getattr(learner.policy.actor, 'wm_residual_scale', 1.0) != wm_scale:
        raise ValueError('Runtime WM scale differs from configuration')
    if getattr(learner.policy.actor, 'policy_temperature', 1.0) != policy_temperature:
        raise ValueError('Runtime policy temperature differs from configuration')
    resources = {"actor_trainable": sum(p.numel() for p in learner.policy.actor.parameters() if p.requires_grad),
                 "critic_trainable": sum(p.numel() for p in learner.policy.critic.parameters() if p.requires_grad),
                 "wm_frozen": sum(p.numel() for p in learner.wm_runtime.model.parameters()) if wm else 0,
                 "pm_frozen": sum(p.numel() for p in learner.pm_runtime.model.parameters()) if pm else 0,
                 "wm_active": bool(wm and learner.wm_runtime.feature_source == "learned"),
                 "device": str(learner.device), "cuda_peak_bytes": None}
    if learner.device.type == "cuda":
        resources["cuda_peak_bytes"] = torch.cuda.max_memory_allocated(learner.device)
    if getattr(learner.all_args, "tg_evaluate", False):
        metadata = dict(learner.evaluation_metadata)
        metadata["actor_unchanged"] = tensor_digest(learner.policy.actor.state_dict()) == metadata["actor_sha256"]
        metadata["critic_unchanged"] = tensor_digest(learner.policy.critic.state_dict()) == metadata["critic_sha256"]
        if not metadata["actor_unchanged"] or not metadata["critic_unchanged"]:
            raise ValueError("Frozen evaluation changed policy weights")
        metadata.update(mode="frozen_greedy_evaluation", wm=wm, pm=pm, optimizer_updates=0)
        metadata["evaluation_optimizer_rates"] = optimizer_rates(learner.policy)
    else:
        path = root / "checkpoints" / "tg_policy_latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "original_tg_policy_v1", "config": signature(learner),
            "training_implementation": TRAINING_IMPLEMENTATION,
            "learning_rate_schedule":schedule_spec(learner.all_args),
            "reward_contract": reward_contract(learner.all_args),
            "use_wm_actor": learner.wm_runtime is not None, "wm": wm,
            "use_pm_reward": learner.pm_runtime is not None, "pm": pm,
            "actor_extra_width": getattr(learner.all_args, "actor_extra_width", 0),
            "wm_actor_input": getattr(learner.all_args, 'wm_actor_input', 'raw'),
            "wm_actor_fusion": getattr(learner.all_args, 'wm_actor_fusion', 'shared'),
            "wm_actor_residual_scale": wm_scale,
            "policy_temperature": policy_temperature,
            "wm_actor_calibration_sha256": (wm_calibration['sha256'] if wm_calibration else None),
            "critic_time_mode":time_mode(learner.all_args),
            "pm_critic_context":context_mode(learner.all_args),
            "pm_action_credit":action_credit_settings(learner.all_args),
            "actor": learner.policy.actor.state_dict(), "critic": learner.policy.critic.state_dict(),
            "value_normalizer": (learner.trainer.value_normalizer.state_dict()
                                 if learner.trainer._use_valuenorm else None),
            "training_resume_supported": False}
        with path.open("xb") as stream:
            torch.save(payload, stream)
        metadata = {"mode": "training", "checkpoint": str(path.resolve()),
                    "training_implementation": TRAINING_IMPLEMENTATION,
                    "learning_rate_schedule":schedule_spec(learner.all_args),
                    "training_updates": learner.training_records,
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "wm": wm,
                    "wm_ppo_updates": learner.wm_update_records, "pm": pm,
                    "pm_ppo_updates": learner.pm_update_records, "training_resume_supported": False}
    metadata.update(profile=learner.profile, resources=resources, reward_contract=reward_contract(learner.all_args),
                    wm_actor_input=getattr(learner.all_args, 'wm_actor_input', 'raw'),
                    wm_actor_fusion=getattr(learner.all_args, 'wm_actor_fusion', 'shared'),
                    wm_actor_residual_scale=wm_scale,
                    policy_temperature=policy_temperature,
                    wm_actor_calibration=getattr(learner,'wm_calibration_metadata',None),
                    pm_critic_context=context_mode(learner.all_args),
                    pm_action_credit=action_credit_settings(learner.all_args),
                    pm_action_cache_audits=getattr(learner,'pm_action_cache_audits',[]),
                    critic_time_mode=time_mode(learner.all_args),
                    timing_scope="Collect/PPO host wall time; diagnostics separated; import/env/plot overhead excluded")
    with (root / "logs" / "TG_policy_report.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
