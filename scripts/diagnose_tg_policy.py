"""Read-only original-TG zero-delivery diagnosis; no training, no policy fix."""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/"src"):
    sys.path.insert(0,str(directory))
import numpy as np
import torch
from buoy_network.main import parse_args
from buoy_network.environment import buoy_environment
from buoy_network.agent import Agent, DeepQNetwork
from buoy_network.runner import Runner
from buoy_network.physical_metrics import PacketMetrics
from buoy_network.wm_runtime import tensor_digest
from buoy_network.policy_diagnostics import TraceAnalysis


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DistributionCapture:
    def __init__(self, actor):
        self.actor, self.last = actor, None
        if actor.use_wm:
            self.original = actor._wm_distribution
            def forward(*args, **kwargs):
                result = self.original(*args, **kwargs)
                self.last = result.probs.detach().cpu().numpy().copy()
                return result
            actor._wm_distribution = forward
            self.handle = None
        else:
            def captured(module, inputs, result):
                self.last = result.probs.detach().cpu().numpy().copy()
            self.handle = actor.act.action_out.register_forward_hook(captured)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
        else:
            self.actor._wm_distribution = self.original


def build(config, output):
    args = SimpleNamespace(**dict(config, output_root=str(output)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = buoy_environment([[10,[i,j]] for i in range(args.grid_size) for j in range(args.grid_size)],args)
    env.build_env()
    learner = DeepQNetwork(max(env.get_n_action(i) for i in range(env.buoy_number)),
        max(env.get_n_obs(i) for i in range(env.buoy_number)),learning_rate=.001,reward_decay=.9,
        e_greedy=1,replace_target_iter=200,memory_size=100000,env=env)
    return env,learner


def worker(suite, output, group, seed, stage, decode):
    manifest = read(suite/"manifest.json")
    for name,checksum in manifest["source"]["files_sha256"].items():
        if (ROOT/name).is_file() and sha(ROOT/name) != checksum:
            raise ValueError("Core differs from pilot snapshot: "+name)
    training = next(j for j in manifest["jobs"] if j["group"] == group and j["train_seed"] == seed and j["kind"] == "training")
    evaluation = next(j for j in manifest["jobs"] if j["group"] == group and j["train_seed"] == seed and j["kind"] == "evaluation")
    train_config = read(Path(training["output"])/"logs/TG_MAPPO_environment.json")["config"]
    eval_config = read(Path(evaluation["output"])/"logs/TG_MAPPO_environment.json")["config"]
    # Reconstruct initialization under the original TRAINING seed, then reset all RNGs for evaluation.
    _,initial = build(train_config,output/"initial_constructor")
    initial_actor = initial.policy.actor
    initial_actor.eval()
    initial_actor_sha = tensor_digest(initial_actor.state_dict())
    initial_critic = {key:value.clone() for key,value in initial.policy.critic.state_dict().items()}
    initial_capture = DistributionCapture(initial_actor)
    env,learner = build(eval_config,output)
    trained_path = Path(training["output"])/"checkpoints/tg_policy_latest.pt"
    if stage == "trained":
        learner.load_policy(trained_path)
    else:
        if group != "A":
            raise ValueError("Full initial-policy replay is restricted to baseline A")
        learner.policy.actor.load_state_dict(initial_actor.state_dict())
        learner.policy.critic.load_state_dict(initial_critic)
    before = {"actor":tensor_digest(learner.policy.actor.state_dict()),"critic":tensor_digest(learner.policy.critic.state_dict())}
    initial_vector = torch.cat([p.detach().flatten() for p in initial_actor.parameters()])
    executed_vector = torch.cat([p.detach().flatten() for p in learner.policy.actor.parameters()])
    parameter_delta = float((executed_vector-initial_vector).norm())
    capture = DistributionCapture(learner.policy.actor)
    analysis = TraceAnalysis(env.buoy_number,env.sink_id)
    output.mkdir(parents=True,exist_ok=False)
    trace = (output/"slots.jsonl").open("x",encoding="utf-8")
    routes = (output/"packet_routes.jsonl").open("x",encoding="utf-8")
    context = {"episode":-1,"positions":None,"frame":None,"events":None}

    class TracedMetrics(PacketMetrics):
        def __init__(self,packets):
            super().__init__(packets)
            context["episode"] += 1
        def attempt(self,packet_id,sender,receiver,success):
            super().attempt(packet_id,sender,receiver,success)
            analysis.attempt(sender,receiver,success,context["positions"])
            context["events"].append({"packet_id":packet_id,"sender":sender,"receiver":receiver,"ack":bool(success)})
        def summary(self,episode,horizon):
            result = super().summary(episode,horizon)
            analysis.finish_episode(result,self.packets)
            for key,value in sorted(self.packets.items()):
                routes.write(json.dumps(dict(episode=episode,packet_id=key,route=value["route"],delivered=value["delay"] is not None,
                                              delivery_step=value["delay"]),allow_nan=False)+"\n")
            return result

    get_actions = learner.policy.get_actions
    def decoded(*args,**kwargs):
        kwargs["deterministic"] = decode == "greedy"
        return get_actions(*args,**kwargs)
    learner.policy.get_actions = decoded
    collect,insert = learner.collect,learner.insert
    def observed_collect(step):
        result = collect(step)
        buffer = learner.buffer
        wm = buffer.wm_features[step,0] if buffer.wm_features is not None else None
        with torch.no_grad():
            initial_actor(buffer.obs[step,0],buffer.node_obs[step,0],buffer.adj[step,0],buffer.agent_id[step,0],
                buffer.rnn_states[step,0],buffer.masks[step,0],available_actions=buffer.available_actions[step,0],
                deterministic=True,wm_features=wm)
        actions = result[1][0,:,0].astype(int).tolist()
        positions = buffer.node_obs[step,0,0,:,:2].tolist()
        mask = buffer.available_actions[step,0].tolist()
        analysis.decision(actions,capture.last,initial_capture.last,mask,positions,env.packets,env.all_args.comm_range)
        context["positions"],context["events"] = positions,[]
        context["frame"] = {"episode":context["episode"],"step":step,"actions":actions,"mask":mask,
            "probabilities":capture.last.tolist(),"initial_probabilities_same_inputs":initial_capture.last.tolist(),
            "positions":positions,"service_nodes":[p.agent_id for p in env.packets if not p.if_sink]}
        return result
    def observed_insert(data):
        frame = context["frame"]
        frame["events"] = context["events"]
        frame["reward_train"] = float(np.asarray(data[5])[0,0,0])
        frame["reward_env"] = (learner.pm_runtime.episode_records[-1]["reward_env"] if learner.pm_runtime is not None else frame["reward_train"])
        trace.write(json.dumps(frame,allow_nan=False)+"\n")
        return insert(data)
    learner.collect,learner.insert = observed_collect,observed_insert
    def no_training(*args,**kwargs):
        raise AssertionError("Diagnosis attempted PPO/returns computation")
    learner.train = learner.compute = no_training
    # Diagnostic decode may be stochastic: do not write the production report's fixed 'greedy' label.
    learner.save_policy = lambda run_dir: None
    try:
        with patch("buoy_network.runner.PacketMetrics",TracedMetrics):
            Runner([Agent(b,learner) for b in env.buoys],env).run()
    finally:
        trace.close()
        routes.close()
        capture.close()
        initial_capture.close()
    after = {"actor":tensor_digest(learner.policy.actor.state_dict()),"critic":tensor_digest(learner.policy.critic.state_dict())}
    if after != before or initial_actor_sha != tensor_digest(initial_actor.state_dict()):
        raise ValueError("Read-only diagnosis changed policy weights")
    for runtime in (learner.wm_runtime,learner.pm_runtime):
        if runtime is not None:
            runtime.assert_frozen()
    result = {"group":group,"training_seed":seed,"stage":stage,"decode":decode,"eval_scene":eval_config["scenario_seed"],
              "policy_sha256":sha(trained_path),"initial_actor_sha256":initial_actor_sha,"weights_unchanged":True,
              "actor_delta_from_initial_l2":parameter_delta,"actor_initial_l2":float(initial_vector.norm()),
              "optimizer_updates":learner.profile["ppo_updates"],"torch_threads":torch.get_num_threads(),
              "analysis":analysis.summary(),"output":str(output)}
    if stage == "trained" and decode == "greedy":
        reference = (Path(evaluation["output"])/"logs/TG_physical_metrics.jsonl").read_text(encoding="utf-8")
        reproduced = (output/"moves_5/logs/TG_physical_metrics.jsonl").read_text(encoding="utf-8")
        if reference != reproduced:
            raise ValueError("Instrumented replay differs from original pilot physical metrics")
        result["pilot_metrics_exact_match"] = True
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("DIAG {} seed {} {} {}: {}/{} delivered, {} revisit hops".format(group,seed,stage,decode,
        result["analysis"]["delivered"],result["analysis"]["generated"],result["analysis"]["revisit_transfers"]),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--output-root",default="results/tg_policy_diagnostics")
    parser.add_argument("--worker",nargs=2,metavar=("GROUP","SEED"))
    parser.add_argument("--stage",choices=("trained","initial"),default="trained")
    parser.add_argument("--decode",choices=("greedy","sample"),default="greedy")
    parser.add_argument("--output-run")
    args = parser.parse_args()
    suite = Path(args.suite).resolve()
    if args.worker:
        worker(suite,Path(args.output_run).resolve(),args.worker[0],int(args.worker[1]),args.stage,args.decode)
        return
    root = Path(args.output_root).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True,exist_ok=False)
    cases = [(g,s,"trained","greedy") for g in "ABCDEF" for s in (11,22,33)]
    cases += [("A",s,"initial","greedy") for s in (11,22,33)]
    cases += [("A",s,"trained","sample") for s in (11,22,33)]
    reports, commands = [], []
    print("Diagnostic batch: "+str(root),flush=True)
    for group,seed,stage,decode in cases:
        tag = "{}_{}_{}_{}".format(group,seed,stage,decode)
        command = [sys.executable,str(Path(__file__).resolve()),"--suite",str(suite),"--worker",group,str(seed),
                   "--stage",stage,"--decode",decode,"--output-run",str(root/tag)]
        print("START "+tag,flush=True)
        with (root/(tag+".log")).open("xb") as stream:
            result = subprocess.run(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        commands.append({"command":command,"exit_code":result.returncode})
        (root/"commands.json").write_text(json.dumps(commands,indent=2),encoding="utf-8")
        if result.returncode:
            raise ValueError("Diagnostic replay failed: "+str(root/(tag+".log")))
        reports.append(read(root/tag/"report.json"))
        print("DONE "+tag,flush=True)
    (root/"summary.json").write_text(json.dumps({"scope":"read-only diagnosis, not a policy fix or new ablation result",
        "suite":str(suite),"reports":reports,"cases":len(reports),"all_weights_unchanged":all(r["weights_unchanged"] for r in reports)},indent=2,allow_nan=False),encoding="utf-8")
    print("Complete: "+str(root/"summary.json"),flush=True)


if __name__ == "__main__":
    main()
