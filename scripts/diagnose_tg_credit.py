"""Exact isolated training replay plus same-input frozen Actor comparison."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RolloutCapture:
    """Never materialize lazy distribution properties in a training graph."""
    def __init__(self,actor):
        import torch
        self.last = None
        def captured(module,inputs,result):
            if not torch.is_grad_enabled():
                self.last = result.probs.detach().cpu().numpy().copy()
        self.handle = actor.act.action_out.register_forward_hook(captured)

    def close(self):
        self.handle.remove()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--seed",type=int,default=33)
    parser.add_argument("--output",required=True)
    parser.add_argument("--cross-control-suite")
    args = parser.parse_args()
    suite,output = Path(args.suite).resolve(),Path(args.output).resolve()
    m = read(suite/"manifest.json")
    snapshot = suite/"source_snapshot"
    for name,checksum in m["source"]["files_sha256"].items():
        if sha(snapshot/name) != checksum or sha(ROOT/name) != checksum:
            raise ValueError("Source differs from completed budget: "+name)
    # Import the archived training implementation; new reporting code stays separate.
    sys.path[:0] = [str(snapshot),str(snapshot/"src"),str(snapshot/"scripts")]
    import numpy as np
    import torch
    import diagnose_tg_policy as d
    helper_path = ROOT/"src/buoy_network/credit_diagnostics.py"
    spec = importlib.util.spec_from_file_location("credit_analysis",str(helper_path))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    job = next(j for j in m["jobs"] if j["kind"] == "training" and j["train_seed"] == args.seed and j["group"] == "A")
    reference = Path(job["output"])
    config = read(reference/"logs/TG_MAPPO_environment.json")["config"]
    if config["use_wm_actor"] or config["use_pm_reward"] or config["use_recurrent_policy"]:
        raise ValueError("This credit replay is restricted to feed-forward baseline A")
    output.mkdir(parents=True,exist_ok=False)
    env,learner = d.build(config,output/"replay")
    capture = RolloutCapture(learner.policy.actor)
    records,rollouts,context = [],[],{"episode":-1,"metrics":None,"rows":{},"episode_rows":[]}
    rows_stream = (output/"training_credit.jsonl").open("x",encoding="utf-8")

    class Metrics(d.PacketMetrics):
        def __init__(self,packets):
            super().__init__(packets)
            context["episode"] += 1
            context["metrics"],context["episode_rows"] = self,[]

        def attempt(self,packet_id,sender,receiver,success):
            row = context["rows"][sender]
            row.update(ack=bool(success),revisit=receiver in self.packets[packet_id]["route"])
            super().attempt(packet_id,sender,receiver,success)

    collect = learner.collect
    def observed_collect(step):
        result = collect(step)
        b,rows = learner.buffer,{}
        service = {}
        for packet in sorted(env.packets,key=lambda p:p.queue_order):
            if not packet.if_sink:
                service.setdefault(packet.agent_id,packet)
        for sender in np.flatnonzero(b.active_masks[step,0,:,0] > 0):
            sender = int(sender)
            action = int(result[1][0,sender,0])
            packet = service[sender]
            probs = capture.last[sender]
            rows[sender] = {"episode":context["episode"],"step":step,"sender":sender,"action":action,
                "sink":env.sink_id,"wait":env.buoy_number,"sink_legal":bool(b.available_actions[step,0,sender,env.sink_id]),
                "packet_id":packet.id,"route_before":list(context["metrics"].packets[packet.id]["route"]),
                "observation":b.obs[step,0,sender].tolist(),"node_self":b.node_obs[step,0,sender,sender].tolist(),
                "node_sink":b.node_obs[step,0,sender,env.sink_id].tolist(),
                "p_selected_before":float(probs[action]),"p_sink_before":float(probs[env.sink_id]),
                "ack":None,"revisit":False}
        context["rows"] = rows
        context["episode_rows"].extend(rows.values())
        return result
    learner.collect = observed_collect

    def probabilities(buffer):
        actor = learner.policy.actor
        training = actor.training
        rng = torch.get_rng_state().clone()
        n = buffer.episode_length*env.buoy_number
        def flat(name):
            data = getattr(buffer,name)[:-1]
            return data.reshape(n,*data.shape[3:])
        arrays = [flat(k) for k in ("obs","node_obs","adj","agent_id","rnn_states","masks","available_actions")]
        values = []
        try:
            actor.eval()
            with torch.no_grad(),torch.random.fork_rng(devices=[]):
                for start in range(0,n,100):
                    x = [a[start:start+100] for a in arrays]
                    actor(*x[:6],available_actions=x[6],deterministic=True)
                    values.append(capture.last.copy())
        finally:
            actor.train(training)
        if not torch.equal(rng,torch.get_rng_state()):
            raise AssertionError("Read-only probability probe consumed RNG")
        return np.concatenate(values).reshape(buffer.episode_length,env.buoy_number,-1)

    train = learner.trainer.train
    def observed_train(buffer,*a,**kw):
        values = learner.trainer.value_normalizer.denormalize(buffer.value_preds[:-1])
        raw,normalized = helper.advantages(buffer.returns[:-1],values,buffer.active_masks[:-1])
        rows = context["episode_rows"]
        for row in rows:
            t,i = row["step"],row["sender"]
            deliveries = sum(r["ack"] is True and r["action"] == env.sink_id for r in rows if r["step"] == t)
            row.update(advantage_raw=float(raw[t,0,i,0]),advantage_normalized=float(normalized[t,0,i,0]),
                value=float(values[t,0,i,0]),**{"return":float(buffer.returns[t,0,i,0])},
                reward=float(buffer.rewards[t,0,i,0]),deliveries=deliveries,
                other_deliveries=deliveries-int(row["ack"] is True and row["action"] == env.sink_id))
        result = train(buffer,*a,**kw)
        post = probabilities(buffer)
        for row in rows:
            p = post[row["step"],row["sender"]]
            row.update(p_selected_change=float(p[row["action"]])-row["p_selected_before"],
                       p_sink_change=float(p[env.sink_id])-row["p_sink_before"])
            rows_stream.write(json.dumps(row,allow_nan=False)+"\n")
        records.extend(rows)
        rollouts.append({"episode":context["episode"],"groups":helper.summarize(rows)})
        return result
    learner.trainer.train = observed_train
    try:
        with patch("buoy_network.runner.PacketMetrics",Metrics):
            d.Runner([d.Agent(b,learner) for b in env.buoys],env).run()
    finally:
        rows_stream.close()
        capture.close()
    current = output/"replay/moves_5"
    original = torch.load(str(reference/"checkpoints/tg_policy_latest.pt"),map_location="cpu")
    for name in ("actor","critic"):
        if d.tensor_digest(getattr(learner.policy,name).state_dict()) != d.tensor_digest(original[name]):
            raise AssertionError("Instrumented training changed final "+name)
    for name in ("TG_physical_metrics.jsonl","TG_reward_components.csv"):
        if (current/"logs"/name).read_bytes() != (reference/"logs"/name).read_bytes():
            raise AssertionError("Training replay differs: "+name)
    original_rows = read(reference/"logs/TG_policy_report.json")["training_updates"]
    if learner.training_records != original_rows:
        raise AssertionError("Original losses/updates differ after instrumentation")
    report = {"success":True,"suite":str(suite),"training_seed":args.seed,"ppo_epoch":config["ppo_epoch"],
        "exact_actor_critic":True,"exact_training_physics":True,"exact_reward_csv":True,"exact_training_updates":True,
        "environment_steps":learner.profile["collect_calls"],"groups":helper.summarize(records),"rollouts":rollouts,
        "row_count":len(records),"training_rows_sha256":sha(output/"training_credit.jsonl"),
        "reference_checkpoint_sha256":sha(reference/"checkpoints/tg_policy_latest.pt"),
        "script_sha256":sha(Path(__file__)),"helper_sha256":sha(helper_path),
        "limitation":"observational team credit, not causal action attribution; no gradients or targets changed"}
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print("PASS exact training replay: epoch {}, {} active decisions".format(config["ppo_epoch"],len(records)),flush=True)
    if args.cross_control_suite:
        control = read(Path(args.cross_control_suite)/"manifest.json")
        control_job = next(j for j in control["jobs"] if j["kind"] == "training" and j["train_seed"] == args.seed)
        _,other = d.build(config,output/"control_constructor")
        other.load_policy(Path(control_job["output"])/"checkpoints/tg_policy_latest.pt")
        other.policy.actor.eval()
        other_capture = RolloutCapture(other.policy.actor)
        control_before = d.tensor_digest(other.policy.actor.state_dict())
        build = d.build
        cross_rows = []
        def cross_build(c,o):
            e,l = build(c,o)
            if c.get("tg_evaluate"):
                original_collect = l.collect
                actual_capture = RolloutCapture(l.policy.actor)
                def cross_collect(step):
                    result = original_collect(step)
                    actual = actual_capture.last.copy()
                    b = l.buffer
                    with torch.no_grad(),torch.random.fork_rng(devices=[]):
                        other.policy.actor(b.obs[step,0],b.node_obs[step,0],b.adj[step,0],b.agent_id[step,0],
                            b.rnn_states[step,0],b.masks[step,0],available_actions=b.available_actions[step,0],deterministic=True)
                    for i in np.flatnonzero(b.active_masks[step,0,:,0] > 0):
                        cross_rows.append({"slot":l.profile["collect_calls"]-1,"sender":int(i),
                            "action":int(result[1][0,i,0]),"sink_legal":bool(b.available_actions[step,0,i,e.sink_id]),
                            "control_action":int(other_capture.last[i].argmax()),"probabilities":actual[i].tolist(),
                            "control_probabilities":other_capture.last[i].tolist(),"observation":b.obs[step,0,i].tolist(),
                            "node_self":b.node_obs[step,0,i,i].tolist(),"node_sink":b.node_obs[step,0,i,e.sink_id].tolist()})
                    return result
                l.collect = cross_collect
            return e,l
        with patch.object(d,"build",cross_build):
            diagnostic = d.worker(suite,output/"frozen_evaluation","A",args.seed,"trained","greedy")
        other_capture.close()
        if d.tensor_digest(other.policy.actor.state_dict()) != control_before:
            raise AssertionError("Same-input control Actor changed")
        if not cross_rows:
            raise AssertionError("No same-input frozen comparisons captured")
        cross_path = output/"cross_probabilities.jsonl"
        cross_path.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in cross_rows),encoding="utf-8")
        sink_rows = [r for r in cross_rows if r["sink_legal"]]
        report["frozen_same_input"] = {"exact_physics":diagnostic["pilot_metrics_exact_match"],
            "control_actor_unchanged":True,
            "rows":len(cross_rows),"sink_legal":len(sink_rows),
            "sink_misses_current":sum(r["action"] != 24 for r in sink_rows),
            "sink_misses_control":sum(r["control_action"] != 24 for r in sink_rows),
            "cross_rows_sha256":sha(cross_path),"control_checkpoint_sha256":sha(Path(control_job["output"])/"checkpoints/tg_policy_latest.pt")}
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        print("PASS exact frozen replay with same-input comparison",flush=True)


if __name__ == "__main__":
    main()
