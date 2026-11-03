"""Explicit scripted reward controls, NOT evaluations of trained Actor decisions."""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from diagnose_tg_policy import build,read,sha,Agent,Runner,tensor_digest
import numpy as np


def run(suite, output, kind, reward_version="legacy_v1"):
    manifest = read(suite/"manifest.json")
    source = next(j for j in manifest["jobs"] if j["group"] == "A" and j["kind"] == "evaluation" and j["train_seed"] == 22)
    config = read(Path(source["output"])/"logs/TG_MAPPO_environment.json")["config"]
    config["tg_reward_version"] = reward_version
    env,learner = build(config,output)
    before = {"actor":tensor_digest(learner.policy.actor.state_dict()),"critic":tensor_digest(learner.policy.critic.state_dict())}
    rewards,attempts = [],[]
    def collect(step):
        actions = np.full((1,25,1),25,dtype=np.int64)
        if kind == "geographic":
            mask = env.available_actions()[0]
            for node in range(25):
                legal = [i for i in range(25) if mask[node,i]]
                if legal:
                    actions[0,node,0] = min(legal,key=lambda target:(math.hypot(
                        env.buoys[target].position[0]-env.sink[0],env.buoys[target].position[1]-env.sink[1]),target))
        values = np.zeros((1,25,1),dtype=np.float32)
        states = np.zeros((1,25,learner.recurrent_N,learner.hidden_size),dtype=np.float32)
        return values,actions,values.copy(),states,states.copy(),None
    original_insert,original_go = learner.insert,env.buoy_go
    def insert(data):
        rewards.append(float(np.asarray(data[5])[0,0,0]))
        return original_insert(data)
    def go(action,history,penalty):
        result = original_go(action,history,penalty)
        attempts.append({"step":env.current_step,"sender":action[0],"receiver":action[1],"history_before":float(history),
                         "penalty":penalty,"reward":float(result[0]),"success":bool(result[2]),"targets_sink":action[1] == env.sink_id})
        return result
    def forbidden(*args,**kwargs):
        raise AssertionError("Reward probe tried to train")
    learner.collect,learner.insert,env.buoy_go = collect,insert,go
    learner.train = learner.compute = forbidden
    learner.save_policy = lambda directory: None
    Runner([Agent(b,learner) for b in env.buoys],env).run()
    after = {"actor":tensor_digest(learner.policy.actor.state_dict()),"critic":tensor_digest(learner.policy.critic.state_dict())}
    if before != after:
        raise ValueError("Scripted reward probe changed model weights")
    rows = [json.loads(line) for line in (output/"moves_5/logs/TG_physical_metrics.jsonl").read_text().splitlines()]
    from buoy_network.task_reward import reward_contract
    spec = reward_contract(env.all_args)
    with (output/"moves_5/logs/TG_reward_components.csv").open(encoding="utf-8", newline="") as stream:
        reward_rows = list(csv.DictReader(stream))
    result = {"control":kind,"scope":"scripted environment/reward probe; Actor unused", "actor_executed":False,
              "weights_unchanged":True,"optimizer_updates":learner.profile["ppo_updates"],"scene":config["scenario_seed"],
              "reward_contract":spec,"return_channel":"configured base reward consumed by PPO; not raw legacy buoy_go",
              "episode_environment_returns":[sum(rewards[i*80:(i+1)*80]) for i in range(2)],
              "episode_terminal_rewards":[rewards[79],rewards[159]],"physical_episodes":rows,
              "first_successful_sink_attempt":next((a for a in attempts if a["targets_sink"] and a["success"]),None),
              "first_delivery_reward_components":next((r for r in reward_rows if int(r["delivered_this_step"]) > 0),None)}
    if reward_version == "task_v2":
        from buoy_network.task_reward_audit import audit_reward_csv
        result["reward_audit"] = audit_reward_csv(output/"moves_5/logs",spec)
    (output/"reward_probe.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    (output/"attempts.json").write_text(json.dumps(attempts,indent=2,allow_nan=False),encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True)
    parser.add_argument("--output-root",default="results/tg_reward_probe")
    parser.add_argument("--reward-version",choices=("legacy_v1","task_v2"),default="legacy_v1")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True,exist_ok=False)
    reports = [run(Path(args.suite).resolve(),root/kind,kind,args.reward_version) for kind in ("wait","geographic")]
    (root/"summary.json").write_text(json.dumps({"reports":reports},indent=2),encoding="utf-8")
    print("Reward probes: "+str(root/"summary.json"),flush=True)


if __name__ == "__main__":
    main()
