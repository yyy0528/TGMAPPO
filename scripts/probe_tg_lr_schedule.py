"""Observe actual optimizer learning rates in a small original-TG run."""
import argparse
import contextlib
import hashlib
import io
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",default=str(ROOT))
    parser.add_argument("--linear",action="store_true")
    parser.add_argument("--output-root",default="results/tg_lr_probes")
    options = parser.parse_args()
    source = Path(options.source_root).resolve()
    sys.path[:0] = [str(source),str(source/"src")]
    import numpy as np
    import torch
    from onpolicy.config import get_config,graph_config
    from buoy_network.main import parse_args
    from buoy_network.environment import buoy_environment
    from buoy_network.agent import Agent,DeepQNetwork
    from buoy_network.runner import Runner
    from buoy_network.wm_runtime import tensor_digest
    output = Path(options.output_root).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    argv = ["--grid_size","5","--num_agents","25","--num_episodes","2","--episode_length","4",
            "--ppo_epoch","1","--num_mini_batch","2","--n_rollout_threads","1","--use_cent_obs","False","--max_edge_dist","1.5",
            "--num_moving_buoys","0","--scenario_seed","11","--output_root",str(output)]
    if options.linear:
        argv.append("--use_linear_lr_decay")
    args,config_parser = parse_args(argv,get_config())
    args,_ = graph_config(argv,config_parser)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    torch.set_num_threads(1)
    env = buoy_environment([[10,[i,j]] for i in range(5) for j in range(5)],args)
    env.build_env()
    learner = DeepQNetwork(26,26,env=env)
    rates = []
    train = learner.train
    def capture():
        rates.append({"episode":len(rates),"actor":learner.policy.actor_optimizer.param_groups[0]["lr"],
                      "critic":learner.policy.critic_optimizer.param_groups[0]["lr"]})
        return train()
    learner.train = capture
    with contextlib.redirect_stdout(io.StringIO()):
        Runner([Agent(b,learner) for b in env.buoys],env).run()
    report = {"source_root":str(source),"source_runner_sha256":hashlib.sha256((source/"src/buoy_network/runner.py").read_bytes()).hexdigest(),
              "linear_requested":args.use_linear_lr_decay,"optimizer_rates":rates,
              "actor_sha256":tensor_digest(learner.policy.actor.state_dict()),
              "critic_sha256":tensor_digest(learner.policy.critic.state_dict()),"output":str(output)}
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
