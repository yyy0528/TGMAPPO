"""Collect disjoint legacy TG simulator runs for WM (Python 3.6 environment)."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from buoy_network.dataset import audit_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 101, 202])
    parser.add_argument("--behaviors", nargs="+", choices=("tg-mappo", "random", "geographic"),
                        default=["tg-mappo", "random", "geographic"])
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per independent seed/run")
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--num-moving-buoys", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--max-bytes", type=int, default=268435456)
    parser.add_argument("--max-packets", type=int, default=10000)
    parser.add_argument("--output-root", default="results/tg_wm_data")
    args = parser.parse_args()
    if (len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds)):
        parser.error("Require at least 3 distinct uint32 seeds; last two are validation/test")
    if len(args.behaviors) != len(set(args.behaviors)):
        parser.error("Duplicate data behaviors")
    requested = len(args.seeds) * len(args.behaviors) * args.episodes * args.episode_length
    if (min(args.episodes, args.episode_length, args.max_steps, args.max_bytes, args.max_packets) <= 0
            or not 0 <= args.num_moving_buoys <= 25 or requested > args.max_steps
            or 24 * args.episodes * len(args.seeds) * len(args.behaviors) > args.max_packets):
        parser.error("Invalid settings or full collection would exceed the data budget")
    destination = Path(args.output_root).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": "legacy_tg_collection_v1", "complete": False,
                "node_count": 25, "requested_steps": requested, "runs": [],
                "split_rule": "independent processes; disjoint explicit scenario seeds; full runs/episodes",
                "normalization": "raw data only; fit on training split in future WM trainer",
                "behaviors": args.behaviors,
                "behavior": "TG-MAPPO online PPO; rules collect without PPO; independent processes",
                "budgets": {"max_steps": args.max_steps, "max_bytes": args.max_bytes,
                            "max_packets": args.max_packets}}

    def save_manifest():
        temporary = destination / "dataset_manifest.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
        os.replace(str(temporary), str(destination / "dataset_manifest.json"))

    save_manifest()
    used_bytes = 0
    print("Collection: " + str(destination), flush=True)
    try:
        jobs = [(index, seed, behavior) for index, seed in enumerate(args.seeds) for behavior in args.behaviors]
        for index, seed, behavior in jobs:
            split = "test" if index == len(args.seeds) - 1 else (
                "validation" if index == len(args.seeds) - 2 else "train")
            command = [sys.executable, str(ROOT / "run_algorithm.py"), "--algorithm", "tg-mappo",
                       "--data_behavior", behavior,
                       "--seed", str(seed), "--scenario_seed", str(seed),
                       "--num_episodes", str(args.episodes), "--episode_length", str(args.episode_length),
                       "--num_env_steps", str(args.episodes * args.episode_length),
                       "--ppo_epoch", "1", "--num_mini_batch", "2",
                       "--num_moving_buoys", str(args.num_moving_buoys), "--random_channel", "true",
                       "--export_wm_data", "true", "--wm_data_split", split,
                       "--wm_export_max_steps", str(args.max_steps),
                       "--wm_export_max_bytes", str(args.max_bytes - used_bytes),
                       "--wm_export_max_packets", str(args.max_packets),
                       "--output-root", str(destination / "source_runs"), "--run-name", "{}_seed_{}".format(behavior, seed)]
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, timeout=600)
            tag = "{}_seed_{}".format(behavior, seed)
            (destination / (tag + ".stdout.log")).write_bytes(completed.stdout)
            (destination / (tag + ".stderr.log")).write_bytes(completed.stderr)
            with (destination / (tag + ".command.json")).open("x", encoding="utf-8") as stream:
                json.dump({"command": command, "exit_code": completed.returncode,
                           "seconds": time.perf_counter() - started}, stream, indent=2)
            if completed.returncode:
                raise ValueError("Seed {} exited {}; see stderr/stdout logs".format(seed, completed.returncode))
            data_path = destination / "source_runs" / "runs" / tag / (
                "moves_{}".format(args.num_moving_buoys)) / "wm_data"
            report = audit_run(data_path)
            used_bytes += report["bytes"]
            manifest["runs"].append({"scenario_seed": seed, "policy_seed": seed, "split": split, "behavior": behavior,
                                     "data_path": str(data_path.relative_to(destination)), "audit": report})
            save_manifest()
            print("{} {} seed {}: {} steps, {} delivered, {} unresolved; audit PASS".format(
                split, behavior, seed, report["steps"], report["delivered"], report["unresolved"]), flush=True)
        manifest.update(complete=True, total_steps=sum(row["audit"]["steps"] for row in manifest["runs"]),
                        total_jsonl_bytes=used_bytes)
        save_manifest()
        print("Dataset: " + str(destination), flush=True)
        return 0
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        manifest["error"] = str(exc)
        save_manifest()
        print("Collection failed: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
