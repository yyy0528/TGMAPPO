"""Original TG experiment orchestration; stdlib only, Python 3.6 compatible."""
import contextlib
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("A", "B", "C", "D", "E", "F")
DEVELOPMENT_SEEDS = {0, 1, 2, 3, 101, 202, 303, 404}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def validate(config):
    config = dict(config)
    config["auxiliary_reports"] = dict(config.get("auxiliary_reports", {}))
    required = {"schema", "purpose", "python", "groups", "train_seeds", "eval_seeds", "train_episodes",
                "eval_episodes", "episode_length", "ppo_epoch", "num_mini_batch", "num_moving_buoys",
                "random_channel", "wm_checkpoint", "wm_sha256", "pm_checkpoint", "pm_sha256",
                "target_delivery_rate", "target_window", "job_timeout_seconds", "auxiliary_reports"}
    require(set(config) == required and config["schema"] == "original_tg_ablation_v1", "Unknown/incomplete suite schema")
    for field in ("python", "wm_checkpoint", "pm_checkpoint"):
        config[field] = str((ROOT/config[field]).resolve())
    config["auxiliary_reports"] = {name:str((ROOT/path).resolve()) for name,path in config["auxiliary_reports"].items()}
    require(config["purpose"] in ("pilot", "formal"), "Unknown purpose")
    require(set(config["groups"]).issubset(GROUPS) and len(config["groups"]) == len(set(config["groups"]))
            and {"A", "B", "C", "D"}.issubset(config["groups"]), "Require unique A-D groups; E/F optional")
    for field in ("train_seeds", "eval_seeds"):
        values = config[field]
        require(values and all(type(v) is int and 0 <= v < 2**32 for v in values)
                and len(values) == len(set(values)), "Invalid/duplicate seeds: "+field)
    require(len(config["train_seeds"]) >= 3, "Need at least three independent training seeds")
    require(not set(config["train_seeds"]) & set(config["eval_seeds"]), "Train/evaluation seed overlap")
    require(not set(config["eval_seeds"]) & DEVELOPMENT_SEEDS, "Evaluation overlaps prior development scenes")
    if config["purpose"] == "formal":
        require(7001 not in config["eval_seeds"], "Pilot scene cannot be a formal blind scene")
    for field in ("train_episodes", "eval_episodes", "episode_length", "ppo_epoch", "num_mini_batch", "target_window", "job_timeout_seconds"):
        require(type(config[field]) is int and config[field] > 0, "Invalid positive integer: "+field)
    require(config["episode_length"] == 80 and 2000 % config["num_mini_batch"] == 0, "Require trained horizon and exact PPO minibatches")
    require(type(config["num_moving_buoys"]) is int and 0 <= config["num_moving_buoys"] <= 25
            and type(config["random_channel"]) is bool, "Invalid scene options")
    require(0 < config["target_delivery_rate"] <= 1 and Path(config["python"]).is_file(), "Invalid target/interpreter")
    for name in ("wm", "pm"):
        require(digest(config[name+"_checkpoint"]) == config[name+"_sha256"], name+" checkpoint SHA mismatch")
    require(set(config["auxiliary_reports"]) == {"wm", "pm", "data"}, "Need auxiliary pretraining/data provenance")
    for path in config["auxiliary_reports"].values():
        require(Path(path).is_file(), "Missing auxiliary report: "+path)
    return config


def budget(config):
    n = len(config["groups"])*len(config["train_seeds"])
    return {"purpose": config["purpose"], "training_jobs": n, "evaluation_jobs": n*len(config["eval_seeds"]),
            "training_steps_per_job": config["train_episodes"]*80,
            "total_training_steps": n*config["train_episodes"]*80,
            "total_evaluation_steps": n*len(config["eval_seeds"])*config["eval_episodes"]*80,
            "independent_training_seeds": config["train_seeds"], "evaluation_scenes": config["eval_seeds"]}


def variant(config, group):
    args = []
    if group in ("B", "D", "F"):
        args += ["--use_wm_actor", "true", "--wm_actor_checkpoint", str(Path(config["wm_checkpoint"]).resolve())]
    if group in ("C", "D"):
        args += ["--use_pm_reward", "true", "--pm_reward_checkpoint", str(Path(config["pm_checkpoint"]).resolve()), "--pm_reward_alpha", "1"]
    if group == "E":
        args += ["--actor_extra_width", "17"]
    if group == "F":
        args += ["--wm_feature_source", "history"]
    return args


def jobs(config, suite):
    suite = Path(suite).resolve()
    result = []
    common = [config["python"], str(suite/"source_snapshot"/"run_algorithm.py"), "--algorithm", "tg-mappo",
              "--episode_length", "80", "--num_moving_buoys", str(config["num_moving_buoys"]),
              "--random_channel", str(config["random_channel"]).lower(), "--ppo_epoch", str(config["ppo_epoch"]),
              "--num_mini_batch", str(config["num_mini_batch"])]
    for group in config["groups"]:
        for seed in config["train_seeds"]:
            train_key = "{}_seed_{}".format(group, seed)
            result.append({"key": train_key, "kind": "training", "group": group, "train_seed": seed,
                           "state": "pending", "attempts": [], "base_command": common+variant(config, group)+[
                               "--seed", str(seed), "--scenario_seed", str(seed), "--num_episodes", str(config["train_episodes"]),
                               "--num_env_steps", str(config["train_episodes"]*80)]})
            for scene in config["eval_seeds"]:
                result.append({"key": train_key+"_scene_{}".format(scene), "kind": "evaluation", "group": group,
                               "train_seed": seed, "scene": scene, "depends_on": train_key, "state": "pending", "attempts": [],
                               "base_command": common+variant(config, group)+["--seed", str(scene), "--scenario_seed", str(scene),
                                   "--tg_evaluate", "true", "--num_episodes", str(config["eval_episodes"]),
                                   "--num_env_steps", str(config["eval_episodes"]*80)]})
    return result


def snapshot(suite):
    target = suite/"source_snapshot"
    files = [ROOT/"run_algorithm.py"]
    for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.json")):
        files.extend((ROOT/folder).rglob(pattern))
    hashes = {}
    for path in sorted(files):
        require(ROOT in path.resolve().parents, "Source escapes workspace")
        relative = path.relative_to(ROOT)
        destination = target/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        hashes[str(relative)] = digest(destination)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return {"git_parent": git.stdout.strip() if git.returncode == 0 else None, "files_sha256": hashes}


@contextlib.contextmanager
def suite_lock(root):
    stream = (root/"runner.lock").open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def inspect_job(job, config):
    root = Path(job["output"])
    for path, checksum in job.get("artifacts_sha256", {}).items():
        require(digest(root/path) == checksum, "Completed artifact changed: "+path)
    report_path = root/"logs"/"TG_policy_report.json"
    report = read(report_path)
    actual = read(root/"logs"/"TG_MAPPO_environment.json")["config"]
    seed = job["train_seed"] if job["kind"] == "training" else job["scene"]
    require(actual["seed"] == actual["scenario_seed"] == seed and actual["ppo_epoch"] == config["ppo_epoch"]
            and actual["num_mini_batch"] == config["num_mini_batch"] and actual["num_moving_buoys"] == config["num_moving_buoys"]
            and actual["random_channel"] == config["random_channel"], "Job options differ from frozen protocol")
    require(actual["use_wm_actor"] == (job["group"] in ("B", "D", "F"))
            and actual["use_pm_reward"] == (job["group"] in ("C", "D"))
            and actual["actor_extra_width"] == (17 if job["group"] == "E" else 0)
            and actual["wm_feature_source"] == ("history" if job["group"] == "F" else "learned"), "Wrong ablation variant")
    require(report["profile"]["collect_calls"] == (config["train_episodes"] if job["kind"] == "training" else config["eval_episodes"])*80,
            "Wrong collected step count")
    if job["kind"] == "training":
        require(report["mode"] == "training" and digest(report["checkpoint"]) == report["checkpoint_sha256"], "Invalid policy artifact")
    else:
        require(report["mode"] == "frozen_greedy_evaluation" and report["optimizer_updates"] == 0
                and report["actor_unchanged"] and report["critic_unchanged"], "Evaluation updated weights")
        require(report["checkpoint_sha256"] == job["expected_policy_sha256"], "Evaluation loaded a different policy")
    for name in ("wm", "pm"):
        if report[name] is not None:
            require(report[name]["frozen"] and report[name]["checkpoint_sha256"] == config[name+"_sha256"], "Wrong frozen auxiliary version")
    rows = [json.loads(line) for line in (root/"logs"/"TG_physical_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows) == (config["train_episodes"] if job["kind"] == "training" else config["eval_episodes"]), "Missing physical episodes")
    for row in rows:
        require(row["schema"] == "original_tg_physical_v1" and row["generated_packets"] == row["delivered_packets"]+row["unresolved_packets"], "Packet metric conservation")
    return report, rows


def summarize(manifest):
    config = manifest["config"]
    require(all(job["state"] == "completed" for job in manifest["jobs"]), "Cannot summarize incomplete suite")
    by_seed = []
    for group in config["groups"]:
        for seed in config["train_seeds"]:
            selected = [job for job in manifest["jobs"] if job["group"] == group and job["train_seed"] == seed]
            train = next(job for job in selected if job["kind"] == "training")
            report, train_rows = inspect_job(train, config)
            rows = [row for job in selected if job["kind"] == "evaluation" for row in inspect_job(job, config)[1]]
            generated = sum(row["generated_packets"] for row in rows)
            delivered = sum(row["delivered_packets"] for row in rows)
            hops = [value for row in rows for value in row["success_hops"]]
            delays = sorted(value for row in rows for value in row["success_delays"])
            threshold = None
            window = config["target_window"]
            for index in range(window-1, len(train_rows)):
                if statistics.mean(r["delivery_rate"] for r in train_rows[index-window+1:index+1]) >= config["target_delivery_rate"]:
                    threshold = (index+1)*80
                    break
            by_seed.append({"group": group, "train_seed": seed, "generated": generated, "delivered": delivered,
                            "delivery_rate": delivered/generated, "unresolved_fraction": 1-delivered/generated,
                            "success_hops_mean": statistics.mean(hops) if hops else None,
                            "success_delay_mean": statistics.mean(delays) if delays else None,
                            "success_delay_p95": delays[math.ceil(.95*len(delays))-1] if delays else None,
                            "retry_attempts": sum(r["retry_attempts"] for r in rows),
                            "loop_transfers": sum(r["loop_transfers"] for r in rows), "true_drop_rate": None,
                            "target_first_training_steps": threshold, "train_worker_seconds": train["attempts"][-1]["seconds"],
                            "resources": report["resources"], "profile": report["profile"]})
    groups = []
    for group in config["groups"]:
        rows = [row for row in by_seed if row["group"] == group]
        item = {"group": group, "independent_training_n": len(rows), "metrics": {}}
        for field in ("delivery_rate", "unresolved_fraction", "success_hops_mean", "success_delay_mean", "success_delay_p95", "retry_attempts", "loop_transfers"):
            values = [row[field] for row in rows if row[field] is not None]
            item["metrics"][field] = {"mean": statistics.mean(values) if values else None,
                                      "sample_std": statistics.stdev(values) if len(values) > 1 else None, "n_defined": len(values)}
        groups.append(item)
    return {"purpose": config["purpose"], "budget": budget(config), "by_training_seed": by_seed, "groups": groups,
            "statistical_unit": "independent trained policy, not evaluation episode", "drop_rate_supported": False,
            "limitations": ["pilot is not convergence or significance evidence", "one frozen WM/PM pretraining seed",
                            "legacy history rewards persist between episodes", "CPU wall timing; CUDA unmeasured",
                            "E matches trainable extra Actor capacity within 2.2%, not frozen-model total capacity",
                            "F uses same four-frame input scope and scorer; idle WM loaded but never called"]}


def run_suite(root):
    root = Path(root).resolve()
    require(ROOT in root.parents, "Suite must remain within workspace")
    with suite_lock(root):
        manifest = read(root/"manifest.json")
        config = validate(manifest["config"])
        for path, checksum in manifest["source"]["files_sha256"].items():
            require(digest(root/"source_snapshot"/path) == checksum, "Source snapshot changed: "+path)
        completed = {}
        for job in manifest["jobs"]:
            if job["state"] == "completed":
                inspect_job(job, config)
                completed[job["key"]] = job
                continue
            attempt = len(job["attempts"])+1
            name = job["key"]+"_attempt_{}".format(attempt)
            command = list(job["base_command"])+["--output_root", str(root/job["kind"]), "--run-name", name]
            if job["kind"] == "evaluation":
                parent = completed[job["depends_on"]]
                parent_report, _ = inspect_job(parent, config)
                command += ["--tg_policy_checkpoint", parent_report["checkpoint"]]
                job["expected_policy_sha256"] = parent_report["checkpoint_sha256"]
            job["state"] = "running"
            job["output"] = str(root/job["kind"]/"runs"/name/"moves_{}".format(config["num_moving_buoys"]))
            record = {"command": command, "started": datetime.now().isoformat(), "seconds": None, "exit_code": None}
            job["attempts"].append(record)
            manifest["state"] = "running"
            atomic(root/"manifest.json", manifest)
            print("START "+name, flush=True)
            started = time.perf_counter()
            process = None
            try:
                with (root/"job_logs"/(name+".log")).open("xb") as stream:
                    process = subprocess.Popen(command, cwd=str(root/"source_snapshot"), stdout=stream, stderr=subprocess.STDOUT)
                    record["exit_code"] = process.wait(timeout=config["job_timeout_seconds"])
                require(record["exit_code"] == 0, "Job failed: "+name)
                inspect_job(job, config)
                job["artifacts_sha256"] = {"logs/"+name: digest(Path(job["output"])/"logs"/name)
                    for name in ("TG_policy_report.json", "TG_physical_metrics.jsonl", "TG_MAPPO_environment.json", "TG_MAPPO_metrics.csv")}
                job["state"] = "completed"
                completed[job["key"]] = job
            except BaseException as exc:
                if process is not None and process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        process.kill()
                    process.wait()
                job["state"], manifest["state"] = "failed", "failed"
                record["error"] = str(exc)
                raise
            finally:
                record["seconds"] = time.perf_counter()-started
                atomic(root/"manifest.json", manifest)
            print("DONE "+name, flush=True)
        result = summarize(manifest)
        atomic(root/"summary.json", result)
        manifest["state"] = "completed"
        atomic(root/"manifest.json", manifest)
        print("COMPLETE: "+str(root/"summary.json"), flush=True)
        return result


def create_suite(config, output_root):
    config = validate(config)
    base = Path(output_root).resolve()
    require(base == ROOT or ROOT in base.parents, "Output must remain within workspace")
    root = base/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    (root/"job_logs").mkdir()
    source = snapshot(root)
    probe = subprocess.run([config["python"], "-c", "import json,sys,platform,torch,pkg_resources; print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(), torch=torch.__version__, cuda_available=torch.cuda.is_available(), versions={n:pkg_resources.get_distribution(n).version for n in ['numpy','scipy','gym','torch-geometric']})))"],
                           cwd=str(root/"source_snapshot"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    require(probe.returncode == 0, "Dependency probe failed: "+probe.stderr)
    atomic(root/"dependencies.json", json.loads(probe.stdout))
    for name, path in config["auxiliary_reports"].items():
        (root/("auxiliary_"+name+".json")).write_bytes(Path(path).read_bytes())
    manifest = {"schema": "original_tg_suite_run_v1", "state": "planned", "config": config,
                "budget": budget(config), "source": source, "jobs": jobs(config, root)}
    atomic(root/"manifest.json", manifest)
    return root
