"""Verify step-6 real runs, frozen weights, alpha=0, and per-step reward algebra."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.buoy_network.wm_runtime import tensor_digest


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(root):
    with (root/"logs"/"TG_MAPPO_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = [{key: float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    require(len(rows) == 2 and all(math.isfinite(value) for row in rows for value in row.values()), "Invalid episode metrics")
    return rows


def inspect(root):
    root = Path(root)
    report = load(root/"logs"/"TG_policy_report.json")
    config = load(root/"logs"/"TG_MAPPO_environment.json")["config"]
    require(config["num_agents"] == 25 and config["episode_length"] == 80 and config["effective_num_env_steps"] == 160,
            "Wrong real-run budget")
    result = {"run": str(root), "metrics": metrics(root), "report": report,
              "config": {key:config[key] for key in ("seed", "scenario_seed", "num_episodes", "episode_length",
                "num_moving_buoys", "random_channel", "gamma", "use_pm_reward", "pm_reward_alpha", "use_wm_actor", "tg_evaluate")}}
    for name in ("pm", "wm"):
        if report[name] is not None:
            info = report[name]
            require(info["frozen"], name+" not frozen")
            require(hashlib.sha256(Path(info["checkpoint"]).read_bytes()).hexdigest() == info["checkpoint_sha256"], name+" file changed")
    if report["mode"] == "training":
        require(hashlib.sha256(Path(report["checkpoint"]).read_bytes()).hexdigest() == report["checkpoint_sha256"], "Policy file changed")
        for row in report["pm_ppo_updates"]:
            require(all(row.values()), "PM PPO validation failed")
        for row in report["wm_ppo_updates"]:
            require(row["wm_frozen"] and row["ratio_max_error"] <= 1e-4 and row["actor_changed"] and row["critic_changed"], "WM PPO failed")
    else:
        require(report["optimizer_updates"] == 0 and report["actor_unchanged"] and report["critic_unchanged"], "Evaluation changed weights")
        require(config["seed"] == config["scenario_seed"] == 404, "Unexpected evaluation seed")
    if report["pm"] is not None:
        with (root/"logs"/"TG_PM_rewards.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 160, "PM reward rows missing")
        gamma, alpha = report["pm"]["gamma"], report["pm"]["alpha"]
        require(gamma == config["gamma"] and alpha == config["pm_reward_alpha"], "PM/PPO discount mismatch")
        errors, progress_values = [], []
        for episode in range(2):
            current = [row for row in rows if int(row["episode"]) == episode]
            require([int(row["step"]) for row in current] == list(range(80)), "PM time alignment")
            discounted = 0.
            for step, row in enumerate(current):
                env, progress, train, before, after = [float(row[key]) for key in ("reward_env", "reward_progress", "reward_train", "phi_t", "phi_tp1")]
                require(all(math.isfinite(x) for x in (env, progress, train, before, after)), "Nonfinite PM log")
                require(math.isclose(progress, alpha*(gamma*after-before), abs_tol=1e-10), "Wrong potential difference")
                require(math.isclose(train, env+progress, abs_tol=1e-10), "Wrong shaped reward")
                require(row["terminated"] == str(step == 79) and row["rollout_truncated"] == "False", "Wrong terminal flag")
                if step:
                    require(float(current[step-1]["phi_tp1"]) == before, "PM history continuity")
                discounted += gamma**step*progress
                progress_values.append(progress)
            require(float(current[-1]["phi_tp1"]) == 0, "Terminal potential not zero")
            error = abs(discounted+alpha*float(current[0]["phi_t"]))
            require(error < 1e-7, "Discounted boundary identity failed")
            errors.append(error)
        require(alpha == 0 or any(value != 0 for value in progress_values), "Shaping is a placeholder")
        require(alpha != 0 or all(value == 0 for value in progress_values), "alpha=0 changed rewards")
        result["reward_audit"] = {"rows": len(rows), "boundary_errors": errors,
                                  "minimum_progress": min(progress_values), "maximum_progress": max(progress_values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-list", required=True)
    parser.add_argument("--pretrain-report", required=True)
    parser.add_argument("--test-reports", nargs="+", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    runs = {name:inspect(path) for name,path in load(args.run_list).items()}
    require(runs["alpha0"]["metrics"] == runs["baseline"]["metrics"], "alpha=0 metric regression")
    baseline = torch.load(runs["baseline"]["report"]["checkpoint"], map_location="cpu", weights_only=True)
    zero = torch.load(runs["alpha0"]["report"]["checkpoint"], map_location="cpu", weights_only=True)
    for key in ("actor", "critic"):
        require(tensor_digest(baseline[key]) == tensor_digest(zero[key]), "alpha=0 "+key+" differs")
    for name in ("baseline", "wm", "pm", "wm_pm"):
        train, evaluated = runs[name]["report"], runs["eval_"+name]["report"]
        require(train["checkpoint_sha256"] == evaluated["checkpoint_sha256"], "Wrong evaluated policy")
        for model in ("pm", "wm"):
            if train[model] is not None:
                require(train[model]["state_sha256"] == evaluated[model]["state_sha256"], model+" state changed")
    require(runs["eval_wm_pm"]["metrics"] == runs["eval_wm_pm_repeat"]["metrics"], "Combined evaluation not reproducible")
    require(runs["eval_wm_pm"]["report"]["checkpoint_sha256"] == runs["eval_wm_pm_repeat"]["report"]["checkpoint_sha256"], "Repeat loaded another policy")
    tests, count = [], 0
    for path in args.test_reports:
        report = load(path)
        require(report["success"] and {row["name"] for row in report["stages"]} == {"functions", "modules", "integration"}, "Incomplete test receipt")
        for row in report["stages"]:
            require(row["tests"] > 0 and not any(row.get(key, 0) for key in ("failures", "errors", "skipped")), "Test failure/skip")
            count += row["tests"]
        tests.append(report)
    for module, number in (("tests.test_tg_progress_model", 5), ("tests.test_tg_world_model", 6)):
        output = subprocess.run([sys.executable, "-m", "unittest", module, "-v"], cwd=str(ROOT),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=60)
        require(output.returncode == 0 and "Ran {} tests".format(number) in output.stdout, "Core unit tests failed: "+output.stdout)
        tests.append({"module": module, "tests": number, "exit_code": output.returncode, "log": output.stdout})
        count += number
    pretrain = load(args.pretrain_report)
    require(pretrain["gate_passed"] and not pretrain["test_evaluated"], "PM pretraining gate/test violation")
    sources = ("src/buoy_network/progress_model.py", "src/buoy_network/progress_runtime.py", "src/buoy_network/agent.py",
               "src/buoy_network/runner.py", "src/buoy_network/main.py", "src/buoy_network/policy_artifacts.py",
               "src/legacy_cli.py", "src/onpolicy/utils/util.py", "tests/test_legacy_targeted_fixes.py",
               "tests/test_tg_progress_model.py", "tests/test_legacy_pm_reward.py", "scripts/train_tg_pm.py",
               "scripts/test_tg_pm.py", "scripts/verify_tg_pm_runs.py")
    receipt = {"success": True, "scope": "censored-time PM minimal integration, not full outcome PM or performance acceptance",
               "test_count": count, "tests": tests, "runs": runs, "pretraining": pretrain,
               "source_files_sha256": {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
               "limitations": ["no true TTL/drop failure supervision", "one actor training seed, short budget", "no performance improvement shown",
                               "constant reliability gate", "CPU only", "no cross-rollout task continuation"]}
    with Path(args.report).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print("PASS: {} tests; alpha=0 exact baseline; reward algebra and frozen evaluations verified".format(count))
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
