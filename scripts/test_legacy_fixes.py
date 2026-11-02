"""Run targeted TG-MAPPO regressions and real CPU training (Python 3.6+)."""

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src", ROOT / "tests"):
    sys.path.insert(0, str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/targeted_fixes")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_MODE", "disabled")
    try:
        import test_legacy_targeted_fixes as regressions
        import test_legacy_budget as budgets
        import test_plot_comparison as comparison
        import test_legacy_transition_data as datasets
    except unittest.SkipTest as exc:
        parser.error(str(exc))
    destination = Path(args.output_root).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"python": sys.executable, "scope": "legacy_fixed_v1 targeted regressions", "stages": []}
    stages = (
        ("functions", (regressions.GeometryAndChannelFixes, budgets.TrainingBudgetTests,
                       datasets.SnapshotFunctionTests)),
        ("modules", (regressions.ActionAndRolloutFixes, comparison.ComparisonPlotTests,
                     datasets.RecorderModuleTests)),
        ("integration", (regressions.ArgumentFixes, regressions.NeuralIntegrationFixes,
                         datasets.ExportProcessIntegrationTests)),
    )
    success = True
    for name, classes in stages:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
        output = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        log_name = name + ".log"
        (destination / log_name).write_text(output.getvalue(), encoding="utf-8")
        stage = {"name": name, "tests": result.testsRun, "failures": len(result.failures),
                 "errors": len(result.errors), "skipped": len(result.skipped),
                 "seconds": round(time.perf_counter() - started, 3), "log": log_name}
        report["stages"].append(stage)
        print("{}: {} tests, {} failures, {} errors".format(name, result.testsRun, stage["failures"], stage["errors"]), flush=True)
        success = result.wasSuccessful() and not result.skipped
        if not success:
            print(output.getvalue())
            break
    if success and not args.skip_smoke:
        command = [sys.executable, str(ROOT / "run_algorithm.py"), "--algorithm", "tg-mappo",
                   "--num_episodes", "2", "--episode_length", "80", "--num_env_steps", "160",
                   "--ppo_epoch", "1", "--num_mini_batch", "2", "--num_moving_buoys", "25",
                   "--random_channel", "true", "--output-root", str(destination / "smoke")]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, timeout=180)
            (destination / "smoke.stdout.log").write_bytes(completed.stdout)
            (destination / "smoke.stderr.log").write_bytes(completed.stderr)
            smoke = {"command": command, "exit_code": completed.returncode,
                     "seconds": round(time.perf_counter() - started, 3)}
            paths = list((destination / "smoke").rglob("TG_MAPPO_metrics.csv"))
            if completed.returncode == 0 and len(paths) == 1:
                with paths[0].open(encoding="utf-8", newline="") as stream:
                    smoke["episodes"] = list(csv.DictReader(stream))
                smoke["metrics"] = str(paths[0].relative_to(destination))
            success = completed.returncode == 0 and len(smoke.get("episodes", [])) == 2
        except subprocess.TimeoutExpired as exc:
            smoke = {"command": command, "error": "Training exceeded 180 seconds"}
            (destination / "smoke.stdout.log").write_bytes(exc.output or b"")
            (destination / "smoke.stderr.log").write_bytes(exc.stderr or b"")
            success = False
        report["smoke"] = smoke
        print("Real training: " + ("PASS" if success else "FAIL"), flush=True)
    report["success"] = success
    report["tests"] = sum(stage["tests"] for stage in report["stages"])
    report_path = destination / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Report: " + str(report_path))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
