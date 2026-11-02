"""Run function -> module -> integration tests and write a reproducible report.

PyTorch is required for the real graph policy/PPO checks; PyG/Gym are not.
Coverage counts executed named concrete functions, not lines/branches or neural
WM/PM algorithms still unimplemented.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

STAGES = {
    "functions": ("test_research_functions.py", "test_research_graph_policy.py",
                  "test_research_ppo.py", "test_research_policy_checkpoint.py"),
    "modules": ("test_research_architecture.py", "test_research_modules.py"),
    "integration": ("test_research_integration.py",),
}


def inventory():
    functions, interfaces = {}, []
    for path in sorted((ROOT / "src/buoy_network/research").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        relative = path.relative_to(ROOT).as_posix()

        def visit(node, parents=()):
            for child in ast.iter_child_nodes(node):
                scope = parents
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope = parents + (child.name,)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    label = relative + ":" + ".".join(scope)
                    if any(isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
                           for decorator in child.decorator_list):
                        interfaces.append(label)
                    else:
                        first_line = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        functions[(str(path.resolve()).casefold(), first_line)] = label
                visit(child, scope)

        visit(tree)
    return functions, interfaces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--output-root", default="results/functional_tests")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(args.output_root).resolve() / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    functions, interfaces = inventory()
    called = set()

    def profile(frame, event, arg):
        if event == "call":
            key = (frame.f_code.co_filename.casefold(), frame.f_code.co_firstlineno)
            if key in functions:
                called.add(key)

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    report = {
        "scope": "src/buoy_network/research concrete functions and root/module CLI entrypoints",
        "python": sys.version, "timezone": "Asia/Shanghai", "stages": [],
        "limitations": [
            "Function calls are measured in the test process; subprocess CLI runs are verified by exit codes and artifacts.",
            "Abstract methods are contracts, verified non-instantiable rather than reported as implemented functions.",
            "Real graph PPO gradients and learned-policy evaluation are covered; neural WM/PM and real CUDA are not. Extension fixtures remain non-learning.",
        ],
    }
    success = True
    for stage in stages:
        suite = unittest.TestSuite()
        for pattern in STAGES[stage]:
            discovered = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern)
            if discovered.countTestCases() == 0:
                raise RuntimeError("No tests discovered for " + pattern)
            suite.addTests(discovered)
        print("Stage: " + stage, flush=True)
        before = set(called)
        started = time.perf_counter()
        previous_profile = sys.getprofile()
        sys.setprofile(profile)
        try:
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            sys.setprofile(previous_profile)
        missing = sorted(label for key, label in functions.items() if key not in called)
        stage_report = {
            "stage": stage, "tests": result.testsRun,
            "failures": [{"test": str(test), "traceback": detail} for test, detail in result.failures],
            "errors": [{"test": str(test), "traceback": detail} for test, detail in result.errors],
            "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "newly_called_functions": sorted(functions[key] for key in called - before),
            "cumulative_called_functions": len(called),
            "concrete_function_count": len(functions),
            "cumulative_uncalled_functions": missing,
        }
        report["stages"].append(stage_report)
        # When function tests are selected, every concrete function must run first.
        success = result.wasSuccessful() and (stage != "functions" or not missing)
        if not success:
            if missing and stage == "functions":
                print("Uncalled concrete functions: " + ", ".join(missing), file=sys.stderr)
            break
    report.update({
        "success": success, "tests": sum(stage["tests"] for stage in report["stages"]),
        "called_functions": sorted(functions[key] for key in called),
        "uncalled_functions": sorted(label for key, label in functions.items() if key not in called),
        "abstract_interfaces": sorted(interfaces),
    })
    destination = directory / "test_report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Test report: " + str(destination), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
