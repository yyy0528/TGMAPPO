"""Ordered function/module tests for the original TG experiment tools."""
import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests"))
import test_legacy_ablation as controls
import test_ablation_suite as suite


def main():
    root = ROOT/"results"/"tg_ablation_tests"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    report = {"success": True, "python": sys.executable, "stages": []}
    for name, cases in (("functions", (controls.AblationFunctions, suite.SuiteFunctions)),
                        ("modules", (controls.AblationModules, suite.SuiteModules))):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in cases))
        (root/(name+".log")).write_text(stream.getvalue(), encoding="utf-8")
        report["stages"].append({"name": name, "tests": result.testsRun, "failures": len(result.failures),
                                 "errors": len(result.errors), "skipped": len(result.skipped)})
        print("{}: {} tests, {} failures, {} errors".format(name, result.testsRun, len(result.failures), len(result.errors)), flush=True)
        if not result.wasSuccessful() or result.skipped:
            report["success"] = False
            print(stream.getvalue())
            break
    (root/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: "+str(root/"report.json"))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
