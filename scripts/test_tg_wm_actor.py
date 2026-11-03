"""Run step-5 functions, modules, and original TG PPO integration in order."""
import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests"))
import test_legacy_wm_actor as tests


def main():
    destination = ROOT/"results"/"tg_wm_actor_tests"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    report = {"python": sys.executable, "stages": [], "success": True}
    for name, case in (("functions", tests.FrozenInputFunctions), ("modules", tests.ActorCacheModules),
                       ("integration", tests.WMIntegrationTests)):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        (destination/(name+".log")).write_text(stream.getvalue(), encoding="utf-8")
        report["stages"].append({"name": name, "tests": result.testsRun,
                                 "failures": len(result.failures), "errors": len(result.errors)})
        print("{}: {} tests, {} failures, {} errors".format(name, result.testsRun, len(result.failures), len(result.errors)), flush=True)
        if not result.wasSuccessful():
            report["success"] = False
            print(stream.getvalue(), flush=True)
            break
    (destination/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report: " + str(destination/"report.json"))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
