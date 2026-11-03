"""Sequential PopArt acceptance using original TG dependencies."""
import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tests"))
import test_legacy_popart as tests


def main():
    root = ROOT/"results/tg_popart_tests"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    report = {"python":sys.executable,"success":True,"stages":[]}
    for name,case in (("functions",tests.PopArtFunctions),("modules",tests.PopArtModules),("integration",tests.PopArtIntegration)):
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        (root/(name+".log")).write_text(stream.getvalue(),encoding="utf-8")
        report["stages"].append({"name":name,"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors)})
        print("{}: {} tests, {} failures, {} errors".format(name,result.testsRun,len(result.failures),len(result.errors)),flush=True)
        if not result.wasSuccessful():
            print(stream.getvalue())
            report["success"] = False
            break
    (root/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Report: "+str(root/"report.json"))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
