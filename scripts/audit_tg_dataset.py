"""Audit legacy TG WM exports without importing Torch, SciPy or Gym."""

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from buoy_network.dataset import audit_run, iter_wm_samples, run_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        reports = [audit_run(path) for path in run_roots(args.dataset)]
        if not reports:
            raise ValueError("No dataset runs found")
        preview = {}
        for split in ("train", "validation", "test"):
            preview[split] = list(itertools.islice(iter_wm_samples(args.dataset, split), 2))
        result = {"passed": True, "runs": reports,
                  "total_steps": sum(row["steps"] for row in reports),
                  "total_packets": sum(row["packets"] for row in reports),
                  "total_delivered": sum(row["delivered"] for row in reports),
                  "total_unresolved": sum(row["unresolved"] for row in reports),
                  "wm_sample_preview": preview}
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({key: value for key, value in result.items() if key != "wm_sample_preview"}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print("Dataset audit failed: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
