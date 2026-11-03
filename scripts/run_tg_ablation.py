"""Plan, execute, resume, or summarize ORIGINAL TG experiments (Python 3.6+)."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.buoy_network.ablation_suite import read, validate, budget, create_suite, run_suite, summarize, atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--resume")
    parser.add_argument("--mode", choices=("plan", "run", "summarize"), default="plan")
    parser.add_argument("--output-root", default="results/tg_ablation")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("Provide exactly one of --config or --resume")
    if args.resume:
        root = Path(args.resume).resolve()
        if args.mode == "run":
            run_suite(root)
        elif args.mode == "summarize":
            result = summarize(read(root/"manifest.json"))
            atomic(root/"summary.json", result)
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(read(root/"manifest.json")["budget"], indent=2))
    else:
        config = validate(read(args.config))
        if args.mode == "plan":
            print(json.dumps(budget(config), indent=2))
        elif args.mode == "run":
            root = create_suite(config, args.output_root)
            print("Suite: "+str(root), flush=True)
            run_suite(root)
        else:
            parser.error("Summarize requires --resume")


if __name__ == "__main__":
    main()
