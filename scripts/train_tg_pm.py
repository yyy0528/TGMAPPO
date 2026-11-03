"""Pretrain original-TG censored-time PM; only train/validation, no test peeking."""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.buoy_network.progress_model import ARCHITECTURE, FEATURES, ProgressModel, load_packets, pack, survival_terms, evaluate
from src.buoy_network.dataset import audit_run, run_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output-root", default="results/tg_pm_pretrain")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden, args.lr) <= 0:
        parser.error("Positive training settings required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    audits = [audit_run(path) for split in ("train", "validation") for path in run_roots(args.dataset, split)]
    training_packets, validation_packets = (load_packets(args.dataset, split) for split in ("train", "validation"))
    training, validation = pack(training_packets), pack(validation_packets)
    if training["horizon"] != validation["horizon"] or not training["events"].any() or training["events"].all():
        parser.error("Require a shared horizon and both delivery/censored training observations")
    output = Path(args.output_root).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print("PM output: "+str(output), flush=True)
    model = ProgressModel(training["horizon"], args.hidden)
    model.fit(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial = evaluate(model, validation)
    best, selected_state, selected_metrics, selected_epoch = None, None, None, None
    def selection(metrics):
        ratios = (metrics["survival_nll"]/max(metrics["prior_survival_nll"], 1e-8),
                  metrics["window_brier"]/max(metrics["prior_window_brier"], 1e-8))
        return (sum(r >= 1 for r in ratios), max(ratios), sum(ratios))
    with (output/"learning_curve.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            # Uniform packet sampling keeps original outcome prevalence and equal packet weight.
            for indices in torch.randperm(len(training_packets)).split(args.batch_size):
                optimizer.zero_grad()
                lengths, events = training["lengths"][indices], training["events"][indices]
                loss = survival_terms(model(training["features"][indices], lengths), lengths, events).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PM loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite PM gradient")
                optimizer.step()
                losses.append(loss.item())
            metrics = evaluate(model, validation)
            row = dict(epoch=epoch+1, training_loss=sum(losses)/len(losses), **metrics)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            key = selection(metrics)
            if best is None or key < best:
                best, selected_metrics, selected_epoch = key, metrics, epoch+1
                selected_state = {name:value.detach().clone() for name,value in model.state_dict().items()}
            if epoch == 0 or (epoch+1) % 5 == 0:
                print("epoch {} nll {:.4f}/{:.4f} brier {:.4f}/{:.4f}".format(epoch+1, metrics["survival_nll"],
                    metrics["prior_survival_nll"], metrics["window_brier"], metrics["prior_window_brier"]), flush=True)
    model.load_state_dict(selected_state)
    passed = best[0] == 0
    payload = {"architecture": ARCHITECTURE, "features": list(FEATURES), "horizon": model.horizon,
               "hidden": model.hidden, "node_count": 25, "state_dict": selected_state,
               "normalization_fit_split": "train", "validation_gate_passed": passed,
               "validation": selected_metrics, "selected_epoch": selected_epoch,
               "target": "right_censored_delivery_time; not eventual success/failure"}
    with (output/"pm_best.pt").open("xb") as stream:
        torch.save(payload, stream)
    report = {"config": vars(args), "initial_validation": initial, "selected_validation": selected_metrics,
              "training": evaluate(model, training), "selected_epoch": selected_epoch, "gate_passed": passed,
              "parameter_count": sum(p.numel() for p in model.parameters()), "seconds": time.perf_counter()-started,
              "checkpoint_sha256": hashlib.sha256((output/"pm_best.pt").read_bytes()).hexdigest(), "audits": audits,
              "train_scenario_seeds": sorted({p["scenario_seed"] for p in training_packets}),
              "validation_scenario_seeds": sorted({p["scenario_seed"] for p in validation_packets}),
              "test_evaluated": False, "normalization_fit_split": "train",
              "limitations": ["no true failure events", "no reliability heads", "CPU only", "no routing benefit proven"]}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("PM validation gate: "+("PASS" if passed else "FAIL (cannot enable shaping)"), flush=True)


if __name__ == "__main__":
    main()
