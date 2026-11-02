"""Supervised pretraining of the original-TG WM, separate from PPO."""
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
from src.buoy_network.world_model import GraphWorldModel, batch, load_split, evaluate
from src.buoy_network.dataset import audit_run, run_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--skip-test", action="store_true", help="Develop/select on validation without another test look")
    parser.add_argument("--output-root", default="results/tg_wm_pretrain")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden, args.history_length, args.lr) <= 0 or args.overfit_samples < 0:
        parser.error("Training settings must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    for path in run_roots(args.dataset):
        audit_run(path)
    training = load_split(args.dataset, "train", args.history_length,
                          limit=args.overfit_samples or None)
    validation = training if args.overfit_samples else load_split(args.dataset, "validation", args.history_length)
    model = GraphWorldModel(args.hidden)
    model.fit_normalization(training)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = Path(args.output_root).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    initial = evaluate(model, validation, args.device)
    best = float("inf")
    started = time.perf_counter()
    with (output/"learning_curve.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for indices in torch.randperm(len(training["length"])).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(batch(training, indices, args.device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite WM loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite WM gradient")
                optimizer.step()
                losses.append(loss.item())
            metrics = evaluate(model, validation, args.device)
            # Validation selection combines calibrated ACK and normalized continuous error.
            score = metrics["ack_brier"] + metrics["queue_mse"] / model.queue_scale.square().mean().item() + metrics["motion_mse"] / model.motion_scale.square().mean().item()
            row = dict(epoch=epoch+1, loss=sum(losses)/len(losses), selection_score=score, **metrics)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if score < best:
                best = score
                torch.save({"architecture": "legacy_tg_graph_gru_wm_v2_ack_history", "hidden": args.hidden,
                            "history_length": args.history_length, "node_count": training["nodes"].shape[2],
                            "state_dict": model.state_dict(), "selected_epoch": epoch+1,
                            "dataset": str(Path(args.dataset).resolve()),
                            "normalization_fit_split": "train", "heads": ["ack", "joint_queue_delta", "own_motion"]}, output/"wm_best.pt")
            if epoch == 0 or (epoch+1) % 10 == 0:
                print("epoch {} loss {:.5f} validation {:.5f}".format(epoch+1, row["loss"], score), flush=True)
    payload = torch.load(output/"wm_best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    checkpoint_hash = hashlib.sha256((output/"wm_best.pt").read_bytes()).hexdigest()
    result = {"mode": "overfit_check" if args.overfit_samples else "supervised_pretrain",
              "initial_validation": initial, "training": evaluate(model, training, args.device),
              "validation": evaluate(model, validation, args.device), "selected_epoch": payload["selected_epoch"],
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter()-started, "checkpoint_sha256": checkpoint_hash}
    if not args.overfit_samples:
        # Test is evaluated once after checkpoint selection; never used in fitting/selection.
        if not args.skip_test:
            result["test"] = evaluate(model, load_split(args.dataset, "test", args.history_length), args.device)
        result["integration_gate"] = {
            "ack": result["validation"]["ack_brier"] < min(result["validation"]["ack_prior_brier"], result["validation"]["ack_history_link_brier"]),
            "queue": result["validation"]["queue_mse"] < result["validation"]["queue_persistence_mse"],
            "motion": result["validation"]["motion_mse"] < result["validation"]["motion_persistence_mse"]}
    result["weights_unchanged_during_evaluation"] = hashlib.sha256((output/"wm_best.pt").read_bytes()).hexdigest() == checkpoint_hash
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Output: " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
