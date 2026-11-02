from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import ExperimentConfig
from ..contracts import CANDIDATE_FEATURES, NODE_FEATURES, SCHEMA_VERSION

METRIC_FIELDS = (
    "episode", "seed", "environment_steps", "delivery_rate", "drop_rate",
    "successful_mean_hops", "successful_mean_delay_slots", "successful_p95_delay_slots",
    "task_return", "earp_return", "shaping_return", "train_return",
)


class RunArtifacts:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def create(cls, config: ExperimentConfig, device: str, command: str) -> RunArtifacts:
        runtime = config.runtime
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        name = runtime.run_name or stamp
        if not re.fullmatch(r"[\w.-]+", name) or name in (".", ".."):
            raise ValueError("run_name must be a single safe directory name")
        root = Path(runtime.output_root).resolve() / "runs" / name
        root.mkdir(parents=True, exist_ok=False)
        result = cls(root)
        result.write_json("config.json", config.to_dict())
        versions = {}
        for package in ("torch", "numpy", "torch-geometric"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        result.write_json("environment_manifest.json", {
            "schema_version": SCHEMA_VERSION, "env_version": config.environment.version,
            "environment_backend": config.environment.backend, "command": command,
            "python": platform.python_version(), "dependencies": versions,
            "requested_device": runtime.device, "resolved_device": device,
            "timezone": "Asia/Shanghai", "created_at": datetime.now(
                timezone(timedelta(hours=8))).isoformat(),
            "node_features": NODE_FEATURES, "candidate_features": CANDIDATE_FEATURES,
            "actions": {"node_indices": config.environment.num_nodes,
                        "wait_index": config.environment.num_nodes},
            "reference_limitations": "No collision/energy/capacity model; IID loss"
            if config.environment.backend == "reference_grid" else None,
        })
        return result

    def write_json(self, relative_path: str, value: dict) -> None:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Artifact path escapes run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)

    def write_metrics(self, rows: list[dict]) -> None:
        with (self.root / "research_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
