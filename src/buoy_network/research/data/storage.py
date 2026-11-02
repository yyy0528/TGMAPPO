"""Streaming, versioned JSONL storage; tensor encoding belongs to trainers."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from ..contracts import CANDIDATE_FEATURES, NODE_FEATURES, SCHEMA_VERSION, PacketTrace, TransitionRecord
from .buffers import EpisodeReplay


def episode_split(episode_ids: list[str], seed: int, train_fraction: float,
                  validation_fraction: float) -> dict[str, list[str]]:
    if (any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1
            for value in (train_fraction, validation_fraction))
            or train_fraction + validation_fraction >= 1):
        raise ValueError("Split fractions must be positive and leave a test fraction")
    ids = sorted(episode_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate episode IDs")
    random.Random(seed).shuffle(ids)
    if len(ids) < 3:
        return {"train": ids, "validation": [], "test": []}
    validation_count = min(max(1, int(len(ids) * validation_fraction)), len(ids) - 2)
    train_count = min(max(1, int(len(ids) * train_fraction)), len(ids) - validation_count - 1)
    return {"train": ids[:train_count],
            "validation": ids[train_count:train_count + validation_count],
            "test": ids[train_count + validation_count:]}


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _episode_path(self, episode_id: str) -> Path:
        if not isinstance(episode_id, str) or not re.fullmatch(r"[\w-]+", episode_id):
            raise ValueError("Unsafe episode ID")
        return self.root / "episodes" / episode_id

    def write_episode(self, records: tuple[TransitionRecord, ...],
                      traces: tuple[PacketTrace, ...]) -> None:
        if not records:
            raise ValueError("Cannot archive an empty episode")
        episode_id = records[0].episode_id
        if (any(record.episode_id != episode_id for record in records)
                or any(trace.key.episode_id != episode_id for trace in traces)):
            raise ValueError("Mixed episodes in archive")
        # Validate before creating files: malformed records must not poison a dataset.
        replay = EpisodeReplay(len(records))
        replay.add_episode(records)
        final = records[-1].state_tp1
        if len(traces) != final.generated or len({trace.key for trace in traces}) != len(traces):
            raise ValueError("Packet traces must account for every generated packet once")
        if any(trace.outcome is None or trace.end_step is None
               or not trace.generated_step <= trace.end_step <= final.env_step for trace in traces):
            raise ValueError("A complete finite episode needs resolved packet traces")
        if (sum(trace.outcome is True for trace in traces) != final.delivered
                or sum(trace.outcome is False for trace in traces) != final.dropped):
            raise ValueError("Packet trace outcomes disagree with final state counters")
        for row in (*records, *traces):
            json.dumps(asdict(row), allow_nan=False)
        directory = self._episode_path(episode_id)
        directory.mkdir(parents=True, exist_ok=False)
        for name, rows in (("transitions.jsonl", records), ("packets.jsonl", traces)):
            with (directory / name).open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(asdict(row), allow_nan=False) + "\n")

    def write_manifest(self, splits: dict[str, list[str]], env_version: str, *,
                       environment_backend: str | None = None) -> None:
        manifest = {"schema_version": SCHEMA_VERSION, "env_version": env_version,
                    "splits": splits, "format": "episode-jsonl"}
        if environment_backend is not None:
            manifest["environment_backend"] = environment_backend
        self._validate_manifest(manifest)
        count = sum(len(ids) for ids in splits.values())
        manifest["split_note"] = (
            "Fewer than 3 episodes cannot form all three splits" if count < 3
            else "Split by complete episode and scenario seed"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "data_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)

    def _validate_manifest(self, manifest: dict) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("Dataset manifest must be an object")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Incompatible dataset schema")
        if not isinstance(manifest.get("env_version"), str) or not manifest["env_version"]:
            raise ValueError("Dataset manifest requires an environment version")
        if manifest.get("format") != "episode-jsonl":
            raise ValueError("Unsupported dataset format")
        if "environment_backend" in manifest and (
                not isinstance(manifest["environment_backend"], str) or not manifest["environment_backend"]):
            raise ValueError("Invalid dataset environment backend")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
            raise ValueError("Dataset requires train, validation and test split lists")
        if any(not isinstance(ids, list) for ids in splits.values()):
            raise ValueError("Dataset split entries must be lists of episode IDs")
        ids = [episode_id for split in splits.values() for episode_id in split]
        for episode_id in ids:
            directory = self._episode_path(episode_id)
            if any(not (directory / name).is_file() or (directory / name).stat().st_size == 0
                   for name in ("transitions.jsonl", "packets.jsonl")):
                raise ValueError("Incomplete dataset episode: " + episode_id)
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset splits overlap")
        # Derive schema/seed metadata from archived data, also for older manifests.
        num_nodes = None
        episode_seeds = {}
        seed_splits = {}
        for split, episode_ids in splits.items():
            for episode_id in episode_ids:
                path = self._episode_path(episode_id) / "transitions.jsonl"
                with path.open(encoding="utf-8") as stream:
                    first = json.loads(stream.readline())
                try:
                    agents = first["obs_t"]["agents"]
                    count = len(agents)
                    seed = first["seed"]
                    valid = (count > 0 and first["episode_id"] == episode_id
                             and first["env_version"] == manifest["env_version"]
                             and type(first["schema_version"]) is int
                             and first["schema_version"] == SCHEMA_VERSION
                             and type(seed) is int and seed >= 0
                             and len(first["state_t"]["nodes"]) == count
                             and all(len(agent["node_features"]) == count
                                     and all(len(row) == len(NODE_FEATURES) for row in agent["node_features"])
                                     and len(agent["available_actions"]) == count + 1 for agent in agents))
                except (KeyError, TypeError):
                    raise ValueError("Malformed episode header: " + episode_id) from None
                if not valid:
                    raise ValueError("Incompatible episode header: " + episode_id)
                if num_nodes is not None and count != num_nodes:
                    raise ValueError("Mixed node counts in one dataset")
                num_nodes = count
                episode_seeds[episode_id] = seed
                if seed in seed_splits and seed_splits[seed] != split:
                    raise ValueError("Scenario seed appears in different dataset splits: " + str(seed))
                seed_splits[seed] = split
        inferred = {
            "num_nodes": num_nodes, "observation_schema": "local_graph_v1",
            "action_schema": "global_node_wait_v1", "node_features": list(NODE_FEATURES),
            "candidate_features": list(CANDIDATE_FEATURES), "episode_seeds": episode_seeds,
        }
        for name, value in inferred.items():
            if name in manifest and manifest[name] != value:
                raise ValueError("Dataset metadata disagrees with archived data: " + name)
            manifest[name] = value

    def manifest(self) -> dict:
        with (self.root / "data_manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        self._validate_manifest(manifest)
        return manifest

    def iter_rows(self, split: str, packets: bool = False) -> Iterator[dict]:
        manifest = self.manifest()
        if split not in manifest["splits"]:
            raise ValueError("Unknown dataset split: " + split)
        filename = "packets.jsonl" if packets else "transitions.jsonl"
        for episode_id in manifest["splits"][split]:
            with (self._episode_path(episode_id) / filename).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    try:
                        key = row["key"]["episode_id"] if packets else row["episode_id"]
                        valid_schema = packets or (type(row["schema_version"]) is int
                                                   and row["schema_version"] == SCHEMA_VERSION)
                    except (KeyError, TypeError):
                        raise ValueError("Malformed {}:{}".format(filename, line_number)) from None
                    if key != episode_id or not valid_schema:
                        raise ValueError("Dataset episode/schema mismatch in {}:{}".format(
                            filename, line_number))
                    yield row
