"""Audit read-only diagnostic evidence and save a compact receipt."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",required=True)
    parser.add_argument("--reward-probes",required=True)
    parser.add_argument("--report",required=True)
    args = parser.parse_args()
    root = Path(args.batch).resolve()
    batch = read(root/"summary.json")
    suite = Path(batch["suite"])
    manifest = read(suite/"manifest.json")
    require(batch["cases"] == len(batch["reports"]) == 24 and batch["all_weights_unchanged"],"Missing diagnostic cases")
    commands = read(root/"commands.json")
    require(len(commands) == 24 and all(row["exit_code"] == 0 for row in commands),"Worker failure")
    for name,checksum in manifest["source"]["files_sha256"].items():
        require(digest(suite/"source_snapshot"/name) == checksum,"Pilot snapshot changed")
        require(not (ROOT/name).is_file() or digest(ROOT/name) == checksum,"Diagnostic used changed core")
    selected,actions_by_seed,probabilities = [],{},{}
    artifacts = {}
    for result in batch["reports"]:
        case = Path(result["output"])
        require(result["optimizer_updates"] == 0 and result["weights_unchanged"],"Diagnosis updated weights")
        require(result["eval_scene"] == 7001,"Not the original development scene")
        frames = [json.loads(line) for line in (case/"slots.jsonl").read_text().splitlines()]
        require(len(frames) == 160,"Missing traced slots")
        for frame in frames:
            for node,action in enumerate(frame["actions"]):
                p,mask = frame["probabilities"][node],frame["mask"][node]
                require(mask[action] and all(math.isfinite(v) for v in p) and abs(sum(p)-1) < 1e-5,"Invalid policy distribution/action")
                require(all(v == 0 for v,legal in zip(p,mask) if not legal),"Illegal action probability")
                if result["decode"] == "greedy":
                    require(action == max(range(26),key=lambda i:p[i]),"Greedy action differs from captured distribution")
        signature = hashlib.sha256(json.dumps([f["actions"] for f in frames],separators=(",",":")).encode()).hexdigest()
        if result["stage"] == "trained" and result["decode"] == "greedy":
            require(result["pilot_metrics_exact_match"],"Pilot metrics did not reproduce")
            actions_by_seed.setdefault(str(result["training_seed"]),{})[result["group"]] = signature
            probabilities[(result["group"],result["training_seed"])] = [f["probabilities"] for f in frames]
        analysis = {key:value for key,value in result["analysis"].items() if key not in ("action_counts_by_sender","episodes")}
        if result["group"] == "A" and result["decode"] == "greedy" and result["stage"] == "trained":
            analysis["representative_routes"] = [p for episode in result["analysis"]["episodes"] for p in episode["unresolved_examples"]]
        selected.append({key:value for key,value in result.items() if key not in ("analysis","output")})
        selected[-1].update(analysis=analysis,action_trace_sha256=signature,output=str(case))
        artifacts[str(case.relative_to(root))] = {name:digest(case/name) for name in ("report.json","slots.jsonl","packet_routes.jsonl")}
    require(all(len(values) == 6 and len(set(values.values())) == 1 for values in actions_by_seed.values()),
            "Do not claim equal actions: traces differ")
    deltas = []
    for group in "BCDEF":
        for seed in (11,22,33):
            a,b = probabilities[("A",seed)],probabilities[(group,seed)]
            difference = max(abs(x-y) for fa,fb in zip(a,b) for ra,rb in zip(fa,fb) for x,y in zip(ra,rb))
            deltas.append({"group":group,"seed":seed,"max_probability_difference_from_A":difference})
    probes = read(Path(args.reward_probes)/"summary.json")["reports"]
    wait = next(row for row in probes if row["control"] == "wait")
    geographic = next(row for row in probes if row["control"] == "geographic")
    require(all(row["weights_unchanged"] and row["optimizer_updates"] == 0 and not row["actor_executed"] for row in probes),"Reward control trained Actor")
    require(wait["episode_environment_returns"] == [0.,0.] and wait["episode_terminal_rewards"] == [0.,0.],"WAIT reward claim false")
    require(sum(row["delivered_packets"] for row in wait["physical_episodes"]) == 0,"WAIT delivered")
    require(sum(row["delivered_packets"] for row in geographic["physical_episodes"]) == 48,"Geographic control did not deliver 48")
    require(geographic["first_successful_sink_attempt"]["reward"] < 0
            and geographic["first_successful_sink_attempt"]["history_before"] == 0,"Cold sink reward claim false")
    tests = []
    test_commands = ([sys.executable,"-m","unittest","tests.test_policy_diagnostics","-v"],
                     [sys.executable,"-m","unittest","discover","-s","tests","-p","test_legacy_diagnostic_capture.py","-v"])
    for command,n in zip(test_commands,(4,2)):
        completed = subprocess.run(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   universal_newlines=True,timeout=60)
        require(completed.returncode == 0 and "Ran {} tests".format(n) in completed.stdout,"Diagnostic test failure: "+completed.stdout)
        tests.append({"command":command,"tests":n,"exit_code":completed.returncode,"log":completed.stdout})
    sources = ("src/buoy_network/policy_diagnostics.py","scripts/diagnose_tg_policy.py","scripts/probe_tg_reward_contract.py",
               "scripts/plot_tg_policy_diagnosis.py","scripts/verify_tg_policy_diagnosis.py",
               "tests/test_policy_diagnostics.py","tests/test_legacy_diagnostic_capture.py")
    # Keep the as-used core and sidecars runnable even after later reward/policy changes.
    archive = root/"diagnostic_source"
    paths = set(manifest["source"]["files_sha256"]) | set(sources)
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT/"tests").glob("*.py"))
    archive_hashes = {}
    for name in sorted(paths):
        destination = archive/name
        checksum = digest(ROOT/name)
        if destination.exists():
            require(digest(destination) == checksum,"Existing diagnostic archive changed")
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open("xb") as stream:
                stream.write((ROOT/name).read_bytes())
        archive_hashes[name] = checksum
    archive_manifest = root/"diagnostic_source_manifest.json"
    if archive_manifest.exists():
        require(read(archive_manifest) == archive_hashes,"Archive manifest changed")
    else:
        archive_manifest.write_text(json.dumps(archive_hashes,indent=2),encoding="utf-8")
    receipt = {"success":True,"scope":"read-only zero-delivery diagnosis; no training or policy/reward fix",
               "new_tests":6,"tests":tests,"diagnostic_cases":24,"reward_control_cases":2,
               "batch":str(root),"commands_sha256":digest(root/"commands.json"),"artifacts_sha256":artifacts,
               "cases":selected,"equal_greedy_action_traces_by_seed":actions_by_seed,"probability_differences":deltas,
               "reward_probes":probes,"source_files_sha256":{name:digest(ROOT/name) for name in sources},
               "source_archive":str(archive),"source_archive_manifest_sha256":digest(archive_manifest),
               "limitations":["scene 7001 only, two episodes per case", "sampling is a diagnosis control, not a selected evaluation fix",
                              "observed SCCs are not physically closed network regions", "WAIT is not proven globally optimal",
                              "no claim that longer training alone resolves the issue"]}
    with Path(args.report).open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print("PASS: 6 new tests; 24 frozen replays and 2 reward controls; 18 exact pilot reproductions; no updates")
    print("Receipt: "+str(Path(args.report).resolve()))


if __name__ == "__main__":
    main()
