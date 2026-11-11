"""Run the selected original-TG WM version from its immutable source/model archive."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected():
    profile = read(ROOT/'configs/tg_wm_active.json')
    study = ROOT/profile['study_root']
    manifest = read(study/'manifest.json')
    assert manifest['status'] == 'completed'
    assert manifest['source']['git_parent'] == profile['source_commit']
    assert sha(study/'report.json') == manifest['report_sha256']
    assert read(study/'report.json')['mean_delivery'] == profile['mean_delivery']
    for name, expected in manifest['source']['files_sha256'].items():
        assert sha(study/'source_snapshot'/name) == expected, name
    assert sha(study/'frozen/wm.pt') == manifest['wm_sha256']
    assert sha(study/'frozen/calibration.json') == manifest['calibration_sha256']
    for job in manifest['jobs'][1:]:
        progress = read(Path(job['work'])/'convergence/progress.json')
        point = progress['points'][-1]
        assert progress['status']=='completed' and point['episode']==300
        assert sha(point['checkpoint']) == point['checkpoint_sha256']
    return profile, manifest


def setarg(command, key, value):
    if key in command:
        command[command.index(key)+1] = str(value)
    else:
        command.extend([key,str(value)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--seed',type=int,choices=(11,22,33),default=11)
    parser.add_argument('--scene',type=int,choices=(8001,8002,8003),default=8001)
    args = parser.parse_args()
    profile,manifest = selected()
    if args.check:
        print(json.dumps(profile,indent=2));return
    job = next(j for j in manifest['jobs'] if j['tag']=='scaled_'+str(args.seed))
    point = read(Path(job['work'])/'convergence/progress.json')['points'][-1]
    output = ROOT/'results/tg_wm_active_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    command = list(job['command'])
    if not Path(command[0]).is_file():
        raise ValueError('Selected GPU Python is missing: '+command[0])
    for key,value in [('--num_episodes',2),('--num_env_steps',160),('--seed',args.scene),
                      ('--scenario_seed',args.scene),('--output_root',output),
                      ('--tg_evaluate','true'),('--tg_policy_checkpoint',point['checkpoint'])]:
        setarg(command,key,value)
    print('OUTPUT '+str(output),flush=True)
    with (output/'evaluation.log').open('wb') as log:
        result = subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,timeout=900)
    if result.returncode:
        raise RuntimeError('Selected evaluation failed; see '+str(output/'evaluation.log'))
    actual = output/'runs/case/moves_5'
    report = read(actual/'logs/TG_policy_report.json')
    assert report['actor_unchanged'] and report['critic_unchanged'] and report['optimizer_updates']==0
    assert report['checkpoint_sha256']==sha(point['checkpoint'])
    reference = Path(job['work'])/'convergence/e0300'/('scene_'+str(args.scene))/'runs/case/moves_5'
    assert (actual/'logs/TG_physical_metrics.jsonl').read_bytes()==(reference/'logs/TG_physical_metrics.jsonl').read_bytes()
    summary = dict(selected_version=profile['selected_version'],seed=args.seed,scene=args.scene,
                   original_physical_replay_exact=True,optimizer_updates=0,source_commit=profile['source_commit'],
                   command=command,output=str(actual))
    (output/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
