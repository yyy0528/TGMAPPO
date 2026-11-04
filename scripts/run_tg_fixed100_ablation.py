"""Prepare original TG A-F fixed100 comparisons, reusing the completed A baseline.

plan/prepare never fit a model. preflight fits only 2 episodes in each variant.
run is separate and requires a successful, unchanged preflight receipt.
"""
import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.buoy_network.ablation_suite import (
    atomic, budget, create_suite, digest, inspect_job, jobs, read, require,
    run_suite, suite_lock, validate,
)

DEFAULT_BASELINE = ROOT/'results/tg_interaction_fixed100/20261104_132048_759710'
DEFAULT_SHORT = ROOT/'results/tg_ablation_lr_fixed_pilot/20261103_211528_024882'
WORKFLOW = 'original_tg_fixed100_reuse_v1'


def fixed100_config(config):
    config = validate(config)
    expected = dict(purpose='pilot', groups=list('ABCDEF'), train_seeds=[11,22,33],
                    eval_seeds=[7001], train_episodes=100, eval_episodes=2,
                    episode_length=80, ppo_epoch=1, num_mini_batch=2,
                    num_moving_buoys=5, random_channel=True,
                    learning_rate_schedule='fixed', critic_time_mode='disabled',
                    target_window=20, target_delivery_rate=.8)
    require(all(config[k] == v for k,v in expected.items()), 'Wrong common fixed100 protocol')
    return config


def compatible_protocol(left, right, exclude=()):
    for field in set(left) | set(right):
        if field not in set(exclude) | {'purpose', 'groups', 'job_timeout_seconds'}:
            a = left.get(field, 'disabled' if field == 'critic_time_mode' else None)
            b = right.get(field, 'disabled' if field == 'critic_time_mode' else None)
            require(a == b, 'Incompatible protocol field: '+field)


def runtime_hashes(root):
    root = Path(root)
    paths = [root/'run_algorithm.py'] + sorted((root/'src').rglob('*.py'))
    return {p.relative_to(root).as_posix(): digest(p) for p in paths}


def check_source(root, manifest):
    for name, checksum in manifest['source']['files_sha256'].items():
        require(digest(Path(root)/'source_snapshot'/name) == checksum,
                'Archived source changed: '+name)


def check_training(report, episodes):
    require(report['profile']['ppo_updates'] == episodes and
            report['profile']['collect_calls'] == episodes*80, 'Wrong actual interaction/update count')
    rows = report['training_updates']
    require(len(rows) == episodes, 'Missing actual PPO records')
    for index, row in enumerate(rows, 1):
        require(row['actor_optimizer_steps'] == row['critic_head_optimizer_steps'] == index*2,
                'Wrong actual Adam budget')
        require(row['actor_lr'] == .0005 and row['critic_lr'] == .0003, 'Fixed learning rate changed')
        require(all(math.isfinite(v) for v in row.values() if type(v) in (int,float)),
                'Nonfinite PPO statistic')
    for name in ('wm','pm'):
        auxiliary = report[name]
        updates = report[name+'_ppo_updates']
        require(len(updates) == (episodes if auxiliary else 0), 'Missing auxiliary PPO checks')
        if auxiliary:
            require(all(u[name+'_frozen'] for u in updates), 'Auxiliary weights updated')
    if report['wm']:
        require(all(u['ratio_max_error'] <= 1e-6 for u in report['wm_ppo_updates']),
                'WM probability replay mismatch')
        require(report['wm']['learned_forward_used'] == (report['wm']['feature_source'] == 'learned'),
                'History control called learned WM')
    if report['pm']:
        require(all(u['gae_uses_shaped_rewards'] for u in report['pm_ppo_updates']),
                'PM reward did not reach GAE')
        require(len(report['pm']['episodes']) == episodes and
                all(abs(e['boundary_error']) < 1e-6 and e['terminal_phi'] == 0
                    for e in report['pm']['episodes']), 'PM boundary identity failed')


def audit_baseline(root, config):
    root = Path(root).resolve()
    manifest = read(root/'manifest.json')
    old = validate(manifest['config'])
    compatible_protocol(config, old)
    require(manifest['state'] == 'completed' and old['groups'] == ['A'] and
            len(manifest['jobs']) == 6, 'Need the complete three-seed fixed100 A baseline')
    check_source(root, manifest)
    runtime = runtime_hashes(root/'source_snapshot')
    require(runtime == runtime_hashes(ROOT), 'Runtime differs from reused baseline')
    for name, path in config['auxiliary_reports'].items():
        require(digest(path) == digest(root/('auxiliary_'+name+'.json')),
                'Auxiliary/data provenance changed: '+name)
    seen = set()
    artifacts = {}
    for job in manifest['jobs']:
        require(job['state'] == 'completed' and job['key'] not in seen, 'Incomplete/duplicate baseline job')
        require(job['attempts'] and job['attempts'][-1]['exit_code'] == 0, 'Baseline attempt failed')
        seen.add(job['key'])
        report, _ = inspect_job(job, config)
        require(report['wm'] is None and report['pm'] is None, 'Baseline contains an auxiliary')
        if job['kind'] == 'training':
            check_training(report, 100)
        artifacts[job['key']] = {
            'output': job['output'], 'policy_sha256': report['checkpoint_sha256'],
            'files_sha256': dict(job['artifacts_sha256']),
            'reward_sha256': digest(Path(job['output'])/'logs/TG_reward_components.csv'),
        }
    expected = {j['key'] for j in jobs(dict(config, groups=['A']), root)}
    require(seen == expected, 'Baseline seed/scene coverage differs')
    return {'root': str(root), 'manifest_sha256': digest(root/'manifest.json'),
            'dependencies': read(root/'dependencies.json'), 'runtime_sha256': runtime,
            'artifacts': artifacts}, manifest


def execution_budget(config):
    total = budget(config)
    return {'logical': total,
            'reused': {'training_jobs': 3, 'evaluation_jobs': 3,
                       'training_steps': 24000, 'evaluation_steps': 480},
            'new_fixed100': {'training_jobs': 15, 'evaluation_jobs': 15,
                             'training_steps': 120000, 'evaluation_steps': 2400,
                             'adam_steps_per_network_per_policy': 200},
            'short_compatibility': {'training_jobs': 18, 'evaluation_jobs': 18,
                                    'training_steps': 2880, 'evaluation_steps': 2880},
            'timing_note': 'B-F fixed100 CPU wall time is unmeasured; step counts are not speed claims'}


def import_baseline(planned, previous, reference):
    by_key = {j['key']: j for j in previous}
    result = copy.deepcopy(planned)
    for job in result:
        if job['group'] != 'A':
            continue
        old = by_key[job['key']]
        require(old['state'] == 'completed' and old['kind'] == job['kind'], 'Cannot import unfinished A')
        for key in ('output','attempts','artifacts_sha256','expected_policy_sha256'):
            if key in old:
                job[key] = copy.deepcopy(old[key])
        job['state'] = 'completed'
        job['reused_from'] = {'root': reference['root'], 'manifest_sha256': reference['manifest_sha256'],
                              'job_key': old['key']}
    return result


def short_config(config):
    return dict(config, train_episodes=2, job_timeout_seconds=600)


def check_executor(manifest):
    config = manifest['config']
    require(Path(sys.executable).resolve() == Path(config['python']).resolve(),
            'Run with the configured Python')
    code = ("import json,sys,platform,torch,pkg_resources; "
            "print(json.dumps(dict(python=sys.version, executable=sys.executable, platform=platform.platform(), "
            "torch=torch.__version__, cuda_available=torch.cuda.is_available(), "
            "versions={n:pkg_resources.get_distribution(n).version "
            "for n in ['numpy','scipy','gym','torch-geometric']})))")
    actual = json.loads(subprocess.check_output([config['python'],'-c',code], universal_newlines=True))
    require(actual == read(Path(manifest['workflow']['baseline_root'])/'dependencies.json'),
            'Installed dependencies changed since baseline')


def prepare(config, baseline, short_reference, output_root):
    config = fixed100_config(config)
    reference, previous = audit_baseline(baseline, config)
    short_reference = Path(short_reference).resolve()
    old_short = read(short_reference/'manifest.json')
    compatible_protocol(short_config(config), old_short['config'])
    require(old_short['state'] == 'completed', 'Historical short suite is incomplete')
    require(Path(sys.executable).resolve() == Path(config['python']).resolve(), 'Prepare with the configured Python')
    root = create_suite(config, output_root)
    require(read(root/'dependencies.json') == reference['dependencies'], 'Dependencies differ from baseline')
    preflight = create_suite(short_config(config), root/'compatibility')
    require(read(preflight/'dependencies.json') == reference['dependencies'], 'Short dependencies differ')
    manifest = read(root/'manifest.json')
    manifest['jobs'] = import_baseline(manifest['jobs'], previous['jobs'], reference)
    manifest['state'] = 'awaiting_preflight'
    manifest['workflow'] = {
        'kind': WORKFLOW, 'baseline_root': reference['root'], 'preflight_root': str(preflight),
        'short_reference': str(short_reference), 'short_reference_manifest_sha256': digest(short_reference/'manifest.json'),
        'execution_budget': execution_budget(config), 'preflight_receipt_sha256': None,
    }
    atomic(root/'baseline_reference.json', reference)
    (root/'baseline_manifest.json').write_bytes((Path(baseline)/'manifest.json').read_bytes())
    atomic(root/'manifest.json', manifest)
    return root


def guard(root):
    root = Path(root).resolve()
    manifest = read(root/'manifest.json')
    require(manifest['workflow']['kind'] == WORKFLOW, 'Wrong workflow')
    config = fixed100_config(manifest['config'])
    check_source(root, manifest)
    require(runtime_hashes(ROOT) == runtime_hashes(root/'source_snapshot'), 'Current runtime changed')
    require(digest(__file__) == digest(root/'source_snapshot/scripts/run_tg_fixed100_ablation.py'),
            'Use the unchanged prepared workflow tool')
    reference, baseline = audit_baseline(manifest['workflow']['baseline_root'], config)
    require(reference == read(root/'baseline_reference.json'), 'Baseline reference changed')
    require(digest(root/'baseline_manifest.json') == reference['manifest_sha256'], 'Copied baseline manifest changed')
    require(read(root/'dependencies.json') == reference['dependencies'], 'Prepared dependencies changed')
    planned = jobs(config, root)
    expected = import_baseline(planned, baseline['jobs'], reference)
    require([j['key'] for j in manifest['jobs']] == [j['key'] for j in expected], 'Job coverage/order changed')
    for job, original in zip(manifest['jobs'], expected):
        require(job['base_command'] == original['base_command'], 'Prepared command changed')
        if job['group'] == 'A':
            require(job == original, 'Imported A changed; refusing to retrain baseline')
    short = Path(manifest['workflow']['short_reference'])
    require(digest(short/'manifest.json') == manifest['workflow']['short_reference_manifest_sha256'],
            'Historical short reference changed')
    preflight = Path(manifest['workflow']['preflight_root'])
    pm = read(preflight/'manifest.json')
    require(pm['config'] == short_config(config), 'Preflight protocol changed')
    check_source(preflight, pm)
    require(pm['source']['files_sha256'] == manifest['source']['files_sha256'], 'Preflight source differs')
    require(read(preflight/'dependencies.json') == reference['dependencies'], 'Preflight dependencies differ')
    expected_short = jobs(short_config(config), preflight)
    require([j['key'] for j in pm['jobs']] == [j['key'] for j in expected_short] and
            all(a['base_command'] == b['base_command'] for a,b in zip(pm['jobs'], expected_short)),
            'Short job coverage/command changed')
    for name,path in config['auxiliary_reports'].items():
        require(digest(path) == digest(root/('auxiliary_'+name+'.json')) ==
                digest(preflight/('auxiliary_'+name+'.json')), 'Copied auxiliary provenance changed')
    return manifest


def preflight_receipt(root, manifest):
    # Local trusted checkpoints only. Importing Torch here never occurs in plan/prepare.
    import torch
    preflight = Path(manifest['workflow']['preflight_root'])
    fresh = read(preflight/'manifest.json')
    old_root = Path(manifest['workflow']['short_reference'])
    old = read(old_root/'manifest.json')
    require(fresh['state'] == 'completed' and len(fresh['jobs']) == 36, 'Short suite did not complete')
    old_by_key = {j['key']:j for j in old['jobs']}
    rows = []
    for job in fresh['jobs']:
        require(job['state'] == 'completed', 'Incomplete short job')
        report, physical = inspect_job(job, fresh['config'])
        previous = old_by_key[job['key']]
        require(previous['state'] == 'completed', 'Historical short job incomplete')
        old_report, _ = inspect_job(previous, old['config'])
        hashes = {}
        for filename in ('TG_physical_metrics.jsonl', 'TG_reward_components.csv'):
            checksum = digest(Path(job['output'])/'logs'/filename)
            require(checksum == digest(Path(previous['output'])/'logs'/filename), 'Short replay changed: '+job['key']+'/'+filename)
            hashes[filename] = checksum
        if job['kind'] == 'training':
            check_training(report, 2)
            kwargs = {'map_location':'cpu'}
            if int(torch.__version__.split('.')[0]) >= 2:
                kwargs['weights_only'] = True
            a, b = [torch.load(r['checkpoint'], **kwargs) for r in (report, old_report)]
            require(all(set(a[n]) == set(b[n]) and all(torch.equal(a[n][k], b[n][k]) for k in a[n])
                        for n in ('actor','critic')), 'Short policy changed: '+job['key'])
        rows.append({'key':job['key'], 'kind':job['kind'], 'policy_sha256':report['checkpoint_sha256'],
                     'files_sha256':hashes, 'delivered':sum(r['delivered_packets'] for r in physical),
                     'generated':sum(r['generated_packets'] for r in physical),
                     'historical_model_or_frozen_parent_exact':True})
    return {'schema':'original_tg_fixed100_preflight_v1', 'success':True, 'suite':str(root),
            'preflight_root':str(preflight), 'historical_root':str(old_root),
            'completed_jobs':36, 'training_models_exact':18, 'physical_and_rewards_exact':36,
            'source_sha256':fresh['source']['files_sha256'], 'dependencies':read(preflight/'dependencies.json'),
            'jobs':rows, 'limitations':['2 episodes only: compatibility, not WM/PM efficacy',
                                       'B-F fixed100 not executed by preflight']}


def run_preflight(root):
    root = Path(root).resolve()
    # Same writer lock as run_suite, so preflight and medium training cannot race.
    with suite_lock(root):
        manifest = guard(root)
        check_executor(manifest)
        require(not any(j['group'] != 'A' and j['attempts'] for j in manifest['jobs']),
                'Fixed100 already started; do not repeat compatibility training')
        run_suite(manifest['workflow']['preflight_root'])
        receipt = preflight_receipt(root, manifest)
        atomic(root/'preflight_receipt.json', receipt)
        manifest['workflow']['preflight_receipt_sha256'] = digest(root/'preflight_receipt.json')
        manifest['state'] = 'ready_for_fixed100'
        atomic(root/'manifest.json', manifest)
        return receipt


def run_fixed100(root):
    root = Path(root).resolve()
    manifest = guard(root)
    check_executor(manifest)
    receipt = root/'preflight_receipt.json'
    require(receipt.is_file() and digest(receipt) == manifest['workflow']['preflight_receipt_sha256'],
            'Successful unchanged preflight receipt required before fixed100')
    require(read(receipt)['success'] is True, 'Preflight failed')
    preflight = read(Path(manifest['workflow']['preflight_root'])/'manifest.json')
    require(preflight['state'] == 'completed', 'Preflight no longer complete')
    for job in preflight['jobs']:
        inspect_job(job, preflight['config'])
    result = run_suite(root)
    # Preserve the original summary format; execution accounting has its own receipt.
    atomic(root/'execution_accounting.json', manifest['workflow']['execution_budget'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--config')
    selection.add_argument('--resume')
    parser.add_argument('--mode', choices=('plan','prepare','preflight','run'), default='plan')
    parser.add_argument('--baseline', default=str(DEFAULT_BASELINE))
    parser.add_argument('--short-reference', default=str(DEFAULT_SHORT))
    parser.add_argument('--output-root', default='results/tg_ablation_fixed100')
    args = parser.parse_args()
    if args.config:
        require(args.mode in ('plan','prepare'), 'Create with plan/prepare, then use --resume')
        config = fixed100_config(read(args.config))
        if args.mode == 'plan':
            reference, _ = audit_baseline(args.baseline, config)
            print(json.dumps({'budget':execution_budget(config), 'baseline':reference}, indent=2))
        else:
            print('Prepared: '+str(prepare(config,args.baseline,args.short_reference,args.output_root)), flush=True)
    else:
        root = Path(args.resume).resolve()
        require(args.mode != 'prepare', 'Prepare requires --config')
        if args.mode == 'plan':
            print(json.dumps(guard(root)['workflow']['execution_budget'], indent=2))
        elif args.mode == 'preflight':
            print(json.dumps(run_preflight(root), indent=2))
        else:
            run_fixed100(root)


if __name__ == '__main__':
    main()
