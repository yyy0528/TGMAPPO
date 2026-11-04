"""Read-only acceptance of original TG fixed100 preparation and short compatibility.

The one resume probe below can only resume the 2-episode compatibility suite;
it never calls the medium-budget run mode.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from scripts import run_tg_fixed100_ablation as driver
from src.buoy_network.ablation_suite import read,require,digest,inspect_job


def attempts(manifest):
    return {j['key']:len(j['attempts']) for j in manifest['jobs']}


def verify(root, tests_root):
    root = Path(root).resolve()
    manifest = driver.guard(root)
    driver.check_executor(manifest)
    require(manifest['state'] == 'ready_for_fixed100', 'Preparation is not ready')
    require(sum(j['group']=='A' and j['state']=='completed' for j in manifest['jobs']) == 6,
            'Wrong reused job count')
    pending = [j for j in manifest['jobs'] if j['group'] != 'A']
    require(len(pending) == 30 and all(j['state']=='pending' and not j['attempts'] and
            'output' not in j for j in pending), 'B-F fixed100 already ran or was altered')
    require(not (root/'training').exists() and not (root/'evaluation').exists() and
            not (root/'summary.json').exists(), 'Unexpected medium-budget outputs')
    receipt = read(root/'preflight_receipt.json')
    require(digest(root/'preflight_receipt.json') == manifest['workflow']['preflight_receipt_sha256'],
            'Receipt changed')
    require(receipt == driver.preflight_receipt(root,manifest), 'Receipt differs from raw results')
    preflight = Path(manifest['workflow']['preflight_root'])
    pm = read(preflight/'manifest.json')
    historical_root = Path(manifest['workflow']['short_reference'])
    historical = read(historical_root/'manifest.json')
    driver.check_source(historical_root,historical)
    require(read(historical_root/'dependencies.json') == read(preflight/'dependencies.json'),
            'Historical short runtime dependencies differ')
    for name in ('wm','pm','data'):
        require(digest(historical_root/('auxiliary_'+name+'.json')) ==
                digest(preflight/('auxiliary_'+name+'.json')), 'Historical auxiliary provenance differs')
    before = {'medium_attempts':attempts(manifest), 'short_attempts':attempts(pm),
              'medium_manifest_sha256':digest(root/'manifest.json'),
              'short_summary_sha256':digest(preflight/'summary.json'),
              'receipt_sha256':digest(root/'preflight_receipt.json')}
    baseline = Path(manifest['workflow']['baseline_root'])
    before['baseline_manifest_sha256'] = digest(baseline/'manifest.json')
    before['baseline_summary_sha256'] = digest(baseline/'summary.json')
    probe = subprocess.run([manifest['config']['python'],str(ROOT/'scripts/run_tg_fixed100_ablation.py'),
                            '--resume',str(root),'--mode','preflight'],cwd=str(ROOT),
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
    require(probe.returncode == 0 and '\nSTART ' not in '\n'+probe.stdout,
            'Completed short resume failed or started another task: '+probe.stdout[-3000:])
    require(attempts(read(root/'manifest.json')) == before['medium_attempts'] and
            attempts(read(preflight/'manifest.json')) == before['short_attempts'] and
            digest(root/'manifest.json') == before['medium_manifest_sha256'] and
            digest(preflight/'summary.json') == before['short_summary_sha256'] and
            digest(root/'preflight_receipt.json') == before['receipt_sha256'] and
            digest(baseline/'manifest.json') == before['baseline_manifest_sha256'] and
            digest(baseline/'summary.json') == before['baseline_summary_sha256'],
            'Resume changed completed results, old baseline, or started fixed100')
    tests_root = Path(tests_root).resolve()
    tests = read(tests_root/'report.json')
    require(tests['python'] == manifest['config']['python'] and tests['success'] and
            [(s['name'],s['tests']) for s in tests['stages']] ==
            [('functions',4),('modules',3),('integration',1)] and
            all(s['failures']==s['errors']==s['skipped']==0 for s in tests['stages']),
            'New layered tests were incomplete/failed')
    tests['root'] = str(tests_root)
    tests['source_sha256'] = {name:digest(ROOT/name) for name in
        ('tests/test_fixed100_ablation.py','scripts/test_tg_fixed100_ablation.py')}
    tests['logs'] = {name:(tests_root/(name+'.log')).read_text(encoding='utf-8')
                     for name in ('functions','modules','integration')}
    costs = []
    for group in 'ABCDEF':
        selected = [j for j in pm['jobs'] if j['group']==group and j['kind']=='training']
        reports = [inspect_job(j,pm['config'])[0] for j in selected]
        resources = [r['resources'] for r in reports]
        require(all(r==resources[0] for r in resources), 'Resources differ within variant')
        costs.append({'group':group, 'resources':resources[0],
                      'short_train_wall_seconds_mean':statistics.mean(j['attempts'][-1]['seconds'] for j in selected),
                      'short_collect_seconds_mean':statistics.mean(r['profile']['collect_seconds'] for r in reports),
                      'short_ppo_seconds_mean':statistics.mean(r['profile']['ppo_seconds'] for r in reports),
                      'short_diagnostic_seconds_mean':statistics.mean(r['profile']['diagnostic_seconds'] for r in reports)})
    return {'schema':'original_tg_fixed100_handoff_acceptance_v1','success':True,
            'suite':str(root),'state':manifest['state'],'source':manifest['source'],
            'baseline_reference':read(root/'baseline_reference.json'),
            'historical_short_source':historical['source'],
            'budget':manifest['workflow']['execution_budget'], 'new_layered_tests':tests,
            'preflight':receipt,'resources_and_short_cpu_costs':costs,
            'resume_probe':{'returncode':probe.returncode,'new_short_tasks':0,'new_fixed100_tasks':0,
                            'completed_results_and_old_baseline_unchanged':True,'before_sha256':before,
                            'stdout_sha256':__import__('hashlib').sha256(probe.stdout.encode('utf-8')).hexdigest()},
            'fixed100_pending_jobs':[j['key'] for j in pending],
            'validator_sha256':digest(__file__),
            'limitations':['B-F fixed100 is prepared, not executed; no new innovation efficacy claim',
                           'short two-episode CPU timings are not long-run speed benchmarks',
                           'core 139 tests and prior 7 interaction-tool tests were not rerun',
                           'one frozen WM/PM pretraining seed; incomplete reliability/failure labels',
                           'development scene 7001; no convergence, significance or blind-scene result']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--tests-root',required=True)
    p.add_argument('--report',required=True)
    args = p.parse_args()
    report = verify(args.suite,args.tests_root)
    with Path(args.report).open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print('PASS: 8 tests; 6 A jobs reused; 18 exact short models / 36 physical-reward replays; '
          '0 new resume tasks; 30 B-F fixed100 jobs pending')
    print('Report: '+str(Path(args.report).resolve()))


if __name__ == '__main__':
    main()
