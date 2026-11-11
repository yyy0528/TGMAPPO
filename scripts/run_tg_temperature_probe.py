"""Paired known-seed TG/WM temperature probe; never auto-start long training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from buoy_network.ablation_suite import snapshot, suite_lock
from buoy_network.task_reward_audit import audit_reward_csv
from tg_convergence import read, atomic, digest, setarg
from run_tg_wm_input_probe import verify_raw_compatibility


def invoke(command, logfile):
    with Path(logfile).open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=5400)
    if result.returncode:
        raise RuntimeError('Native command failed: '+str(logfile))


def audit_wm_updates(report, episodes, evaluation):
    if evaluation:
        assert report['optimizer_updates'] == 0
        assert not report.get('wm_ppo_updates', [])
    else:
        rows = report['wm_ppo_updates']
        assert len(rows) == episodes
        assert all(r['wm_frozen'] and r['ratio_max_error'] <= 1e-4 for r in rows)


def audit(output, job, calibration_sha, evaluation=False, checkpoint=None):
    import math
    output = Path(output)
    report = read(output/'logs/TG_policy_report.json')
    cfg = read(output/'logs/TG_MAPPO_environment.json')['config']
    wm = job['group'] == 'WM'
    episodes = 2 if evaluation else job['episodes']
    assert report['resources']['actor_trainable'] == (24165 if wm else 21956)
    assert report['resources']['critic_trainable'] == 18615
    assert report['resources']['device'] == 'cuda' and report['pm'] is None
    assert report['policy_temperature'] == job['temperature'] == cfg['tg_policy_temperature']
    assert cfg['ppo_epoch'] == 1 and cfg['num_mini_batch'] == 2
    assert cfg['episode_length'] == 80 and cfg['num_moving_buoys'] == 5
    assert cfg['tg_reward_version'] == 'task_v2'
    assert report['profile']['collect_calls'] == episodes*80
    if wm:
        assert report['wm']['checkpoint_sha256'] == job['wm_sha256'] and report['wm']['frozen']
        assert report['wm_actor_input'] == job['wm_input']
        if job['wm_input'] != 'raw':
            assert report['wm_actor_calibration']['sha256'] == calibration_sha
        audit_wm_updates(report, episodes, evaluation)
    else:
        assert report['wm'] is None
    if evaluation:
        assert report['actor_unchanged'] and report['critic_unchanged'] and report['optimizer_updates'] == 0
        assert report['checkpoint_sha256'] == digest(checkpoint)
    else:
        updates = report['training_updates']
        assert len(updates) == episodes
        for i, u in enumerate(updates):
            assert u['actor_optimizer_steps'] == u['critic_head_optimizer_steps'] == 2*(i+1)
            assert u['actor_lr'] == .0005 and u['critic_lr'] == .0003
            assert all(math.isfinite(v) for v in u.values() if isinstance(v, (int, float)))
    audit_reward_csv(output/'logs', report['reward_contract'])
    physical = [json.loads(x) for x in (output/'logs/TG_physical_metrics.jsonl').read_text().splitlines()]
    assert len(physical) == episodes
    return dict(delivery=sum(x['delivered_packets'] for x in physical)/sum(x['generated_packets'] for x in physical),
                loops=sum(x['loop_transfers'] for x in physical),
                last10_training_delivery=statistics.mean(x['delivery_rate'] for x in physical[-10:]),
                report_sha256=digest(output/'logs/TG_policy_report.json'))


def prepare(reuse_from=None):
    reference = ROOT/'results/tg_wm_input_probe/20261109_122413_853474'
    short = ROOT/'results/tg_wm_input_probe/20261109_115159_931796'
    baseline = ROOT/'results/tg_five_gpu300/20261108_140256_956180'
    assert all(read(p/'manifest.json')['status'] == 'completed' for p in (reference, short, baseline))
    for old in (ROOT/'results/tg_temperature_probe').glob('*/manifest.json'):
        assert read(old)['status'] != 'running', 'Temperature probe already running: '+str(old.parent)
    root = ROOT/'results/tg_temperature_probe'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (root/'frozen').mkdir(parents=True)
    for name in ('wm.pt', 'calibration.json'):
        (root/'frozen'/name).write_bytes((reference/'frozen'/name).read_bytes())
    m = dict(schema='tg_temperature_development_v1', status='prepared', root=str(root), source=snapshot(root), scenes=[8001,8002,8003],
             training_resume_supported=False, automatic_long_training=False, seeds=[11,22,33],
             wm_sha256=digest(root/'frozen/wm.pt'), calibration_sha256=digest(root/'frozen/calibration.json'),
             reference_short=str(short), reference_baseline=str(baseline), jobs=[])
    cases = [('TG',22,2,1.),('WM',22,2,1.)]
    cases += [(g,s,20,.5) for s in (11,22,33) for g in ('TG','WM')]
    for group,seed,episodes,temperature in cases:
        compat = episodes == 2
        tag = ('compat_' if compat else '')+group+'_'+str(seed)
        work = root/tag
        work.mkdir()
        args = [r'D:\anaconda3\envs\buoy_gpu\python.exe','-s','-u',str(root/'source_snapshot/scripts/tg_eval_worker.py'),
                '--algorithm','tg-mappo','--tg_device','cuda','--tg_reward_version','task_v2',
                '--tg_policy_temperature',str(temperature),'--num_episodes',str(episodes),
                '--num_env_steps',str(episodes*80),'--episode_length','80','--ppo_epoch','1',
                '--num_mini_batch','2','--num_moving_buoys','5','--seed',str(seed),
                '--scenario_seed',str(seed),'--output_root',str(work/'train'),'--run-name','case']
        wm_input = 'raw' if compat else 'residual_scaled'
        if group == 'WM':
            args += ['--use_wm_actor','true','--wm_actor_checkpoint',str(root/'frozen/wm.pt'), '--wm_actor_input',wm_input]
            if not compat:args += ['--wm_actor_calibration',str(root/'frozen/calibration.json')]
        job = dict(tag=tag,group=group,seed=seed,episodes=episodes,temperature=temperature,
                   wm_input=wm_input,wm_sha256=m['wm_sha256'],work=str(work),command=args,state='pending')
        if compat:
            job['reference'] = str(ROOT/'results/tg_five_gpu_probe/20261108_135245_396245/jobs'/
                                   (group+'_22')/'training/runs/case/moves_5')
            assert Path(job['reference']).is_dir()
        m['jobs'].append(job)
    if reuse_from:
        old_root = Path(reuse_from).resolve()
        previous = read(old_root/'manifest.json')
        assert previous['status'] == 'failed', 'Only stopped failed probes can be recovered'
        assert previous['scenes']==m['scenes'] and previous['seeds']==m['seeds']
        def core_hashes(source):
            return {k:v for k,v in source['files_sha256'].items() if Path(k).parts[0]=='src'
                    or Path(k).name in ('run_algorithm.py','tg_eval_worker.py')}
        assert core_hashes(previous['source']) == core_hashes(m['source']), 'Training code changed; cannot reuse'
        assert previous['wm_sha256']==m['wm_sha256'] and previous['calibration_sha256']==m['calibration_sha256']
        m['recovery_from'] = dict(root=str(old_root),manifest_sha256=digest(old_root/'manifest.json'))
        for job in m['jobs']:
            old_job = next(j for j in previous['jobs'] if j['tag']==job['tag'])
            assert all(job[k]==old_job[k] for k in ('seed','group','episodes','temperature','wm_input'))
            old_work = Path(old_job['work'])
            old_output = Path(old_job.get('reuse_training',{}).get('output',str(old_work/'train/runs/case/moves_5')))
            checkpoint = old_output/'checkpoints/tg_policy_latest.pt'
            if (old_output/'logs/TG_policy_report.json').is_file() and checkpoint.is_file():
                audit(old_output,job,m['calibration_sha256'])
                job['reuse_training'] = dict(output=str(old_output),report_sha256=digest(old_output/'logs/TG_policy_report.json'),checkpoint_sha256=digest(checkpoint))
                job['reuse_evaluations'] = {}
                for scene in m['scenes']:
                    ev = old_work/('eval_'+str(scene))/'runs/case/moves_5'
                    if not (ev/'logs/TG_policy_report.json').exists() and str(scene) in old_job.get('reuse_evaluations',{}):
                        ev = Path(old_job['reuse_evaluations'][str(scene)]['output'])
                    if (ev/'logs/TG_policy_report.json').exists():
                        audit(ev,job,m['calibration_sha256'],True,checkpoint)
                        job['reuse_evaluations'][str(scene)] = dict(output=str(ev),report_sha256=digest(ev/'logs/TG_policy_report.json'))
    atomic(root/'manifest.json',m)
    print('OUTPUT '+str(root),flush=True)


def run(root):
    root = Path(root).resolve()
    with suite_lock(root):
        m = read(root/'manifest.json')
        assert m['status'] == 'prepared'
        assert all(digest(root/'source_snapshot'/k) == v for k,v in m['source']['files_sha256'].items())
        assert digest(root/'frozen/wm.pt') == m['wm_sha256']
        assert digest(root/'frozen/calibration.json') == m['calibration_sha256']
        m['status'] = 'running';atomic(root/'manifest.json',m)
        try:
            for job in m['jobs']:
                job.update(state='running',started_at=datetime.now().isoformat());atomic(root/'manifest.json',m)
                print('START '+job['tag'],flush=True)
                work = Path(job['work'])
                if 'reuse_training' in job:
                    output = Path(job['reuse_training']['output'])
                    assert digest(output/'logs/TG_policy_report.json') == job['reuse_training']['report_sha256']
                    assert digest(output/'checkpoints/tg_policy_latest.pt') == job['reuse_training']['checkpoint_sha256']
                    print('REUSE_TRAIN '+job['tag'],flush=True)
                else:
                    invoke(job['command'],work/'train.log')
                    output = work/'train/runs/case/moves_5'
                job['training'] = audit(output,job,m['calibration_sha256'])
                if 'reference' in job:
                    verify_raw_compatibility(output,job['reference']);job['legacy_exact'] = True
                else:
                    job['evaluations'] = []
                    cp = output/'checkpoints/tg_policy_latest.pt'
                    for scene in m['scenes']:
                        command = list(job['command'])
                        for k,v in [('--num_episodes',2),('--num_env_steps',160),('--seed',scene),('--scenario_seed',scene),
                                    ('--output_root',work/('eval_'+str(scene))),('--tg_evaluate','true'),('--tg_policy_checkpoint',cp)]:setarg(command,k,v)
                        reused = job.get('reuse_evaluations',{}).get(str(scene))
                        if reused:
                            ev = Path(reused['output'])
                            assert digest(ev/'logs/TG_policy_report.json') == reused['report_sha256']
                        else:
                            invoke(command,work/('eval_'+str(scene)+'.log'))
                            ev = work/('eval_'+str(scene))/'runs/case/moves_5'
                        job['evaluations'].append(dict(scene=scene,**audit(ev,job,m['calibration_sha256'],True,cp)))
                    job['delivery'] = statistics.mean(x['delivery'] for x in job['evaluations'])
                job.update(state='completed',finished_at=datetime.now().isoformat());atomic(root/'manifest.json',m)
                print('DONE '+job['tag']+' delivery '+str(job.get('delivery')),flush=True)
            old_tg = read(Path(m['reference_baseline'])/'report.json')
            old_wm = read(Path(m['reference_short'])/'report.json')
            rows = []
            for seed in (11,22,33):
                tg,wm = [next(j for j in m['jobs'] if j['group']==g and j['seed']==seed and j['episodes']==20) for g in ('TG','WM')]
                warm_tg = next(p['delivery'] for x in old_tg['models'] if x['group']=='TG' and x['seed']==seed for p in x['points'] if p['episode']==20)
                warm_wm = next(r['delivery'] for r in old_wm['rows'] if r['seed']==seed)
                rows.append(dict(seed=seed,tg_cold=tg['delivery'],wm_cold=wm['delivery'],tg_warm=warm_tg,wm_warm=warm_wm,
                                 wm_train_eval_gap=wm['training']['last10_training_delivery']-wm['delivery']))
            report = dict(status='completed',engineering_passed=True,episodes=20,rows=rows,
                          mean_tg_cold=statistics.mean(r['tg_cold'] for r in rows),mean_wm_cold=statistics.mean(r['wm_cold'] for r in rows),
                          scope='Known-development-seed pilot; temperature is a shared PPO change, WM benefit requires paired TG control; no300 or unseen-seed claim')
            report['wm_above_paired_tg'] = report['mean_wm_cold'] > report['mean_tg_cold']
            report['wm_no_seed_regression_vs_warm'] = all(r['wm_cold']>=r['wm_warm'] for r in rows)
            report['wm_no_seed_regression_vs_paired_tg'] = all(r['wm_cold']>=r['tg_cold'] for r in rows)
            render(root,rows)
            atomic(root/'report.json',report)
            m.update(status='completed',report_sha256=digest(root/'report.json'));atomic(root/'manifest.json',m)
            print(json.dumps(report),flush=True)
        except Exception as exc:
            m.update(status='failed',error=repr(exc));atomic(root/'manifest.json',m);raise


def render(root, rows):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ['tg_warm','wm_warm','tg_cold','wm_cold']
    fig,ax = plt.subplots(figsize=(8,5),constrained_layout=True)
    means = [100*statistics.mean(r[k] for r in rows) for k in keys]
    sd = [100*statistics.stdev(r[k] for r in rows) for k in keys]
    ax.bar(range(4),means,yerr=sd,capsize=4,color=['#286f9a','#2c8b60','#286f9a','#2c8b60'])
    ax.set_xticks(range(4));ax.set_xticklabels(['TG T=1','WM T=1','TG T=0.5','WM T=0.5'])
    ax.set_ylabel('Greedy frozen delivery (%)');ax.set_ylim(0,110)
    ax.set_title('20 episodes, 3 known training seeds; mean +/- sample SD')
    ax.grid(axis='y',alpha=.2)
    for suffix in ('png','svg','pdf'):fig.savefig(str(root/('comparison.'+suffix)),dpi=160)
    plt.close(fig)
    with (root/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run')
    parser.add_argument('--reuse-from')
    args = parser.parse_args()
    if args.prepare:prepare(args.reuse_from)
    elif args.run:run(args.run)
    else:parser.error('Use --prepare or --run ROOT')
