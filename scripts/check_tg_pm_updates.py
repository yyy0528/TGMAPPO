"""Prefix compatibility checks, then three exact seed22 training replays."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from diagnose_tg_pm_updates import read, sha, require

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(str(path))


def main():
    diagnosis = ROOT/'results/tg_seed22_signals/20261105_165019_896744'
    require(read(diagnosis/'acceptance.json')['diagnosis_complete'], 'Preceding diagnosis incomplete')
    root = ROOT/'results/tg_pm_update_replay'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    tools = [ROOT/'scripts'/k for k in ('diagnose_tg_pm_updates.py', 'check_tg_pm_updates.py', 'diagnose_tg_credit.py')]
    m = {'status': 'running', 'root': str(root), 'diagnosis': str(diagnosis),
         'tool_sha256': {str(p.relative_to(ROOT)): sha(p) for p in tools}, 'jobs': [],
         'budget': {'prefix_steps': 480, 'exact_repeat_training_steps': 4800,
                    'total_new_steps': 5280, 'new_independent_efficacy_seeds': 0}}
    save(root/'manifest.json', m)
    print('OUTPUT '+str(root), flush=True)
    try:
        for phase in ('prefix', 'full'):
            for group in ('A', 'task_return', 'alpha01'):
                tag = phase+'_'+group
                command = [sys.executable, str(ROOT/'scripts/diagnose_tg_pm_updates.py'),
                    '--diagnosis', str(diagnosis), '--group', group, '--output', str(root/tag)]
                if phase == 'prefix':
                    command += ['--episodes', '2']
                job = {'tag': tag, 'phase': phase, 'group': group, 'command': command,
                       'output': str(root/tag), 'state': 'running', 'started': datetime.now().isoformat()}
                m['jobs'].append(job)
                save(root/'manifest.json', m)
                print('START '+tag, flush=True)
                with (root/(tag+'.log')).open('xb') as log:
                    result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                job['exit_code'] = result.returncode
                require(result.returncode == 0, 'Exact training replay failed: '+tag)
                report = read(root/tag/'report.json')
                require(report['success'] and report['exact_training_updates'], 'Replay audit failed')
                if phase == 'full':
                    require(report['exact_final_actor_critic'] is True, 'Final parameters differ')
                job.update(state='completed', finished=datetime.now().isoformat(),
                           report_sha256=sha(root/tag/'report.json'))
                save(root/'manifest.json', m)
                print('DONE '+tag, flush=True)
        require(all(sha(ROOT/k) == v for k, v in m['tool_sha256'].items()), 'Observer sources changed during replay')
        m['status'] = 'completed'
    except Exception as exc:
        m['status'], m['error'] = 'failed', repr(exc)
        if m['jobs'] and m['jobs'][-1]['state'] == 'running':
            m['jobs'][-1]['state'] = 'failed'
        save(root/'manifest.json', m)
        raise
    save(root/'manifest.json', m)
    print('PASS '+str(root/'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
