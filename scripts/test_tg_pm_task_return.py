"""Persist function -> module -> PPO checks, then original-TG regressions."""
import io
import json
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
import test_pm_task_return as tests


def main():
    root=ROOT/'results/tg_pm_task_return_tests'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True,exist_ok=False)
    report={'python':sys.executable,'success':True,'stages':[]}
    for name,case in (('functions',tests.ReturnFunctions),('modules',tests.ReturnModules),('integration',tests.ReturnIntegration)):
        stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        (root/(name+'.log')).write_text(stream.getvalue(),encoding='utf-8')
        report['stages'].append({'name':name,'tests':result.testsRun,'failures':len(result.failures),
                                'errors':len(result.errors),'skipped':len(result.skipped)})
        print('{}: {} tests, {} failures, {} errors'.format(name,result.testsRun,len(result.failures),len(result.errors)),flush=True)
        if not result.wasSuccessful():report['success']=False;break
    if report['success']:
        for name,pattern in (('legacy','test_legacy*.py'),('value','test_pm_value.py'),('credit','test_pm_completion_credit.py')):
            command=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v']
            with (root/(name+'.log')).open('xb') as stream:
                p=subprocess.run(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            report['stages'].append({'name':name,'command':command,'exit_code':p.returncode})
            print('{} exit {}'.format(name,p.returncode),flush=True)
            if p.returncode!=0:report['success']=False;break
    (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REPORT '+str(root/'report.json'),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':sys.exit(main())
