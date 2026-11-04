import json
import contextlib
import io
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'src',ROOT/'tests'):
    sys.path.insert(0,str(path))
from scripts.interaction_budget_worker import equal_states,capture_prefix,install_capture
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import run


def fake():
    return SimpleNamespace(policy=SimpleNamespace(actor=torch.nn.Linear(2,2),critic=torch.nn.Linear(2,1)),
        profile={'ppo_updates':1,'collect_calls':4},training_records=[{'update':1,'actor_lr':.0005}])


class BudgetFunctions(unittest.TestCase):
    def test_state_comparison_checks_names_and_values(self):
        a={'x':torch.ones(2)}
        self.assertTrue(equal_states(a,{'x':torch.ones(2)}))
        self.assertFalse(equal_states(a,{'y':torch.ones(2)}))
        self.assertFalse(equal_states(a,{'x':torch.zeros(2)}))

    def test_capture_preserves_weights_rng_and_record_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            model=fake()
            rng=torch.get_rng_state().clone()
            before={n:{k:v.clone() for k,v in getattr(model.policy,n).state_dict().items()} for n in ('actor','critic')}
            report=capture_prefix(model,temp)
            model.training_records.append({'update':2})
            self.assertEqual(len(report['training_updates']),1)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for n in before:
                self.assertTrue(equal_states(before[n],getattr(model.policy,n).state_dict()))
            self.assertTrue(Path(report['checkpoint']).is_file())

    def test_mismatched_reference_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            model=fake()
            reference=root/'wrong.pt'
            torch.save({'actor':{},'critic':{}},str(reference))
            with self.assertRaisesRegex(ValueError,'Prefix'):
                capture_prefix(model,root/'bad',reference)
            self.assertFalse(list((root/'bad').glob('*.pt')))
            capture_prefix(model,root/'good')
            with self.assertRaises(FileExistsError):
                capture_prefix(model,root/'good')


class BudgetModules(unittest.TestCase):
    def test_config_is_fixed100_and_model_variants_disabled(self):
        config=json.loads((ROOT/'configs/tg_interaction_fixed100.json').read_text())
        self.assertEqual(config['groups'],['A'])
        self.assertEqual(config['train_seeds'],[11,22,33])
        self.assertEqual((config['train_episodes'],config['episode_length']),(100,80))
        self.assertEqual((config['ppo_epoch'],config['num_mini_batch']),(1,2))
        self.assertEqual((config['learning_rate_schedule'],config['critic_time_mode']),('fixed','disabled'))

    def test_resume_plan_only_reads_budget(self):
        from scripts import run_tg_interaction_budget as driver
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'manifest.json').write_text(json.dumps({'budget':{'training_jobs':3}}))
            with patch.object(sys,'argv',['run','--resume',temp]),patch.object(driver,'run_suite') as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.main()
            execute.assert_not_called()


class BudgetIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_read_only_capture_does_not_change_actual_ppo(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs=[]
            for enabled in (False,True):
                random.seed(33)
                np.random.seed(33)
                torch.manual_seed(33)
                env=make_environment(Path(temp)/str(enabled),length=4)
                env.all_args.num_episodes=2
                env.all_args.use_linear_lr_decay=False
                model=make_learner(env)
                if enabled:
                    context,observed=install_capture(type(model),1,Path(temp)/'prefix')
                    with context:
                        run(env,model)
                    self.assertEqual(len(observed),1)
                    self.assertEqual(len(observed[0]['training_updates']),1)
                else:
                    run(env,model)
                outputs.append({n:{k:v.clone() for k,v in getattr(model.policy,n).state_dict().items()} for n in ('actor','critic')})
            self.assertTrue(all(equal_states(outputs[0][n],outputs[1][n]) for n in outputs[0]))

    def test_public_entry_adapter_matches_original_training_and_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            common=['--algorithm','tg-mappo','--episode_length','4','--ppo_epoch','1','--num_mini_batch','2',
                '--num_moving_buoys','0','--seed','33','--scenario_seed','33','--tg_reward_version','task_v2']
            commands=[
                [sys.executable,str(ROOT/'run_algorithm.py')]+common+['--num_episodes','1','--num_env_steps','4','--output_root',str(root/'ref'),'--run-name','case'],
                [sys.executable,str(ROOT/'run_algorithm.py')]+common+['--num_episodes','2','--num_env_steps','8','--output_root',str(root/'native'),'--run-name','case'],
                [sys.executable,str(ROOT/'scripts/interaction_budget_worker.py'),'--snapshot',str(ROOT),'--capture-update','1',
                    '--prefix-reference',str(root/'ref/runs/case/moves_0/checkpoints/tg_policy_latest.pt')]+common+
                    ['--num_episodes','2','--num_env_steps','8','--output_root',str(root/'adapter'),'--run-name','case']]
            for command in commands:
                result=subprocess.run(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout)
            a,b=[torch.load(str(root/k/'runs/case/moves_0/checkpoints/tg_policy_latest.pt')) for k in ('native','adapter')]
            self.assertTrue(all(equal_states(a[n],b[n]) for n in ('actor','critic')))
            self.assertEqual((root/'native/runs/case/moves_0/logs/TG_physical_metrics.jsonl').read_bytes(),
                             (root/'adapter/runs/case/moves_0/logs/TG_physical_metrics.jsonl').read_bytes())


if __name__=='__main__':
    unittest.main(verbosity=2)
