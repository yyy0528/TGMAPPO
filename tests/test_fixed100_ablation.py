"""Guard real baseline reuse and keep compatibility distinct from long training."""
import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'tests')]
from scripts import run_tg_fixed100_ablation as driver
from test_ablation_suite import configuration


def config(root):
    value = configuration(root)
    value.update(schema='original_tg_ablation_v3', train_episodes=100,
                 learning_rate_schedule='fixed', critic_time_mode='disabled', target_window=20,
                 reward_contract=dict(version='task_v2', delivery_weight=1,delay_weight=1,
                                      unfinished_weight=1,normalization='initial_packets_and_task_horizon_v1'))
    return value


class Fixed100Functions(unittest.TestCase):
    def test_protocol_rejects_unequal_budgets_and_information(self):
        with tempfile.TemporaryDirectory() as temp:
            value = config(Path(temp))
            driver.fixed100_config(value)
            for change in (dict(train_episodes=20),dict(ppo_epoch=4),dict(critic_time_mode='remaining'),
                           dict(learning_rate_schedule='linear'),dict(groups=list('ABCD')),
                           dict(eval_seeds=[7002]),dict(random_channel=False)):
                with self.assertRaises(ValueError):
                    driver.fixed100_config(dict(value,**change))

    def test_incremental_budget_does_not_charge_reused_training(self):
        with tempfile.TemporaryDirectory() as temp:
            result = driver.execution_budget(driver.fixed100_config(config(Path(temp))))
            self.assertEqual(result['logical']['total_training_steps'],144000)
            self.assertEqual(result['new_fixed100']['training_steps'],120000)
            self.assertEqual(result['reused']['training_steps'],24000)
            self.assertEqual(result['short_compatibility']['training_steps'],2880)
            self.assertEqual(result['new_fixed100']['evaluation_steps']+result['reused']['evaluation_steps'],2880)

    def test_runtime_comparison_detects_changes_additions_and_removal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'src').mkdir()
            (root/'run_algorithm.py').write_text('entry')
            path = root/'src/a.py'
            path.write_text('original')
            old = driver.runtime_hashes(root)
            path.write_text('changed')
            self.assertNotEqual(old,driver.runtime_hashes(root))
            path.write_text('original')
            added = root/'src/b.py'
            added.write_text('extra')
            self.assertNotEqual(old,driver.runtime_hashes(root))
            added.unlink()
            path.unlink()
            self.assertNotEqual(old,driver.runtime_hashes(root))

    def test_import_copies_actual_attempts_without_mutating_old_jobs(self):
        planned = [dict(key='A_seed_11',group='A',kind='training',state='pending',attempts=[],base_command=['new']),
                   dict(key='B_seed_11',group='B',kind='training',state='pending',attempts=[])]
        previous = [dict(key='A_seed_11',kind='training',state='completed',attempts=[{'command':['actual']}],
                         output='old/output',artifacts_sha256={'log':'sha'})]
        before = copy.deepcopy(previous)
        result = driver.import_baseline(planned,previous,dict(root='old',manifest_sha256='sha'))
        self.assertEqual(result[0]['attempts'][0]['command'],['actual'])
        self.assertEqual(result[0]['base_command'],['new'])
        self.assertEqual(result[1]['state'],'pending')
        result[0]['attempts'][0]['command'].append('changed')
        self.assertEqual(previous,before)
        previous[0]['state'] = 'failed'
        with self.assertRaises(ValueError):
            driver.import_baseline(planned,previous,dict(root='old',manifest_sha256='sha'))


class Fixed100Modules(unittest.TestCase):
    def test_plan_is_read_only_and_never_probes_or_starts_subprocesses(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            path.write_text(json.dumps(config(Path(temp))))
            with patch.object(sys,'argv',['run','--config',str(path)]), \
                 patch.object(driver,'audit_baseline',return_value=({},{})), \
                 patch.object(driver,'create_suite') as create, \
                 patch.object(driver.subprocess,'Popen',side_effect=AssertionError('plan launched a process')), \
                 contextlib.redirect_stdout(io.StringIO()):
                driver.main()
            create.assert_not_called()

    def test_medium_training_requires_receipt_and_never_falls_back_to_new_a(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = {'workflow':{'preflight_receipt_sha256':None}}
            with patch.object(driver,'guard',return_value=manifest), \
                 patch.object(driver,'check_executor'),patch.object(driver,'run_suite') as run:
                with self.assertRaisesRegex(ValueError,'preflight receipt'):
                    driver.run_fixed100(temp)
            run.assert_not_called()

    def test_actual_update_validation_rejects_budget_nan_and_auxiliary_changes(self):
        row = dict(actor_optimizer_steps=2,critic_head_optimizer_steps=2,actor_lr=.0005,critic_lr=.0003,
                   policy_loss=.1,approx_kl=.001)
        report = dict(profile=dict(ppo_updates=1,collect_calls=80), training_updates=[row],
                      wm=None,pm=None,wm_ppo_updates=[],pm_ppo_updates=[])
        driver.check_training(report,1)
        for change in (dict(actor_optimizer_steps=4),dict(critic_lr=.00015),dict(policy_loss=float('nan'))):
            with self.assertRaises(ValueError):
                driver.check_training(dict(report,training_updates=[dict(row,**change)]),1)
        with self.assertRaises(ValueError):
            driver.check_training(dict(report,wm={'frozen':True},wm_ppo_updates=[{'wm_frozen':False}]),1)


class Fixed100Integration(unittest.TestCase):
    def test_real_prepare_imports_only_a_and_rejects_tampering_or_early_run(self):
        # Real archived three-seed weights/logs + two real dependency probes; no fitting.
        with tempfile.TemporaryDirectory(dir=str(ROOT/'results')) as temp:
            root = driver.prepare(driver.read(ROOT/'configs/tg_ablation_fixed100.json'),
                                  driver.DEFAULT_BASELINE,driver.DEFAULT_SHORT,temp)
            manifest = driver.guard(root)
            self.assertEqual(len(manifest['jobs']),36)
            self.assertEqual(sum(j['state']=='completed' for j in manifest['jobs']),6)
            with patch.object(driver,'run_suite') as run:
                with self.assertRaisesRegex(ValueError,'preflight receipt'):
                    driver.run_fixed100(root)
                run.assert_not_called()
            original = copy.deepcopy(manifest)
            manifest['jobs'][0]['state'] = 'pending'
            driver.atomic(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Imported A changed'):
                driver.guard(root)
            driver.atomic(root/'manifest.json',original)
            pm_root = Path(original['workflow']['preflight_root'])
            pm = driver.read(pm_root/'manifest.json')
            pm['jobs'][0]['base_command'] += ['--num_episodes','100']
            driver.atomic(pm_root/'manifest.json',pm)
            with self.assertRaisesRegex(ValueError,'Short job coverage/command'):
                driver.guard(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
