"""Scenario meaning, conservation, isolation and crossed-seed reporting checks."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT)]
from buoy_network.scenario_evaluation import BASELINE,PROFILES,validate,apply_environment,generate_packets,scenario_context


class ScenarioFunctions(unittest.TestCase):
    def test_registered_stresses_change_one_factor_only(self):
        for name,settings in PROFILES.items():
            validate(settings)
            self.assertEqual(sum(settings[k]!=BASELINE[k] for k in BASELINE),0 if name=='baseline' else 1)
        for bad in [dict(BASELINE,packets_per_source=0),dict(BASELINE,packets_per_source=True),
                    dict(BASELINE,loss_scale=float('nan')),dict(BASELINE,walk_step=0),dict(BASELINE,unknown=1)]:
            with self.assertRaises(ValueError):validate(bad)

    def environment(self):
        from test_legacy_targeted_fixes import make_environment
        return make_environment()

    def test_loss_scaling_persists_across_reset_without_changing_observation_prior(self):
        env=self.environment();node=env.buoys[0]
        schedules=[list(n.lost_rate_list) for n in env.buoys];prior=[list(n.success_rate_neighbor) for n in env.buoys]
        apply_environment(env,PROFILES['loss_high']);env.reset();env.buoy_net.reset_lost_rate()
        for i,n in enumerate(env.buoys):
            expected=[min(1.,v*1.5) for v in schedules[i]]
            self.assertEqual(n.lost_rate_list,expected)
            self.assertEqual(n.lost_rate,expected[1]);self.assertEqual(n.success_rate_neighbor,prior[i])

    def test_identity_settings_are_exact(self):
        env=self.environment()
        before=[(list(n.lost_rate_list),n.lost_rate,n.max_speed,list(n.position)) for n in env.buoys]
        apply_environment(env,PROFILES['baseline'])
        self.assertEqual(before,[(list(n.lost_rate_list),n.lost_rate,n.max_speed,list(n.position)) for n in env.buoys])

    def test_double_load_real_generator_unique_ids_fifo_and_conservation(self):
        env=self.environment()
        row=generate_packets(env,type(env).packet_generation,PROFILES['load_double'])
        self.assertEqual(row['generated'],48);self.assertEqual(sum(n.packet for n in env.buoys),48)
        self.assertEqual([p.id for p in env.packets],list(range(48)))
        for node in env.buoys:
            self.assertEqual(node.packet,0 if node.if_sink else 2)
            self.assertEqual([p.queue_order for p in env.packets if p.agent_id==node.id],[] if node.if_sink else [node.id,node.id+24])

    def test_walk_step_changes_actual_motion_and_keeps_boundary(self):
        env=self.environment();node=env.buoys[12];node.sink_neighbor=[]
        for name,expected in [('walk_slow',1.95),('walk_fast',2.1)]:
            node.position_init=[2.,2.];node.position=[1.9,2.]
            apply_environment(env,PROFILES[name])
            with patch('buoy_network.buoy.random.choice',side_effect=[1,0]):node.move()
            self.assertAlmostEqual(node.position[0],expected)
            self.assertTrue(1.9-1e-8<=node.position[0]<=2.1+1e-8)

    def test_patches_restore_after_error_and_reject_training(self):
        from buoy_network.environment import buoy_environment
        method=buoy_environment.packet_generation
        with self.assertRaises(RuntimeError):
            with scenario_context(buoy_environment,BASELINE):raise RuntimeError('test')
        self.assertIs(method,buoy_environment.packet_generation)
        with self.assertRaises(ValueError):
            with scenario_context(buoy_environment,BASELINE):self.environment()
        self.assertIs(method,buoy_environment.packet_generation)


class Reporting(unittest.TestCase):
    def test_crossed_scene_repeats_do_not_inflate_training_seed_count(self):
        from run_thesis_generalization import analyze,GROUPS,ROBUST,SCENES
        temp=ROOT/'results/thesis_test_temp';temp.mkdir(parents=True,exist_ok=True)
        rows=[];models=[]
        for condition in PROFILES:
            for group in GROUPS if condition=='baseline' else ROBUST:
                for seed,rate in [(11,.25),(22,.5),(33,.75)]:
                    if condition=='baseline':models.append(dict(algorithm=group,seed=seed,development_delivery=.5))
                    for scene in SCENES:
                        delivered=0 if group=='VDN' else int(48*rate)
                        rows.append(dict(algorithm=group,seed=seed,condition=condition,scene=scene,generated=48,delivered=delivered,
                            delivery=delivered/48,unresolved_fraction=1-delivered/48,success_hops=3 if delivered else None,
                            success_delay=10 if delivered else None,loop_transfers=0,ack_failures=0,wall_seconds=1,
                            optimizer_updates=0,weights_unchanged=True,physical_sha256='fixture',checkpoint_sha256='fixture',identity_replay_exact=None))
        self.assertEqual(len(rows),324)
        with tempfile.TemporaryDirectory(dir=str(temp)) as folder:
            root=Path(folder);(root/'protocol.json').write_text(json.dumps(dict(scope='Synthetic test fixture')))
            analyze(root,dict(rows=rows,models=models))
            report=json.loads((root/'report.json').read_text());r=next(r for r in report['aggregate'] if r['algorithm']=='TG' and r['condition']=='baseline')
            self.assertAlmostEqual(r['mean'],.5);self.assertAlmostEqual(r['sample_sd'],.25);self.assertEqual(r['training_policies'],3)
            self.assertFalse(report['all_thesis_experiments_completed']);self.assertEqual(report['formal_tasks'],324)
            import csv
            with (root/'per_model.csv').open() as f:
                vdn=[r for r in csv.DictReader(f) if r['algorithm']=='VDN']
            self.assertTrue(all(r['success_delay']=='' and r['success_hops']=='' for r in vdn))
            for name in ['unseen_comparison','robustness']:
                for ext in ['png','svg','pdf']:self.assertGreater((root/(name+'.'+ext)).stat().st_size,1000)


if __name__=='__main__':unittest.main()
