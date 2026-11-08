"""Prepare and run five original-TG variants with a frozen, unified GPU protocol."""
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from tg_convergence import read, atomic, digest, require, checkpoint_grid, setarg
from tg_five import validate, budget, VARIANTS
from buoy_network.ablation_suite import snapshot


def prepare(config_path):
    cfg = validate(read(config_path))
    require(Path(sys.executable).resolve() == Path(cfg['python']).resolve(), 'Prepare with the registered GPU Python')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch, torch_geometric, numpy, site
    require(torch.cuda.is_available() and not site.ENABLE_USER_SITE, 'CUDA unavailable or shared user packages enabled; use python -s')
    reference = cfg.get('probe_reference')
    if cfg['episodes'] == 300:
        require(reference, 'A completed five-group probe is required before full300')
        previous = read(Path(reference)/'manifest.json')
        require(previous['status'] == 'completed' and read(Path(reference)/'report.json')['engineering_passed'], 'Probe incomplete')
        for name,h in previous['source']['files_sha256'].items():
            if name.startswith(('src', 'scripts')):
                require(digest(ROOT/name) == h, 'Source changed since probe: '+name)
    root = ROOT/cfg['output_root']/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    source = snapshot(root)
    frozen = root/'frozen_models'; frozen.mkdir()
    refs = {}
    for name in ('wm','pm'):
        path = Path(cfg[name+'_checkpoint']).resolve()
        refs[str(path)] = digest(path)
        target = frozen/(name+'.pt');shutil.copyfile(str(path),str(target))
        cfg['frozen_'+name+'_sha256'] = digest(target)
        cfg[name+'_checkpoint'] = str(target.resolve())
    if reference:
        for name in ('manifest.json','report.json'):
            path = (Path(reference)/name).resolve();refs[str(path)] = digest(path)
    cfg.update(root=str(root.resolve()), created=datetime.now().isoformat(), checkpoints=checkpoint_grid(cfg['episodes'],cfg['interval']),
               software=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                             pyg=torch_geometric.__version__, numpy=numpy.__version__, gpu=torch.cuda.get_device_name(0)))
    atomic(root/'protocol.json',cfg)
    (root/'dependencies.txt').write_text(subprocess.check_output([sys.executable,'-s','-m','pip','freeze','--all'],text=True),encoding='utf-8')
    manifest=dict(status='prepared',root=str(root.resolve()),source=source,protocol_sha256=digest(root/'protocol.json'),
                  reference_manifests=refs,jobs=[],budget=budget(cfg),prefix_scope='Fresh unified CUDA models; legacy CPU300 curves are separate.')
    for seed in cfg['seeds']:
        for group in cfg['groups']:
            variant = VARIANTS[group];tag=group+'_'+str(seed);work=root/'jobs'/tag;work.mkdir(parents=True)
            output=work/'training/runs/case/moves_5'
            common=['--algorithm','tg-mappo','--episode_length','80','--num_moving_buoys','5','--random_channel','true',
                    '--ppo_epoch','1','--num_mini_batch','2','--tg_reward_version','task_v2','--tg_device','cuda',
                    '--pm_action_credit',variant['mode'],'--pm_critic_context','disabled']
            if variant['wm']:common+=['--use_wm_actor','true','--wm_actor_checkpoint',cfg['wm_checkpoint']]
            if variant['pm']:common+=['--use_pm_reward','true','--pm_reward_checkpoint',cfg['pm_checkpoint'],'--pm_reward_alpha','.1',
                                      '--pm_potential_mode','task_return','--pm_value_mode','base_residual']
            training=common+['--seed',str(seed),'--scenario_seed',str(seed),'--num_episodes',str(cfg['episodes']),
                             '--num_env_steps',str(cfg['episodes']*80),'--snapshot',str(root/'source_snapshot'),
                             '--output_root',str(work/'training'),'--run-name','case']
            ev=[cfg['python'],'-s','-u',str(root/'source_snapshot/scripts/tg_eval_worker.py')]+common+['--run-name','case']
            refout=None
            if reference and seed == 22:
                old=next(j for j in previous['jobs'] if j['group']==group and j['seed']==22);refout=old['output']
            spec=dict(root=str(root.resolve()),work_dir=str(work),group=group,mode=variant['mode'],seed=seed,output=str(output),
                      reference_episodes=2 if refout else None,reference_output=refout,eval_command=ev)
            atomic(work/'job.json',spec)
            cmd=[cfg['python'],'-s','-u',str(root/'source_snapshot/scripts/tg_convergence_worker.py'),'--study-job',str(work/'job.json')]+training
            manifest['jobs'].append(dict(spec,tag=tag,command=cmd,state='pending',job_spec_sha256=digest(work/'job.json')))
    atomic(root/'manifest.json',manifest)
    print('OUTPUT '+str(root.resolve()),flush=True);print(json.dumps(manifest['budget']),flush=True)
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare');g.add_argument('--run');a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    else:
        from run_tg_convergence import run
        run(a.run)


if __name__ == '__main__':main()
