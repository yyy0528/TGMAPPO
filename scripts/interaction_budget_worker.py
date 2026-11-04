"""Run the original TG entry point and capture a read-only prefix checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch


def equal_states(left,right):
    import torch
    return set(left)==set(right) and all(torch.equal(left[k],right[k]) for k in left)


def capture_prefix(learner,root,reference=None):
    import torch
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    rng=torch.get_rng_state().clone()
    payload={name:{k:v.detach().cpu().clone() for k,v in getattr(learner.policy,name).state_dict().items()}
             for name in ('actor','critic')}
    if reference is not None:
        expected=(torch.load(str(reference),map_location='cpu',weights_only=True)
                  if int(torch.__version__.split('.')[0])>=2 else torch.load(str(reference),map_location='cpu'))
        if not all(equal_states(payload[n],expected[n]) for n in payload):
            raise ValueError('Prefix Actor/Critic differs from preserved reference checkpoint')
    path=root/('prefix_at_{}.pt'.format(learner.profile['ppo_updates']))
    with path.open('xb') as stream:
        torch.save(payload,stream)
    if not torch.equal(rng,torch.get_rng_state()):
        raise AssertionError('Prefix capture consumed Torch RNG')
    report={'update':learner.profile['ppo_updates'],'environment_steps':learner.profile['collect_calls'],
            'reference':str(Path(reference).resolve()) if reference else None,'reference_exact':reference is not None,
            'checkpoint':str(path.resolve()),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'torch_rng_unchanged':True,'training_updates':[dict(row) for row in learner.training_records]}
    (root/('prefix_at_{}.json'.format(learner.profile['ppo_updates']))).write_text(
        json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def install_capture(learner_class,update,root,reference=None):
    original=learner_class.train
    observed=[]
    def train(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if self.profile['ppo_updates']==update:
            observed.append(capture_prefix(self,root,reference))
        return result
    return patch.object(learner_class,'train',train),observed


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--capture-update',type=int,default=20)
    parser.add_argument('--prefix-reference',required=True)
    special,argv=parser.parse_known_args()
    if special.capture_update<1:
        raise ValueError('Capture update must be positive')
    snapshot=Path(special.snapshot).resolve()
    sys.path[:0]=[str(snapshot),str(snapshot/'src')]
    import run_algorithm as launcher
    known,extra=launcher.build_parser().parse_known_args(argv)
    if known.algorithm!='tg-mappo':
        raise ValueError('Interaction worker only supports original TG-MAPPO')
    launcher.reject_research_options(extra)
    output=Path(launcher.create_run_root(known.output_root,known.run_name))
    arguments=launcher.with_tg_mappo_defaults(list(extra)+['--output_root',str(output)])
    from buoy_network.agent import DeepQNetwork
    from buoy_network.main import main as original_main
    context,observed=install_capture(DeepQNetwork,special.capture_update,output/'prefix',special.prefix_reference)
    with context:
        original_main(arguments)
    if len(observed)!=1:
        raise AssertionError('Required prefix was not captured exactly once')
    print('PASS original TG prefix weights exact; RNG unchanged',flush=True)


if __name__=='__main__':
    main()
