"""Original public TG entry plus read-only cached value/GAE diagnostics."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--snapshot',required=True)
    special,argv=p.parse_known_args()
    snapshot=Path(special.snapshot).resolve()
    sys.path[:0]=[str(snapshot),str(snapshot/'src')]
    import numpy as np
    import run_algorithm as launcher
    from buoy_network.agent import DeepQNetwork
    from buoy_network.main import main as original_main
    from buoy_network.pm_value import coordinate_audit
    known,extra=launcher.build_parser().parse_known_args(argv)
    if known.algorithm!='tg-mappo':raise ValueError('Original TG only')
    launcher.reject_research_options(extra)
    output=Path(launcher.create_run_root(known.output_root,known.run_name))
    arguments=launcher.with_tg_mappo_defaults(list(extra)+['--output_root',str(output)])
    compute=DeepQNetwork.compute
    def observed(self):
        compute(self)
        runtime=self.pm_runtime
        if runtime is None:return
        b=self.buffer;rows=runtime.episode_records
        def expand(x):return np.broadcast_to(np.asarray(x,dtype=np.float64).reshape(-1,1,1,1),
                                             (len(x),)+b.rewards.shape[1:]).copy()
        base=expand([r['reward_env'] for r in rows]);shaped=expand([r['reward_train'] for r in rows])
        phi=expand([r['phi_t'] for r in rows]+[rows[-1]['phi_tp1']])
        values=(self.trainer.value_normalizer.denormalize(b.value_preds) if b._use_popart or b._use_valuenorm else b.value_preds.copy())
        a=coordinate_audit(base,shaped,values,phi,b.masks,b.gamma,b.gae_lambda,runtime.alpha)
        selected=a['base_gae'] if runtime.value_mode=='base_residual' else a['legacy_same_numeric_value_gae']
        actual=b.returns[:-1]-values[:-1]
        active=b.active_masks[:-1]>0
        g_base=np.zeros_like(base);g_shaped=np.zeros_like(base)
        for t in reversed(range(len(base))):
            g_base[t]=base[t]+(b.gamma*g_base[t+1]*b.masks[t+1] if t+1<len(base) else 0)
            g_shaped[t]=shaped[t]+(b.gamma*g_shaped[t+1]*b.masks[t+1] if t+1<len(base) else 0)
        effective=a['shaped_values'] if runtime.value_mode=='base_residual' else values
        record={'episode':runtime.episode,'value_mode':runtime.value_mode,'gamma':b.gamma,'lambda':b.gae_lambda,
            'aligned_td_error':float(np.max(np.abs(a['base_delta']-a['aligned_delta']))),
            'aligned_gae_error':float(np.max(np.abs(a['base_gae']-a['aligned_gae']))),
            'same_numeric_value_gae_change':float(np.max(np.abs(a['base_gae']-a['legacy_same_numeric_value_gae']))),
            'actual_buffer_gae_error_float32':float(np.max(np.abs(selected-actual))),
            'sampled_effective_value_minus_shaped_MC_active_mean':float((effective[:-1]-g_shaped)[active].mean()),
            'scope':'cached same-trajectory counterfactual; MC is sampled, not true conditional V'}
        folder=output/'value_audit';folder.mkdir(exist_ok=True)
        context=getattr(b,'critic_progress',None)
        if context is not None and not np.array_equal(context,phi.astype(context.dtype)):
            raise ValueError('PM Critic cache must contain decision-time Phi, including terminal zero')
        targets=getattr(b,'pm_action_targets',None)
        if targets is not None:
            if not b.pm_action_ready.all() or not np.isfinite(targets).all() or not np.allclose(targets.sum(-1),1.,atol=1e-5) or np.any(targets[b.available_actions[:-1]==0]!=0):
                raise ValueError('PM action target cache invalid or misaligned')
        np.savez_compressed(str(folder/('episode_{}.npz'.format(runtime.episode))),
            base_rewards=base,shaped_rewards=shaped,raw_values=values,phi=phi,masks=b.masks,
            base_gae=a['base_gae'],aligned_gae=a['aligned_gae'],actual_gae=actual,
            **dict(({'critic_progress':context.copy()} if context is not None else {}),
                   **({'pm_action_targets':targets.copy(),'pm_available_actions':b.available_actions[:-1].copy(),'pm_active_masks':b.active_masks[:-1].copy()} if targets is not None else {})))
        with (folder/'summary.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    with patch.object(DeepQNetwork,'compute',observed):original_main(arguments)


if __name__=='__main__':main()
