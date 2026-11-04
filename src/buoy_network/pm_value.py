"""Raw reward/value coordinate identities for a frozen PM on full finite tasks.

The residual critic predicts U=V_base. Its shaped value is U-alpha*Phi.
An aligned frozen potential cancels from each TD residual, not just MC return.
This is an invariance control, not evidence of an additional PM policy benefit.
"""
import numpy as np


def gae_from_delta(delta, masks_next, gamma, lam):
    result=np.zeros_like(delta,dtype=np.float64)
    following=np.zeros_like(delta[0],dtype=np.float64)
    for t in reversed(range(len(delta))):
        following=delta[t]+gamma*lam*masks_next[t]*following
        result[t]=following
    return result


def coordinate_audit(base_rewards, shaped_rewards, values, phi, masks, gamma, lam, alpha):
    base,shaped,values,phi,masks=[np.asarray(x,dtype=np.float64) for x in
                                (base_rewards,shaped_rewards,values,phi,masks)]
    if (base.shape!=shaped.shape or values.shape!=(len(base)+1,)+base.shape[1:]
            or phi.shape!=values.shape or masks.shape!=values.shape
            or not all(np.isfinite(x).all() for x in (base,shaped,values,phi,masks))
            or not 0<gamma<=1 or not 0<=lam<=1 or not np.isfinite(alpha) or alpha<0):
        raise ValueError('Invalid PM value coordinate arrays/settings')
    expected=base+alpha*(gamma*phi[1:]*masks[1:]-phi[:-1])
    if not np.allclose(shaped,expected,rtol=1e-6,atol=1e-7):
        raise ValueError('PM rewards/potential/terminal masks do not align')
    base_delta=base+gamma*values[1:]*masks[1:]-values[:-1]
    legacy_delta=shaped+gamma*values[1:]*masks[1:]-values[:-1]
    shaped_values=values-alpha*phi
    aligned_delta=shaped+gamma*shaped_values[1:]*masks[1:]-shaped_values[:-1]
    base_gae=gae_from_delta(base_delta,masks[1:],gamma,lam)
    legacy_gae=gae_from_delta(legacy_delta,masks[1:],gamma,lam)
    aligned_gae=gae_from_delta(aligned_delta,masks[1:],gamma,lam)
    return {'base_delta':base_delta,'legacy_same_numeric_value_delta':legacy_delta,
            'aligned_delta':aligned_delta,'base_gae':base_gae,
            'legacy_same_numeric_value_gae':legacy_gae,'aligned_gae':aligned_gae,
            'shaped_values':shaped_values}


def prepare_residual_rollout(buffer, runtime, normalizer):
    """Validate immutable cached PM rows and return base-coordinate rewards.

    Keep buffer.rewards as collected shaped rewards. Returns/value_preds/PopArt
    remain entirely in base coordinates; never subtract raw Phi from normalized V.
    """
    rows=runtime.episode_records
    t=buffer.episode_length
    if (len(rows)!=t or not buffer._use_gae or
            any(r['step']!=i or r['terminated']!=(i==t-1) or r['rollout_truncated'] for i,r in enumerate(rows))
            or not np.all(buffer.masks[1:-1]==1) or not np.all(buffer.masks[-1]==0)
            or not np.all(buffer.bad_masks==1)):
        raise ValueError('PM base residual supports complete finite-task GAE rollouts only')
    if (rows[-1]['phi_tp1']!=0 or any(abs(rows[i]['phi_tp1']-rows[i+1]['phi_t'])>1e-10 for i in range(t-1))):
        raise ValueError('PM potential cache time/boundary mismatch')
    shape=buffer.rewards.shape
    def expand(values):
        return np.broadcast_to(np.asarray(values,dtype=np.float64).reshape((-1,)+(1,)*(len(shape)-1)),
                               (len(values),)+shape[1:]).copy()
    base=expand([r['reward_env'] for r in rows])
    shaped=expand([r['reward_train'] for r in rows])
    phi=expand([r['phi_t'] for r in rows]+[rows[-1]['phi_tp1']])
    if not np.allclose(buffer.rewards,shaped,rtol=1e-6,atol=1e-7):
        raise ValueError('Collected PM rewards differ from immutable PM cache')
    values=normalizer.denormalize(buffer.value_preds) if buffer._use_popart or buffer._use_valuenorm else buffer.value_preds
    arrays=coordinate_audit(base,shaped,values,phi,buffer.masks,buffer.gamma,buffer.gae_lambda,runtime.alpha)
    def error(a,b):return float(np.max(np.abs(a-b)))
    record={'raw_td_identity_max_error':error(arrays['base_delta'],arrays['aligned_delta']),
            'raw_gae_identity_max_error':error(arrays['base_gae'],arrays['aligned_gae']),
            'same_numeric_value_gae_change_max':error(arrays['base_gae'],arrays['legacy_same_numeric_value_gae']),
            'critic_output_coordinate':'base','popart_target_coordinate':'base',
            'effective_shaped_value':'V_base - alpha * Phi',
            'interpretation':'analytic invariance control, not additional PM benefit'}
    if max(record['raw_td_identity_max_error'],record['raw_gae_identity_max_error'])>1e-9:
        raise ValueError('PM value/GAE coordinate identity failed')
    # Algebraically simplified base rewards avoid float32 cancellation drift.
    return base.astype(buffer.rewards.dtype),record
