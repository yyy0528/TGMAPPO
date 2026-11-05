"""Audit seed22 frozen actions, packet routes and reconstructed first PPO inputs."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from buoy_network.ablation_suite import read,digest,require
from buoy_network.credit_diagnostics import advantages
from tg_seed22_signals import signal_arrays,packet_waiting,aggregate


def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def first_rollout(root):
    folder=root/'initial_sample';cache=read(folder/'raw_credit_cache.json')['cache'];frames=lines(folder/'slots.jsonl')
    replay=read(folder/'report.json')
    require(replay['weights_unchanged'] and replay['optimizer_updates']==0 and replay['stage']=='initial'
            and replay['actor_delta_from_initial_l2']==0,'Initial replay was not frozen initialization')
    require(len(cache)==len(frames)==80,'Incomplete initial replay')
    values=np.asarray([x['values'] for x in cache]);phi=np.asarray([x['shadow_phi'] for x in cache]+[0.])
    base=np.asarray([x['reward_env'] for x in frames]);masks=np.asarray([x['mask'] for x in frames]);active=masks[:,:,:25].any(-1)
    actions=np.asarray([x['actions'] for x in frames]);probs=np.asarray([x['probabilities'] for x in frames])
    view=read(root/'view/manifest.json');result={'matched':{},'groups':{},'rows':[]};zs={}
    initial_physics=lines(folder/'moves_5/logs/TG_physical_metrics.jsonl')[0]
    for group,alpha in (('A',0.),('task_return',1.),('alpha01',.1)):
        j=next(j for j in view['jobs'] if j['group']==group and j['kind']=='training')
        out=Path(j['output']);require(lines(out/'logs/TG_physical_metrics.jsonl')[0]==initial_physics,'Initial physical rollout differs')
        shaped=base+alpha*(.99*phi[1:]-phi[:-1])
        calc=signal_arrays(base,shaped,values,phi,masks,.99,.95,alpha)
        if group!='A':
            with np.load(str(out.parent/'value_audit/episode_0.npz')) as a:
                require(np.array_equal(values,a['raw_values'][:-1,0,:,0]) and np.array_equal(base,a['base_rewards'][:,0,0,0])
                    and np.array_equal(phi,a['phi'][:,0,0,0]),'Initial values/rewards/PM history differ')
                raw=a['actual_gae'][:,0,:,0].copy()
                error=float(np.max(np.abs(raw-calc['raw'])))
                require(error<1e-5,'Reconstructed first GAE mismatch')
            result['matched'][group]={'original_values_rewards_Phi_exact':True,'physical_exact':True,'float32_GAE_error':error}
        else:
            raw=calc['raw'].astype(np.float32)
            result['matched'][group]={'physical_exact':True,'scope':'base-reward comparison on verified initial policy; no original A value cache'}
        _,z=advantages(raw,np.zeros_like(raw),active);zs[group]=z
        comparison=[]
        for action in (17,19,25):
            chosen=active[:,13]&(actions[:,13]==action)
            comparison.append({'action':action,'count':int(chosen.sum()),
                'mean_standardized_advantage':float(z[:,13][chosen].mean()) if chosen.any() else None})
        common=active[:,13]&(masks[:,13,17]>0)&(masks[:,13,19]>0)
        contrast=z[:,13]*((actions[:,13]==17).astype(float)-(actions[:,13]==19).astype(float)-probs[:,13,17]+probs[:,13,19])
        result['groups'][group]={'node13_actions':comparison,'contrast_17_minus_19_score_mean':float(contrast[common].mean()),
            'contrast_count':int(common.sum()),'scope':'ratio=1 likelihood term only, not the historical parameter/entropy update'}
    flips=active&(zs['task_return']*zs['alpha01']<0)
    result.update(active_decisions=int(active.sum()),normalized_advantage_sign_flips=int(flips.sum()))
    slot_means={g:np.asarray([z[t][active[t]].mean() for t in range(80) if active[t].any()]) for g,z in zs.items()}
    result['joint_slots']=len(slot_means['task_return'])
    result['joint_slot_advantage_sign_flips']=int((slot_means['task_return']*slot_means['alpha01']<0).sum())
    for t,node in np.argwhere(active):
        result['rows'].append({'step':int(t),'sender':int(node),'action':int(actions[t,node]),
            'base_z':float(zs['A'][t,node]),'alpha1_z':float(zs['task_return'][t,node]),'alpha01_z':float(zs['alpha01'][t,node])})
    (root/'first_rollout_rows.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in result.pop('rows')),encoding='utf-8')
    result['limitations']=['Only the first rollout is reconstructed; later historical actions/probability changes were not cached.',
        'Initial actor executes without loading trained weights; shadow PM only reads the environment.',
        'This does not identify the update that caused the final 17/19 ranking reversal.']
    return result


def audit(root):
    m=read(root/'manifest.json');require(m['status']=='completed' and len(m['jobs'])==6,'Incomplete frozen diagnostics')
    for path,sha in m['reference_manifests'].items():require(digest(path)==sha,'Original manifest changed')
    view=read(root/'view/manifest.json');snapshot=Path(m['source_snapshot'])
    for path,sha in view['source']['files_sha256'].items():require(digest(snapshot/path)==sha,'Archived runtime source changed')
    summaries=[];all_packets={};artifacts=[]
    for job in m['jobs']:
        out=Path(job['output']);require(job['state']=='completed' and job['exit_code']==0,'Replay incomplete')
        r,s=read(out/'report.json'),read(out/'signals.json')
        require(r['weights_unchanged'] and r['optimizer_updates']==0 and s['optimizer_updates']==0,'Weights/updates changed')
        training=next(j for j in view['jobs'] if j['group']==job['group'] and j['kind']=='training')
        original_report=read(Path(training['output'])/'logs/TG_policy_report.json')
        require(digest(Path(training['output'])/'checkpoints/tg_policy_latest.pt')==r['policy_sha256']
                ==original_report['checkpoint_sha256'],'Original checkpoint changed')
        require(digest(out/'report.json')==job['report_sha256'] and digest(out/'signals.json')==job['signals_sha256'],'Diagnostic report changed')
        if job['decode']=='greedy':
            ref=next(j for j in view['jobs'] if j['group']==job['group'] and j['kind']=='evaluation' and j['scene']==8001)
            require(digest(out/'moves_5/logs/TG_physical_metrics.jsonl')==digest(Path(ref['output'])/'logs/TG_physical_metrics.jsonl'),'Greedy replay differs')
        frames=lines(out/'slots.jsonl');raw=read(out/'raw_credit_cache.json');cache=raw['cache']
        credit=lines(out/'credit_rows.jsonl')
        require(aggregate(credit)==s['groups'],'Action/advantage aggregation mismatch')
        require(len(frames)==len(cache)==160,'Missing frames')
        recomputed=[]
        for ep in (0,1):
            f,c=frames[ep*80:(ep+1)*80],cache[ep*80:(ep+1)*80]
            a=signal_arrays([x['reward_env'] for x in f],[x['reward_train'] for x in f],[x['values'] for x in c],
                [x['phi'] for x in c]+[0.],[x['mask'] for x in f],raw['context']['gamma'],raw['context']['lam'],s['alpha'])
            with np.load(str(out/('signals_episode_'+str(ep)+'.npz'))) as stored:
                require(np.array_equal(a['raw'],stored['raw_advantages']) and np.array_equal(a['normalized'],stored['normalized_advantages']),'Signal arrays differ')
            for row in credit:
                if row['episode']==ep:
                    t,node=row['step'],row['sender']
                    require(row['action']==f[t]['actions'][node] and row['advantage']==float(a['normalized'][t,node])
                            and row['raw_advantage']==float(a['raw'][t,node]),'Action/value time indices differ')
            require(a['value_coordinate_error']<1e-9,'Reward/value coordinate identity failed')
            recomputed.append(a['value_coordinate_error'])
        packet_rows=packet_waiting(frames,cache,lines(out/'packet_routes.jsonl'),25)
        require(packet_rows==read(out/'packet_waiting.json'),'Packet service ledger differs')
        all_packets[job['tag']]={(x['episode'],x['packet_id']):x for x in packet_rows}
        report=r['analysis']
        summaries.append({'group':job['group'],'decode':job['decode'],**{k:report[k] for k in
            ('delivered','generated','active_decisions','active_wait_count','active_wait_fraction','sink_legal_count','sink_chosen_count',
             'revisit_transfers','mean_normalized_entropy','mean_top_probability_gap')},'groups':s['groups'],
             'value_coordinate_error_max':max(recomputed)})
        artifacts.append({'tag':job['tag'],'report_sha256':digest(out/'report.json'),'signals_sha256':digest(out/'signals.json'),
            'raw_cache_sha256':digest(out/'raw_credit_cache.json'),'packet_ledger_sha256':digest(out/'packet_waiting.json'),
            'input_cache_sha256':digest(out/'actor_inputs.npz')})
    base,low=all_packets['A_greedy'],all_packets['alpha01_greedy']
    require(set(base)==set(low) and all(base[k]['route'][0]==low[k]['route'][0] for k in base),'Packet identities differ')
    differences=[{'episode':k[0],'packet_id':k[1],'baseline':base[k],'alpha01':low[k]} for k in base if base[k]['delivered']!=low[k]['delivered']]
    cross=read(root/'same_inputs/report.json')
    require(cross['own_probability_max_error']<1e-6 and all(x['weights_unchanged'] and x['RNG_unchanged'] for x in cross['fingerprints'].values()),'Same-input replay invariant failed')
    packet_pm=read(root/'packet_pm_replay/packet13_signals.json')
    require(packet_pm['physical_exact'] and packet_pm['weights_unchanged'] and packet_pm['optimizer_updates']==0,'PM extraction changed policy')
    require(digest(root/'packet_pm_replay/moves_5/logs/TG_physical_metrics.jsonl')==digest(root/'alpha01_greedy/moves_5/logs/TG_physical_metrics.jsonl'),'Per-packet replay differs')
    return {'diagnosis_complete':True,'training_started':False,'optimizer_updates':0,'root':str(root),
        'manifest_sha256':digest(root/'manifest.json'),'auditor_sha256':digest(Path(__file__)),
        'summaries':summaries,'packet_delivery_differences':differences,'same_inputs':cross,
        'first_rollout':first_rollout(root),'packet_PM':packet_pm,'artifacts':artifacts,
        'new_environment_steps':1200,
        'budget_description':'6 frozen 160-slot replays + 80-slot initial replay + 160-slot per-packet replay = 1200 slots',
        'limitations':['One known training seed and development scene; no new efficacy claim.',
            'Frozen-policy advantages differ from original training histories except verified initial rollout.',
            'Exact historical parameter updates after rollout zero require an instrumented training replay; not done.']}


def plot(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=['A','task_return','alpha01'];labels=['TG','PM alpha=1','PM alpha=0.1']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,decode in ((axes[0,0],'greedy'),(axes[0,1],'sample')):
        counts=[next(x['delivered'] for x in report['summaries'] if x['group']==g and x['decode']==decode) for g in groups]
        ax.bar(range(3),counts,color=['#6b8fad','#62a085','#d38350'])
        for i,v in enumerate(counts):ax.text(i,v+.5,str(v),ha='center')
        ax.set_xticks(range(3));ax.set_xticklabels(labels);ax.set_ylim(0,25)
        ax.set_ylabel('Delivered / 48');ax.set_title('Frozen '+decode+' replay');ax.grid(axis='y',alpha=.2)
    sample=report['same_inputs']['packet13_examples'][0];ax=axes[1,0];xs=np.arange(3)
    for offset,action,color in ((-.18,17,'#d38350'),(.18,19,'#6b8fad')):
        values=[next(v for k,v in sample['rankings'][g] if k==action) for g in groups]
        ax.bar(xs+offset,values,width=.36,color=color,label='Next hop '+str(action))
    ax.set_xticks(xs);ax.set_xticklabels(labels);ax.set_ylim(0,.18);ax.set_ylabel('Action probability')
    ax.set_title('Node 13, identical step-0 observation');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1,1];frame=lines(root/'alpha01_greedy/slots.jsonl')[0]
    pos={n:frame['positions'][n] for n in (12,13,17,19,24)}
    for src,dst,color,rad in ((13,19,'#3977a5',0),(19,24,'#3977a5',0),(13,17,'#c45e46',0),
                               (17,12,'#c45e46',.18),(12,17,'#c45e46',.18)):
        ax.annotate('',xy=pos[dst],xytext=pos[src],arrowprops=dict(arrowstyle='->',color=color,lw=2,
            shrinkA=12,shrinkB=12,connectionstyle='arc3,rad='+str(rad)))
    for node,(x,y) in pos.items():
        ax.scatter([x],[y],s=170,color='#e0e6eb',zorder=3)
        ax.text(x,y,'Sink' if node==24 else str(node),ha='center',va='center',fontsize=9,zorder=4)
    ax.set_xlim(min(x[0] for x in pos.values())-.4,max(x[0] for x in pos.values())+.4)
    ax.set_ylim(min(x[1] for x in pos.values())-.4,max(x[1] for x in pos.values())+.4)
    ax.set_aspect('equal');ax.axis('off');ax.set_title('Packet 13: blue TG; red low-alpha loop')
    fig.suptitle('Seed 22 diagnosis: fixed checkpoints; no PPO updates')
    fig.tight_layout(rect=[0,0,1,.95]);out=root/'figures';out.mkdir(exist_ok=True)
    fig.savefig(str(out/'seed22_actions_credit.png'),dpi=160);fig.savefig(str(out/'seed22_actions_credit.svg'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args();root=Path(a.root).resolve()
    report=audit(root);plot(report,root)
    report['figures']={str(p.relative_to(root)):digest(p) for p in (root/'figures').iterdir() if p.is_file()}
    (root/'acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS frozen policies, packet ledgers, same-input comparisons and initial rollout reconstruction')
    print('REPORT '+str(root/'acceptance.json'))


if __name__=='__main__':main()
