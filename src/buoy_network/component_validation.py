"""Held-out component metrics, explicitly respecting finite-window censoring."""
import torch


def calibration(probability,labels,weights=None,bins=5):
    p=torch.as_tensor(probability,dtype=torch.float64).reshape(-1)
    y=torch.as_tensor(labels,dtype=torch.float64).reshape(-1)
    w=torch.ones_like(p) if weights is None else torch.as_tensor(weights,dtype=torch.float64).reshape(-1)
    if p.shape!=y.shape or p.shape!=w.shape or not torch.isfinite(p).all() or not torch.isfinite(y).all() or not torch.isfinite(w).all() or (w<0).any() or ((p<-1e-9)|(p>1+1e-9)).any() or ((y<0)|(y>1)).any() or w.sum()<=0:
        raise ValueError('Invalid calibration inputs')
    p=p.clamp(0,1)
    result=[];ece=0.;total=float(w.sum())
    for i in range(bins):
        select=(p>=i/bins)&((p<(i+1)/bins) if i<bins-1 else p<=1)&(w>0)
        if select.any():
            mass=w[select].sum();pred=(p[select]*w[select]).sum()/mass;actual=(y[select]*w[select]).sum()/mass
            ece+=float(mass/total*(pred-actual).abs())
            result.append(dict(bin=i,count=int(select.sum()),weight=float(mass),predicted=float(pred),observed=float(actual)))
    return dict(ece=ece,bins=result,weight_total=total)


def packet_weighted_mean(values,mask):
    if values.shape!=mask.shape or values.ndim!=2:raise ValueError('Invalid packet-prefix shape')
    count=mask.sum(-1);known=count>0
    if not known.any():return None
    per_packet=(values*mask).sum(-1)/count.clamp_min(1)
    return float(per_packet[known].mean())


def observed_task_return(lengths,events,horizon,gamma=.99):
    lengths=torch.as_tensor(lengths);events=torch.as_tensor(events,dtype=torch.bool)
    if type(horizon) is not int or horizon<1 or not 0<gamma<=1 or lengths.ndim!=1 or events.shape!=lengths.shape or (lengths<1).any() or (lengths>horizon).any():
        raise ValueError('Invalid observed finite-task labels')
    t=torch.arange(horizon)[None,:];remaining=horizon-t;valid=t<lengths[:,None]
    known=valid&(events[:,None]|(lengths[:,None]>=horizon))
    duration=torch.where(events[:,None],lengths[:,None]-t,remaining).clamp_min(0).double()
    wait=duration if gamma==1 else (1-gamma**duration)/(1-gamma)
    outcome=torch.where(events[:,None],torch.ones_like(duration),-torch.ones_like(duration))
    result=outcome*gamma**(duration-1).clamp_min(0)-wait/horizon
    return result,known


def survival_diagnostics(logits,prior_logits,data):
    from .pm_return import expected_task_return
    from .task_reward import reward_contract
    from types import SimpleNamespace
    logits=logits.detach().cpu();prior=prior_logits.detach().cpu().expand_as(logits)
    horizon=data['horizon'];lengths=data['lengths'].cpu();events=data['events'].cpu()
    actual,known=observed_task_return(lengths,events,horizon)
    remaining=(horizon-torch.arange(horizon))[None,:].expand(len(lengths),-1)
    contract=reward_contract(SimpleNamespace(tg_reward_version='task_v2'))
    current=expected_task_return(logits,remaining,horizon,.99,contract)
    base=expected_task_return(prior,remaining,horizon,.99,contract)
    labels=events[:,None].expand_as(remaining)
    weights=known.double()/known.sum(-1,keepdim=True).clamp_min(1)
    return dict(task_return_mse=packet_weighted_mean((current['task_return']-actual).square(),known),
        prior_task_return_mse=packet_weighted_mean((base['task_return']-actual).square(),known),
        task_return_bias=packet_weighted_mean(current['task_return']-actual,known),
        identifiable_packets=int(known.any(-1).sum()),unknown_prefixes_excluded=int(((torch.arange(horizon)[None,:]<lengths[:,None])&~known).sum()),
        completion_calibration=calibration(current['probability'],labels,weights),
        scope='Equal packet weight over identifiable prefixes; finite task reward, not eventual failure or exact policy value')
