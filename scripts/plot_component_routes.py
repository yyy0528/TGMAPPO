"""Plot archived packet routes with an external legend; never reruns a policy."""
import argparse,json,sys
from pathlib import Path


def render(ledger,prefix,episode=0,title='Packet routes at final node coordinates'):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    data=json.loads(Path(ledger).read_text(encoding='utf-8'))
    row=next(x for x in data if x['episode']==episode)
    positions=row['final_positions'];packets=row['packets']
    unfinished=sorted([(k,v) for k,v in packets.items() if v['delay'] is None],key=lambda item:int(item[0]))
    delivered=sorted([(k,v) for k,v in packets.items() if v['delay'] is not None],key=lambda item:(len(item[1]['route']),int(item[0])))[:2]
    if not positions or any(not p['route'] or any(type(n) is not int or not 0<=n<len(positions) for n in p['route']) for _,p in unfinished+delivered):raise ValueError('Invalid packet routes')
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    targets=[Path(str(prefix)+'.'+ext) for ext in ['png','svg','pdf','json']]
    if any(p.exists() for p in targets):raise FileExistsError('Refuse to overwrite a previous figure export')
    fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(left=.07,right=.72,bottom=.09,top=.89)
    for n,xy in enumerate(positions):
        ax.scatter(*xy,color='#aaaaaa',s=25,zorder=2);ax.annotate(str(n),xy,xytext=(3,4),textcoords='offset points',fontsize=8)
    for k,p in unfinished+delivered:
        xs=[positions[n][0] for n in p['route']];ys=[positions[n][1] for n in p['route']]
        unresolved=p['delay'] is None
        ax.plot(xs,ys,'-o',alpha=.7,color='#ba4035' if unresolved else '#33885b',label='packet{} {}'.format(k,'unfinished' if unresolved else 'delivered'))
        if len(xs)==1:ax.scatter(xs,ys,color='#ba4035',marker='x',s=60,zorder=3)
    ax.set_aspect('equal');ax.margins(.07);ax.set_xlabel('Grid x');ax.set_ylabel('Grid y')
    fig.suptitle(title,fontsize=12);ax.set_title('Routes drawn on final coordinates; movement paths are not reconstructed',fontsize=9)
    if unfinished or delivered:ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0,fontsize=8)
    for p in targets[:-1]:fig.savefig(p,dpi=160,bbox_inches='tight')
    plt.close(fig)
    info=dict(source=str(Path(ledger).resolve()),episode=episode,unfinished=[k for k,_ in unfinished],delivered_examples=[k for k,_ in delivered],coordinates='final node positions, not historical trajectories')
    targets[-1].write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ledger');p.add_argument('prefix');p.add_argument('--episode',type=int,default=0);p.add_argument('--title',default='Packet routes at final node coordinates');a=p.parse_args()
    print(json.dumps(render(a.ledger,a.prefix,a.episode,a.title)))
