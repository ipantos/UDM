"""Predetermined coarse-profile/global-start audit and fine-step LR checks."""
from run_coverage import *
def audit(config,record):
    rng=np.random.default_rng(record['seed']);base=Likelihood()
    obs=[np.array(m)+ch@rng.standard_normal(len(m)) for m,ch in zip(config['mean'],base.chols)]
    like=Likelihood(observations=obs);vel=config['velocity'];truthcs=config['cs']
    uu=np.unique(np.r_[-12.,-9.,-8.,-7.5,-7.,-6.5,-6.,-5.5,-5.,-4.5,-4.,-3.5,-3.,-2.5,-2.,np.log10(truthcs)])
    points=[];prev=record['eps_best']
    for u in uu:
        cs=10.**u
        trials=[general_fit(like,cs,start=st,velocity=vel) for st in (prev,record['eps_fixed'])]
        good=[t for t in trials if t['valid'] and t['success']]
        if not good:
            trials.extend(general_fit(like,cs,start=st,velocity=vel,steps=1600,maxiter=600) for st in anchors()+[np.full(7,-1.)])
            good=[t for t in trials if t['valid'] and t['success']]
        if not good:return dict(index=record['index'],failed=True,at_u=float(u))
        best=min(good,key=lambda x:x['chi2']);prev=best['eps'];points.append(best)
    pmin=min(points,key=lambda x:x['chi2'])
    # A grid point better than the continuous fit signals a missed basin.
    improvement=record['best_chi2']-pmin['chi2']
    fine=Solver(1600,vel)
    fixed_fine=float(like.profile(general_prediction(record['eps_fixed'],truthcs,fine))[0])
    best_fine=float(like.profile(general_prediction(record['eps_best'],record['best_cs'],fine))[0])
    qfine=0. if record['best_cs']>truthcs else max(0.,fixed_fine-best_fine)
    cs=np.array([p['cs'] for p in points]);ch=np.array([p['chi2'] for p in points]);imin=int(ch.argmin())
    crossing=ext.crossing(cs[imin:],(ch-min(ch.min(),record['best_chi2']))[imin:],2.706)
    return dict(index=record['index'],failed=False,max_grid_improvement=improvement,q_fine=qfine,q_change=qfine-record['q'],
                coarse_first_crossing= crossing,points=points)

def run(vel):
    path=RESULTS/f'coverage_profile_audit_{vel}.json'
    out=[r for r in json.loads(path.read_text()) if not r['failed']] if path.exists() else []
    done={(r['factor'],r['index']) for r in out}
    jobs=[]
    for factor in (1.,.5):
        config=truth(vel,factor)
        records=[json.loads(l) for l in (RESULTS/f'coverage_general_{vel}_{factor:g}.jsonl').read_text().splitlines()]
        # First 10 of the every-100 predetermined indices; independent of result.
        jobs.extend((config,r) for r in records if r['index']%100==0 and not r['failed'] and (factor,r['index']) not in done)
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(audit,c,r):c['factor'] for c,r in jobs}
        for task in as_completed(futures):
            v=task.result();v['factor']=futures[task];out.append(v)
            (RESULTS/f'coverage_profile_audit_{vel}.json').write_text(json.dumps(out,indent=2))
            print(vel,v['factor'],v['index'],v.get('max_grid_improvement'),v.get('q_change'),flush=True)
if __name__=='__main__':run(sys.argv[1])
