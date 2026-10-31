"""Parametric coverage of one-sided profile-LR tests in the compressed likelihood.

Each mock redraws all BAO, SN and growth entries using the stated covariance
blocks, and refits background and all three linear nuisance parameters.
This is NOT a test of the survey compression or of the assumed covariances.
"""
from run_profiles import *
from concurrent.futures import ProcessPoolExecutor, as_completed

def truth(velocity,factor):
    limit=8.795398874832254e-6 if velocity=='pressureless' else 7.8e-7
    cs=limit*factor
    like=Likelihood()
    starts=anchors()
    fits=[general_fit(like,cs,st,velocity=velocity) for st in starts]
    best=min([f for f in fits if f['valid'] and f['success']],key=lambda f:f['chi2'])
    pred=general_prediction(best['eps'],cs,Solver(1600,velocity))
    d=like.profile(pred,True)
    mean=[d['A'][0]*pred[0][0],pred[1][0]+d['offset'][0],d['sigma8'][0]*pred[2][0]]
    return dict(velocity=velocity,factor=factor,cs=cs,fit=best,mean=[m.tolist() for m in mean])

def mock_one(config,index):
    velocity,cs=config['velocity'],config['cs']
    seed=2026102700+(0 if velocity=='pressureless' else 100000)+int(config['factor']*10000)+index
    rng=np.random.default_rng(seed)
    baseline=Likelihood()
    obs=[np.array(m)+ch@rng.standard_normal(len(m)) for m,ch in zip(config['mean'],baseline.chols)]
    like=Likelihood(observations=obs)
    fixed=general_fit(like,cs,start=config['fit']['eps'],velocity=velocity)
    zero=general_fit(like,0.,start=fixed['eps'],velocity=velocity)
    trials=[general_fit(like,start=np.r_[fixed['eps'],max(cs,1e-9)/1e-5],velocity=velocity),
            general_fit(like,start=np.r_[zero['eps'],1e-6 if velocity=='pressureless' else 1e-7],velocity=velocity)]
    if not zero['success'] or not zero['valid']:
        for trial in trials:
            if trial['success'] and trial['valid']:
                zero=general_fit(like,0.,start=trial['eps'],velocity=velocity,maxiter=600)
                if zero['success'] and zero['valid']:break
    good=[f for f in trials+[zero,fixed] if f['valid'] and f['success']]
    retried=False
    if not fixed['success'] or not fixed['valid']:
        alternatives=sorted([f for f in trials+[zero] if f['valid'] and f['success']],key=lambda f:f['chi2'])
        for other in alternatives+[dict(eps=BASE_EPS)]:
            fixed=general_fit(like,cs,start=other['eps'],velocity=velocity,maxiter=600);retried=True
            if fixed['success'] and fixed['valid']:
                break
        good=[f for f in trials+[zero,fixed] if f['valid'] and f['success']]
    if not fixed['success'] or not fixed['valid'] or not good or not any(t['success'] and t['valid'] for t in trials):
        return dict(index=index,seed=seed,failed=True,fixed=fixed,trials=trials,zero=zero)
    best=min(good,key=lambda f:f['chi2'])
    # An independent high-sound-speed start on a predetermined 10% audit subset.
    high=None
    if index%10==0:
        high=general_fit(like,start=np.r_[fixed['eps'],10.],velocity=velocity,maxiter=500)
        if high['success'] and high['valid'] and high['chi2']<best['chi2']:
            best=high
    q=0. if best['cs']>cs else max(0.,fixed['chi2']-best['chi2'])
    return dict(index=index,seed=seed,failed=False,q=q,covered_2706=bool(q<=2.706),covered_3841=bool(q<=3.841),
                fixed_chi2=fixed['chi2'],best_chi2=best['chi2'],best_cs=best['cs'],
                failed_optimizer_starts=sum(not t['success'] or not t['valid'] for t in trials+[zero]),
                converged_free_starts=sum(t['success'] and t['valid'] for t in trials),zero_fit_converged=bool(zero['success'] and zero['valid']),
                free_start_spread=(max(t['chi2'] for t in trials if t['success'] and t['valid'])-min(t['chi2'] for t in trials if t['success'] and t['valid'])) if all(t['success'] and t['valid'] for t in trials) else None,
                high_start=high,retried=retried,eps_fixed=fixed['eps'],eps_best=best['eps'])

def wilson(k,n,z=1.95996398454):
    p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-radius),float(mid+radius)]

def summarize(path,config,records):
    good=[r for r in records if not r['failed']]
    n=len(good);k=sum(r['covered_2706'] for r in good)
    result=dict(config=config,completed=len(records),successful=n,failed=len(records)-n,
                covered=k,coverage=k/n if n else None,
                binomial_se=float(np.sqrt((k/n)*(1-k/n)/n)) if n else None,
                wilson95=wilson(k,n) if n else None,
                fraction_at_boundary=sum(r['best_cs']<1e-10 for r in good)/n if n else None,
                failed_optimizer_starts=sum(r['failed_optimizer_starts'] for r in good),
                high_start_improvements=sum(r['high_start'] is not None and r['high_start']['success'] and abs(r['best_cs']-r['high_start']['cs'])<1e-15 and r['best_cs']>1e-4 for r in good))
    path.with_suffix('.summary.json').write_text(json.dumps(result,indent=2))
    return result

def run(velocity,n=1000):
    configs=[truth(velocity,f) for f in (1.,.5,0.)]
    tasks=[];records={};paths={}
    for c in configs:
        f=c['factor'];path=RESULTS/f'coverage_general_{velocity}_{f:g}.jsonl';paths[f]=path
        records[f]=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        done={r['index'] for r in records[f]}
        for i in range(n if f else min(n,200)):
            if i not in done:tasks.append((c,i))
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(mock_one,c,i):(c,i) for c,i in tasks}
        for task in as_completed(futures):
            c,i=futures[task];f=c['factor'];r=task.result();records[f].append(r)
            with paths[f].open('a') as handle:handle.write(json.dumps(r)+'\n')
            if len(records[f])%50==0:
                report=summarize(paths[f],c,records[f]);print(velocity,f,report['successful'],report['coverage'],report['failed'],flush=True)
    for c in configs:
        print(json.dumps(summarize(paths[c['factor']],c,records[c['factor']]),indent=2),flush=True)

if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'pressureless',int(sys.argv[2]) if len(sys.argv)>2 else 1000)
