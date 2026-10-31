"""Fresh, reoptimized pressureless exclusion profiles; checkpoint every point."""
from audit_engine import *
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1
from concurrent.futures import ProcessPoolExecutor, as_completed

def growing(om):
    r=(1-np.asarray(om))/np.asarray(om)/(1+ext.fj.Z_INI)**3
    return 1-(6/11)*r*hyp2f1(4/3,2,17/6,-r)/hyp2f1(1/3,1,11/6,-r)

def minimal_prediction(om,cs,solver,velocity='pressureless'):
    inv,q,ip,c,ob=asm.background('minimal',solver,om,cs,asm.FB if velocity=='pressureless' else 0.)
    shape=asm.shapes(solver,fields(solver,q,ip,c,ob,growing(om)))[velocity]
    return design(inv,shape)

def minimal_fit(like,cs,steps=400,velocity='pressureless'):
    solver=Solver(steps,velocity)
    def objective(om):
        return float(like.profile(minimal_prediction(om,cs,solver,velocity))[0])
    start=time.perf_counter()
    fit=minimize_scalar(objective,bounds=(.15,.6),method='bounded',options={'xatol':1e-8})
    return dict(omega=float(fit.x),cs=float(cs),chi2=float(fit.fun),success=bool(fit.success),
                valid=bool(fit.success),seconds=time.perf_counter()-start)

def anchors():
    om=.303
    eps=-1.5*om*(1+G.Z_NODES)**3/(om*(1+G.Z_NODES)**3+1-om)
    return [BASE_EPS,np.array(STORED_EXT['bestfit']['with_V_nonnegative']['eps_nodes']),eps]

def run_case(model,subset):
    like=Likelihood(ROWS[subset]);sol=Solver(1600)
    path=RESULTS/f'profile_{model}_{subset}.json'
    data=json.loads(path.read_text()) if path.exists() else dict(model=model,subset=subset,rows=ROWS[subset].tolist(),complete=False,points=[])
    # Dense onset, logarithmic extension through the repository's tested domain.
    ug=np.unique(np.r_[-12.,-10.,-9.,np.arange(-8.,-3.99,.1),np.arange(-3.8,-1.99,.2)]) if model=='general' else np.unique(np.r_[-12.,-10.,-9.,np.arange(-8.,-3.99,.15),np.arange(-4.,-1.99,.05),np.arange(-1.8,-.999,.2)])
    for u in ug:
        if any(abs(r['u']-u)<1e-8 for r in data['points']):
            continue
        cs=10.**u
        if model=='general':
            starts=anchors()
            if data['points']:
                starts=[np.array(data['points'][-1]['eps'])]+starts
            trials=[general_fit(like,cs,start=ext.moved_inside(st,cs)) for st in starts]
            good=[r for r in trials if r['valid'] and r['success']]
            if not good:
                data['failure']=dict(u=float(u),trials=trials);path.write_text(json.dumps(data,indent=2));raise RuntimeError(f'No converged fit {model} {subset} {u}')
            best=min(good,key=lambda r:r['chi2'])
            prediction=general_prediction(best['eps'],cs,sol)
            best['starts']=len(trials);best['converged_starts']=len(good)
            best['start_spread']=max(r['chi2'] for r in good)-best['chi2']
        else:
            best=minimal_fit(like,cs)
            prediction=minimal_prediction(best['omega'],cs,sol)
        detail=like.profile(prediction,True)
        best['fine_chi2']=float(detail['chi2'][0]);best['step_chi2_difference']=best['fine_chi2']-best['chi2']
        best['A']=float(detail['A'][0]);best['sigma8']=float(detail['sigma8'][0]);best['offset']=float(detail['offset'][0])
        best['u']=float(u)
        data['points'].append(best);path.write_text(json.dumps(data,indent=2))
        print(model,subset,f'u={u:.2f} chi2={best["fine_chi2"]:.6f}',flush=True)
    pts=sorted(data['points'],key=lambda r:r['u']);data['points']=pts
    uu=np.array([r['u'] for r in pts]);cc=np.array([r['fine_chi2'] for r in pts])
    imin=int(cc.argmin())
    if model=='general':
        candidates=[general_fit(like,start=np.r_[p['eps'],p['cs']/1e-5]) for p in (pts[imin],pts[max(0,imin-2)],pts[min(len(pts)-1,imin+2)])]
        best=min([r for r in candidates if r['valid'] and r['success']],key=lambda r:r['chi2'])
        best['fine_chi2']=float(like.profile(general_prediction(best['eps'],best['cs'],sol))[0])
    else:
        fit=minimize_scalar(lambda u:minimal_fit(like,10.**u)['chi2'],bounds=(uu[max(0,imin-1)],uu[min(len(uu)-1,imin+1)]),method='bounded',options={'xatol':1e-6})
        best=minimal_fit(like,10.**fit.x)
        best['fine_chi2']=float(like.profile(minimal_prediction(best['omega'],best['cs'],sol))[0])
    reference=min(best['fine_chi2'],float(cc.min()));delta=cc-reference
    crossing=ext.crossing(uu[imin:],delta[imin:],2.706)
    data.update(best=best,chi2_minimum=reference,first_crossing_cs=None if crossing is None else float(10.**crossing),
                minimum_delta_beyond_crossing=None if crossing is None else float(delta[uu>crossing].min()),
                delta_at_upper_end=float(delta[-1]),complete=True)
    path.write_text(json.dumps(data,indent=2))
    return {k:data[k] for k in ('model','subset','chi2_minimum','first_crossing_cs','minimum_delta_beyond_crossing','delta_at_upper_end')}

if __name__=='__main__':
    if len(sys.argv)>1:
        print(run_case(*sys.argv[1:3]))
    else:
        with ProcessPoolExecutor(max_workers=6) as pool:
            tasks=[pool.submit(run_case,m,s) for m in ('general','minimal') for s in ROWS]
            for t in as_completed(tasks):
                print('COMPLETED',t.result(),flush=True)
