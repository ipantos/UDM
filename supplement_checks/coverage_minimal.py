"""Grid coverage with all nuisance/background parameters profiled in each mock."""
from run_coverage import wilson
from run_profiles import *

def run(velocity,n=4000):
    limit=.0008006136242943255 if velocity=='pressureless' else 4.494602034728707e-6
    csgrid=np.unique(np.r_[0.,10.**np.r_[-12.,-10.,-9.,np.arange(-8.,-0.999,.05)],limit/2,limit])
    omega=np.arange(.20,.45001,.001)
    cache=RESULTS/f'minimal_coverage_design_{velocity}.npz'
    if cache.exists():
        d=np.load(cache);tb,ts,tg=[d[k] for k in ('bao','sn','growth')]
    else:
        cc,oo=[x.ravel() for x in np.meshgrid(csgrid,omega,indexing='ij')]
        solver=Solver(400,velocity);parts=[]
        for first in range(0,len(cc),500):
            parts.append(minimal_prediction(oo[first:first+500],cc[first:first+500],solver,velocity))
        tb,ts,tg=[np.concatenate([p[j] for p in parts]) for j in range(3)]
        np.savez_compressed(cache,bao=tb,sn=ts,growth=tg,cs=csgrid,omega=omega)
    like=Likelihood();wb,ws,wg=like.whiten
    b=tb@wb;g=tg@wg;s=ts@ws
    s-=((s@like.one)/(like.one@like.one))[:,None]*like.one
    bn=np.sum(b*b,axis=1);gn=np.sum(g*g,axis=1);sn=np.sum(s*s,axis=1)
    def profiles(obs):
        yb,ys,yg=[np.atleast_2d(o)@w for o,w in zip(obs,like.whiten)]
        ys-=((ys@like.one)/(like.one@like.one))[:,None]*like.one
        out=np.sum(yb*yb,axis=1)[:,None]-(yb@b.T)**2/bn
        out+=np.sum(ys*ys,axis=1)[:,None]+sn-2*(ys@s.T)
        out+=np.sum(yg*yg,axis=1)[:,None]-(yg@g.T)**2/gn
        return out.reshape(-1,len(csgrid),len(omega)).min(axis=2)
    cases=[]
    for j,factor in enumerate((1.,.5,0.)):
        cs=limit*factor;iu=int(np.argmin(abs(csgrid-cs)))
        fixed=like.profile((tb[iu*len(omega):(iu+1)*len(omega)],ts[iu*len(omega):(iu+1)*len(omega)],tg[iu*len(omega):(iu+1)*len(omega)]))
        io=int(fixed.argmin());idx=iu*len(omega)+io
        pred=(tb[idx:idx+1],ts[idx:idx+1],tg[idx:idx+1]);detail=like.profile(pred,True)
        mean=[detail['A'][0]*pred[0][0],pred[1][0]+detail['offset'][0],detail['sigma8'][0]*pred[2][0]]
        rng=np.random.default_rng(2026102800+(100000 if velocity=='scalar' else 0)+j*10000)
        qs=[];bestcs=[];miss_first=0
        for first in range(0,n,64):
            count=min(64,n-first)
            obs=[m+rng.standard_normal((count,len(m)))@ch.T for m,ch in zip(mean,like.chols)]
            p=profiles(obs);ib=p.argmin(axis=1);minimum=p[np.arange(count),ib]
            q=np.where(csgrid[ib]>cs,0.,p[:,iu]-minimum)
            qs.extend(q.tolist());bestcs.extend(csgrid[ib].tolist())
            # Also check the literal first-crossing convention, which can differ
            # from a one-sided test if an allowed region re-enters before truth.
            for pp,best in zip(p,ib):
                crossing=ext.crossing(csgrid[best:],(pp-pp[best])[best:],2.706)
                miss_first+=int(crossing is not None and crossing<cs)
        q=np.array(qs);k=int(np.sum(q<=2.706))
        cases.append(dict(factor=factor,true_cs=cs,true_omega=float(omega[io]),simulations=n,covered=k,coverage=k/n,
                          binomial_se=float(np.sqrt((k/n)*(1-k/n)/n)),wilson95=wilson(k,n),
                          first_crossing_coverage=1-miss_first/n,fraction_at_boundary=float(np.mean(np.array(bestcs)<1e-10)),
                          q95=float(np.quantile(q,.95))))
        np.savez_compressed(RESULTS/f'coverage_minimal_{velocity}_{factor:g}.npz',q=q,bestcs=bestcs)
        print(velocity,cases[-1],flush=True)
    report=dict(velocity=velocity,omega_grid=[float(omega[0]),float(omega[-1]),.001],cs_grid=csgrid.tolist(),cases=cases)
    (RESULTS/f'coverage_minimal_{velocity}.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else 'pressureless')
