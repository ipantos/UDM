"""Driver of the supplementary checks; imports the UDM code of the parent folder.

The compiled RK4 kernel implements assumptions.two_fluid verbatim algebraically.
No fast-math, altered initial conditions, or altered data are used.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
from pathlib import Path
import sys, json, time
import numpy as np
from numba import njit
from scipy.optimize import minimize, LinearConstraint

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
import assumptions as asm
import extensions as ext
G = ext.G
RESULTS = Path(os.environ.get('UDM_AUDIT_RESULTS',str(ROOT/'results'))).resolve()
RESULTS.mkdir(exist_ok=True)
STORED = json.loads((REPO/'assumptions.json').read_text())
STORED_EXT = json.loads((REPO/'extensions.json').read_text())
OB = asm.FB * STORED_EXT['lcdm']['same_initial_rate']['omega_m']
BASE = STORED['tracer_general']['pressureless component']
BASE_EPS = np.array(BASE['eps_nodes_at_minimum'])
ROWS = {'all': np.arange(17),
        'no_lowz': np.flatnonzero(~np.isin(np.round(ext.GZ, 3), [.07,.18,.38])),
        'no_desi': np.flatnonzero(~np.isin(np.round(ext.GZ, 3), [.295,.51,.706,.919,1.317,1.491]))}

@njit(cache=True)
def rates(y,v,d,b,q,ob,w,p):
    s = 1.5*((1-ob)*y+ob*d)
    return 3*(w-1)*y+w*v, -(1-q)*v+s-p*y/w, b, -(1-q)*b+s

@njit(cache=True)
def kernel(times, k2, q, ob, w, pk, initial, output_indices):
    n, nt = q.shape[0], len(times)
    nk, nz = len(k2), len(output_indices)
    out = np.empty((6,n,nz,nk))
    for a in range(n):
        for k in range(nk):
            y, v, d, b = w[a,0], initial[a], 1., initial[a]
            for i in range(nt):
                for j in range(nz):
                    if output_indices[j] == i:
                        out[0,a,j,k] = y/w[a,i]
                        out[1,a,j,k] = v
                        out[2,a,j,k] = d
                        out[3,a,j,k] = b
                        out[4,a,j,k] = w[a,i]
                        out[5,a,j,k] = ob[a,i]
                if i == nt-1:
                    break
                dt, m = times[i+1]-times[i], nt+i
                a1,a2,a3,a4 = rates(y,v,d,b,q[a,i],ob[a,i],w[a,i],pk[a,i]*k2[k])
                b1,b2,b3,b4 = rates(y+dt*a1/2,v+dt*a2/2,d+dt*a3/2,b+dt*a4/2,q[a,m],ob[a,m],w[a,m],pk[a,m]*k2[k])
                c1,c2,c3,c4 = rates(y+dt*b1/2,v+dt*b2/2,d+dt*b3/2,b+dt*b4/2,q[a,m],ob[a,m],w[a,m],pk[a,m]*k2[k])
                d1,d2,d3,d4 = rates(y+dt*c1,v+dt*c2,d+dt*c3,b+dt*c4,q[a,i+1],ob[a,i+1],w[a,i+1],pk[a,i+1]*k2[k])
                y += dt*(a1+2*b1+2*c1+d1)/6
                v += dt*(a2+2*b2+2*c2+d2)/6
                d += dt*(a3+2*b3+2*c3+d3)/6
                b += dt*(a4+2*b4+2*c4+d4)/6
    return out

def fields(solver,q,inv,cs,ob,initial=1.):
    w = 1+((2*q-1)/3)/(1-ob)
    pk = np.broadcast_to(cs,q.shape)*(G.C_OVER_H0*inv*np.exp(solver.x))**2
    indices = np.searchsorted(solver.times, -np.log1p(np.r_[solver.z_out,ext.fj.Z_NORM]))
    return kernel(solver.times, solver.k_all**2, np.ascontiguousarray(q),
                  np.ascontiguousarray(ob), w, pk,
                  np.broadcast_to(initial,(len(q),)).copy(), indices)

class Solver(ext.Solver):
    def __init__(self,steps=400,velocity='pressureless',**kw):
        super().__init__(steps,**kw)
        self.velocity=velocity
        self.ob=OB if velocity=='pressureless' else 0.
    def shape(self,q,inv,cs,initial=1.):
        ob = self.ob*(1+self.z)**3*inv**2
        return asm.shapes(self,fields(self,q,inv,cs,ob,initial))[self.velocity]

def design(inv,shape):
    dm = ext.distances(inv)
    bao = np.empty((len(inv),13))
    bao[:,:12:2] = G.interpolate_rows(dm,ext.ANISO[:,0])
    bao[:,1:12:2] = G.interpolate_rows(inv,ext.ANISO[:,0])
    bao[:,12] = np.cbrt(ext.BGS[0]*G.interpolate_rows(dm,[ext.BGS[0]])[:,0]**2*G.interpolate_rows(inv,[ext.BGS[0]])[:,0])
    sn = 5*np.log10((1+ext.SNZ)*G.interpolate_rows(dm,ext.SNZ))
    return bao,sn,shape

BAOY = np.r_[np.column_stack([ext.ANISO[:,1],ext.ANISO[:,3]]).ravel(),ext.BGS[1]]
BAOC = np.zeros((13,13))
for j,ci in enumerate(ext.BAO_CINV):
    BAOC[2*j:2*j+2,2*j:2*j+2] = np.linalg.inv(ci)
BAOC[-1,-1] = ext.BGS[2]**2

class Likelihood:
    def __init__(self,rows=None,observations=None):
        self.rows = np.arange(17) if rows is None else np.array(rows)
        self.covs = [BAOC,ext.OBS['sn_cov'],ext.GCOV[np.ix_(self.rows,self.rows)]]
        self.chols = [np.linalg.cholesky(c) for c in self.covs]
        self.whiten = [np.linalg.inv(ch).T for ch in self.chols]
        self.obs = [BAOY,ext.SNY,ext.GY[self.rows]] if observations is None else observations
        self.y = [a@w for a,w in zip(self.obs,self.whiten)]
        self.one = np.ones(22)@self.whiten[1]
    def profile(self,pred,details=False):
        tb,ts,tg = pred
        b = tb@self.whiten[0]
        g = tg[:,self.rows]@self.whiten[2]
        s = ts@self.whiten[1]
        A = (b@self.y[0])/np.sum(b*b,axis=1)
        amp = (g@self.y[2])/np.sum(g*g,axis=1)
        offset = (self.y[1]-s)@self.one/(self.one@self.one)
        pieces = np.array([np.sum((self.y[0]-A[:,None]*b)**2,axis=1),
                           np.sum((self.y[1]-s-offset[:,None]*self.one)**2,axis=1),
                           np.sum((self.y[2]-amp[:,None]*g)**2,axis=1)])
        val = pieces.sum(axis=0)
        if details:
            return dict(chi2=val,parts=pieces,A=A,sigma8=amp,offset=offset)
        return val
    def simulate(self,pred,seed):
        p = self.profile(pred,True)
        mean = [p['A'][0]*pred[0][0],pred[1][0]+p['offset'][0],p['sigma8'][0]*pred[2][0,self.rows]]
        rng = np.random.default_rng(seed)
        return [m+ch@rng.standard_normal(len(m)) for m,ch in zip(mean,self.chols)]

def general_prediction(eps,cs,solver):
    inv,q,ip = solver.spline(eps)
    return design(inv,solver.shape(q,ip,np.reshape(cs,(-1,1))))

def scalar_margin(eps,ob=OB):
    return -G.SMAT@eps-1.5*ob*np.exp(3*G.XG+2*(G.IMAT@eps))

def general_fit(like,cs=None,start=None,steps=400,maxiter=300,velocity='pressureless'):
    solver = Solver(steps,velocity)
    ob=solver.ob
    free = cs is None
    x0 = np.r_[BASE_EPS,BASE['c_s2_at_minimum']/1e-5] if start is None and free else (BASE_EPS.copy() if start is None else np.array(start).copy())
    if free and len(x0)==7:
        x0 = np.r_[x0,.3]
    if not free:
        x0 = x0[:7]
    n = len(x0)
    def vg(x):
        h = 1e-5
        batch = np.repeat(x[None,:],2*n+1,axis=0)
        batch[1+np.arange(n),np.arange(n)] += h
        batch[1+n+np.arange(n),np.arange(n)] -= h
        c = np.maximum(batch[:,7]*1e-5,0) if free else cs
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            vals = like.profile(general_prediction(batch[:,:7],c,solver))
        if not np.all(np.isfinite(vals)):
            return 1e30,np.zeros(n)
        grad=(vals[1:1+n]-vals[1+n:])/(2*h)
        if free and x[7]<h:
            grad[7]=(vals[8]-vals[0])/h
        return vals[0],grad
    # Same dense V>=0 constraint as the repository, with explicit positive scalar enthalpy.
    mat = np.zeros((len(G.XG),n));mat[:,:7]=G.SMAT
    if free:
        mat[:,7] = -1.5e-5
    lower = -1.5+1e-9 if free else -1.5*(1-cs)+1e-9
    constraints = [LinearConstraint(mat,lower,np.inf)]
    upper=mat.copy()
    if free:
        upper[:,7]=0.
    constraints.append(LinearConstraint(upper,-np.inf,-1e-9))
    def enthalpy(x):
        return scalar_margin(x[:7],ob)-1e-8
    def enthalpy_jac(x):
        out = np.zeros((len(G.XG),n))
        out[:,:7] = -G.SMAT-3*ob*np.exp(3*G.XG+2*(G.IMAT@x[:7]))[:,None]*G.IMAT
        return out
    constraints.append({'type':'ineq','fun':enthalpy,'jac':enthalpy_jac})
    bounds=[(-10+1e-6,-1e-6)]*7+([(0,1000)] if free else [])
    started=time.perf_counter()
    fit=minimize(vg,x0,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                 options={'ftol':2e-8,'maxiter':maxiter})
    attempts=1
    while not fit.success and attempts<4:
        retry=fit.x.copy();rc=retry[7]*1e-5 if free else cs
        retry[:7]=ext.moved_inside(retry[:7],rc)
        # Starting a failed line search a little inside the linear boundary
        # removes numerical degeneracy without changing the objective/domain.
        retry[:7]=.9999*retry[:7]+.0001*(-.75)
        newer=minimize(vg,retry,jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                       options={'ftol':2e-8,'maxiter':maxiter})
        attempts+=1
        fit=newer
    c = fit.x[7]*1e-5 if free else cs
    projection=0.
    grid=G.SMAT@fit.x[:7]
    violation=max(float(np.max(grid)),float(np.max(-1.5*(1-c)-grid)),float(-np.min(scalar_margin(fit.x[:7],ob))),0.)
    if 0.<violation<=1e-6:
        original=fit.x[:7].copy()
        for alpha in 10.**np.arange(-10.,-3.9,.5):
            candidate=(1-alpha)*original+alpha*(-.75)
            cg=G.SMAT@candidate
            if np.max(cg)<0 and np.min(cg)>=-1.5*(1-c) and np.min(scalar_margin(candidate,ob))>0:
                fit.x[:7]=candidate;fit.fun=vg(fit.x)[0];projection=float(alpha);break
    grid=G.SMAT@fit.x[:7]
    valid=bool(np.min(scalar_margin(fit.x[:7],ob))>0 and np.max(grid)<1e-7 and np.min(grid)>-1.5*(1-c)-1e-7 and np.isfinite(fit.fun))
    return dict(eps=fit.x[:7].tolist(),cs=float(c),chi2=float(fit.fun),success=bool(fit.success),
                valid=valid,status=int(fit.status),message=str(fit.message),iterations=int(fit.nit),attempts=attempts,interior_projection=projection,
                seconds=time.perf_counter()-started,scalar_margin=float(np.min(scalar_margin(fit.x[:7],ob))))

def validate():
    report={'commit':'24a45240ab122c1c202262c7baf3b05e185547f3','omega_b':OB,'cases':[]}
    for steps in (400,1600):
        solver=Solver(steps)
        for cs in (0.,BASE['c_s2_at_minimum'],8.8e-6,.01):
            inv,q,ip=solver.spline(BASE_EPS)
            ob=OB*(1+solver.z)**3*ip**2
            ref=asm.two_fluid(solver,q,ip,cs,ob)
            start=time.perf_counter();got=fields(solver,q,ip,cs,ob);elapsed=time.perf_counter()-start
            pred=general_prediction(BASE_EPS,cs,solver)
            chis=Likelihood().profile(pred,True)
            old=ext.linear_profile(inv,asm.shapes(solver,ref)['pressureless'])
            record=dict(steps=steps,cs=cs,max_field_abs_diff=float(np.max(np.abs(ref-got))),
                        chi2=float(chis['chi2'][0]),chi2_difference=float(chis['chi2'][0]-old['chi2'][0]),seconds=elapsed)
            report['cases'].append(record);print(record,flush=True)
    report['stored_chi2']=BASE['chi2_minimum']
    report['fit']=general_fit(Likelihood())
    print(report['fit'],flush=True)
    (RESULTS/'validation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    validate()
