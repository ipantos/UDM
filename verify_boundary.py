"""Optional regression at the former tracer-profile failure; two single-start fits.
Run with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for predictable timings.
This does not certify a global minimum or complete the scientific profile.
"""
import sys,json,numpy as np,time
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
import extensions as e, assumptions as a
eps=np.array([-0.792979426877, -0.831816166908, -0.991225295815, -1.25549049871, -1.36095220406, -1.44681401939, -1.1958938085e-09])
e.FAST=a.TracerSolver(400,0.);e.FINE=a.TracerSolver(1600,0.)
results=[]
for u in [-4.,-3.5]:
    start=time.time();x,c,chi=e.minimise_nodes(e.moved_inside(eps,10**u),10**u,rows=np.arange(len(e.GZ)))
    g=e.G.SMAT@x
    out=dict(u=u,chi2_fast=chi,feasible=e.feasible(x,c),lower_margin=float(min(g+1.5*(1-c))),upper_margin=float(-max(g)),seconds=time.time()-start)
    results.append(out);print(out,flush=True)
    assert out['feasible'];eps=x
(ROOT/'provenance/boundary_regression.json').write_text(json.dumps({'scope':'Two single-start tracer fits, not a complete profile or global-minimum certification','input':'corrected Qin data','results':results},indent=2))
