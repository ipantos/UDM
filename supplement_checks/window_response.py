"""DESI-window linear template injection/recovery diagnostic.

Uses actual released windows/covariances, not actual galaxy spectra as UDM data.
Kaiser density+velocity spectra with a fixed primordial shape are synthetic
injections. This deliberately does not claim a nonlinear ShapeFit validation.
"""
from run_profiles import *
import h5py, hashlib
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.special import eval_legendre

def template():
    path=RESULTS/'fiducial_linear_template.npz'
    if path.exists():
        d=np.load(path);return CubicSpline(np.log(d['k']),np.log(d['punit']))
    # CAMB's Fortran file loader can fail on non-ASCII paths. UDM_CAMB_PACKAGE_ROOT
    # may point to the parent folder of an unchanged camb package.
    support=os.environ.get('UDM_CAMB_PACKAGE_ROOT')
    if support:sys.path.insert(0,support)
    import camb
    p=camb.CAMBparams();p.set_cosmology(H0=67.36,ombh2=.02237,omch2=.1200,mnu=.06,nnu=3.046)
    p.InitPower.set_params(As=2.083e-9,ns=.9649)
    p.WantCls=False;p.set_matter_power(redshifts=[0.],kmax=3.)
    result=camb.get_results(p)
    interp=result.get_matter_power_interpolator(nonlinear=False,var1='delta_nonu',var2='delta_nonu',hubble_units=True,k_hunit=True)
    k=np.geomspace(1e-4,2.,3000);power=interp.P(0.,k)
    # Integrate the same cold+baryon spectrum for sigma_8, avoiding total-matter
    # vs cb normalization differences in the massive-neutrino fiducial model.
    sig2=float(result.get_sigmaR(8.,z_indices=0,var1='delta_nonu',var2='delta_nonu',hubble_units=True))**2
    np.savez_compressed(path,k=k,punit=power/sig2,sigma8_cb=np.sqrt(sig2),camb_version=camb.__version__)
    return CubicSpline(np.log(k),np.log(power/sig2))

def response(model,factor,z):
    src=json.loads((RESULTS/f'profile_{model}_all.json').read_text());cs=src['first_crossing_cs']*factor
    like=Likelihood()
    fit=general_fit(like,cs) if model=='general' else minimal_fit(like,cs)
    solver=Solver(1600,z_out=np.array([z]),k_out=np.array([0.]))
    solver.k_all=np.r_[0.,np.geomspace(.0001,.7,600)]
    if model=='general':
        inv,q,ip=solver.spline(fit['eps']);c=cs;ob=OB*(1+solver.z)**3*ip**2;rate=1.
        pred=general_prediction(fit['eps'],cs,Solver(1600))
        inv0,q0,ip0=inv,q,ip;c0=0.;ob0=ob
    else:
        inv,q,ip,c,ob=asm.background('minimal',solver,fit['omega'],cs,asm.FB);rate=growing(fit['omega'])
        pred=minimal_prediction(fit['omega'],cs,Solver(1600))
        # Isolate scale dependence at the SAME background with pressure removed.
        inv0,q0,ip0,c0,ob0=inv,q,ip,0.,ob
    out=fields(solver,q,ip,c,ob,rate);zero=fields(solver,q0,ip0,c0,ob0,rate)
    norm=out[0,0,-1,0];detail=like.profile(pred,True);s8=detail['sigma8'][0]
    dd=out[2,0,0]/zero[2,0,0,0];vv=out[3,0,0]/zero[3,0,0,0]
    bamp=s8*zero[2,0,0,0]/norm;famp=s8*zero[3,0,0,0]/norm
    chi=float(np.interp(z,G.ZG,ext.distances(inv)[0]));ih=float(np.interp(z,G.ZG,inv[0]))
    fid=Solver(1600);finv=fid.lcdm(.1430/.6736**2)[0]
    fchi=float(np.interp(z,G.ZG,ext.distances(finv)[0]));fih=float(np.interp(z,G.ZG,finv[0]))
    # Volume-preserving AP warp retains H0-independent DH/DM anisotropy.
    qap=(ih/chi)/(fih/fchi)
    return dict(cs=cs,k=solver.k_all,rd=dd,rv=vv,B=bamp,F=famp,qap=qap)

def run():
    logp=template();mu,wt=np.polynomial.legendre.leggauss(64)
    leg=np.array([(2*ell+1)/2*wt*eval_legendre(ell,mu) for ell in (0,2,4)])
    rows=[];files=[]
    for path in sorted((ROOT/'desi_data').glob('*.h5')):
        with h5py.File(path) as h:
            k=h['window/theory/spectrum/0/k'][:];W=h['window/value'][:];C=h['covariance/value'][:]
            z=float(h['observable/spectrum'].attrs['zeff'])
            outk=h['observable/spectrum/0/k'][:]
        L=np.linalg.cholesky(C);whiten=np.linalg.inv(L)
        files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),z=z))
        def spectrum(B,F,m,qap,rd=None,rv=None):
            aperp=qap**(-1/3);apar=qap**(2/3)
            scale=np.sqrt((1-mu**2)/aperp**2+mu**2/apar**2)
            kt=k[:,None]*scale;mut=mu/apar/scale
            shape=np.exp(logp(np.log(kt))+(m/.6)*np.tanh(.6*np.log(kt/.03)))
            d=1. if rd is None else np.interp(kt,rd[0],rd[1])
            v=1. if rv is None else np.interp(kt,rv[0],rv[1])
            power=(B*d+F*v*mut**2)**2*shape/(aperp**2*apar)
            return W@np.concatenate([power@l for l in leg])
        for model in ('general','minimal'):
            for factor in (0.,.5,1.):
                res=response(model,factor,z)
                for bias in (1.5,2.,2.5):
                    B,F=bias*res['B'],res['F'];keff=np.sqrt(.6*(.2**5-.02**5)/(.2**3-.02**3))
                    fk=F*np.interp(keff,res['k'],res['rv'])
                    for geometry in ('identity','ap_warp'):
                        qtrue=1. if geometry=='identity' else res['qap']
                        injection=spectrum(B,F,0.,qtrue,(res['k'],res['rd']),(res['k'],res['rv']))
                        fits={}
                        for allow_ap in (False,True):
                            def residual(x):
                                return whiten@(spectrum(x[0],x[1],x[2],x[3] if allow_ap else 1.)-injection)
                            x0=[B,F,0.,qtrue] if allow_ap else [B,F,0.]
                            bounds=([0.,0.,-.8,.7],[5.,2.,.8,1.3]) if allow_ap else ([0.,0.,-.8],[5.,2.,.8])
                            fit=least_squares(residual,x0,bounds=bounds,ftol=1e-10,xtol=1e-10,gtol=1e-9,max_nfev=250)
                            pcov=np.linalg.pinv(fit.jac.T@fit.jac);sig=np.sqrt(pcov[1,1])
                            fits['free_ap' if allow_ap else 'fixed_ap']=dict(F=float(fit.x[1]),sigma_F=float(sig),
                                F_minus_keff=float(fit.x[1]-fk),bias_in_sigma=float((fit.x[1]-fk)/sig),
                                delta_chi2=float(fit.fun@fit.fun),parameters=fit.x.tolist(),success=bool(fit.success))
                        item=dict(sample=path.name,z=z,model=model,factor=factor,cs=res['cs'],galaxy_bias=bias,
                                  geometry=geometry,qap_true=qtrue,F_zero=F,F_keff=fk,**fits)
                        rows.append(item)
                print('window',path.name,model,factor,flush=True)
                (RESULTS/'window_response.json').write_text(json.dumps(dict(complete=False,files=files,rows=rows),indent=2))
    (RESULTS/'window_response.json').write_text(json.dumps(dict(complete=True,files=files,rows=rows,
        scope='Noiseless linear Kaiser injections; density and pressureless velocity evolved together; fixed fiducial cb spectral shape. Bias and ShapeFit-like m refitted. Exact released DESI DR1 rotated windows and systematics covariance. No nonlinear EFT terms, survey estimator rerun, halo mocks, or replacement cosmological likelihood.'),indent=2))

if __name__=='__main__':run()
