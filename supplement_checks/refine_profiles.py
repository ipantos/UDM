from run_profiles import *
from scipy.optimize import brentq
def run():
    results=[]
    for model in ('general','minimal'):
        for subset in ROWS:
            data=json.loads((RESULTS/f'profile_{model}_{subset}.json').read_text());like=Likelihood(ROWS[subset])
            c=data['first_crossing_cs'];lo=max(p['cs'] for p in data['points'] if p['cs']<c);hi=min(p['cs'] for p in data['points'] if p['cs']>c)
            guesses=sorted(data['points'],key=lambda p:abs(np.log10(p['cs']/c)))
            calls=[]
            def objective(logc):
                cc=10.**logc
                if model=='general':
                    trials=[general_fit(like,cc,start=g['eps']) for g in guesses[:2]]
                    fit=min([f for f in trials if f['success'] and f['valid']],key=lambda x:x['chi2'])
                    pred=general_prediction(fit['eps'],cc,Solver(1600))
                else:
                    fit=minimal_fit(like,cc);pred=minimal_prediction(fit['omega'],cc,Solver(1600))
                fit['chi2_fine']=float(like.profile(pred)[0]);calls.append(fit)
                return fit['chi2_fine']-data['chi2_minimum']-2.706
            root=brentq(objective,np.log10(lo),np.log10(hi),xtol=1e-7)
            objective(root);fit=calls[-1]
            if model=='general':
                pred=general_prediction(fit['eps'],fit['cs'],Solver(3200))
            else:
                pred=minimal_prediction(fit['omega'],fit['cs'],Solver(3200))
            r=dict(model=model,subset=subset,limit=10.**root,fit=fit,
                   delta_chi2_3200=float(like.profile(pred)[0]-data['chi2_minimum']),calls=calls)
            results.append(r);print(model,subset,r['limit'],r['delta_chi2_3200'],flush=True)
    (RESULTS/'refined_limits.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':run()
