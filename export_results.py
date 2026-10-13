"""Export new numerical results without assertions tied to the old manuscript."""
import csv
import json
from pathlib import Path
import platform
import numpy as np
import scipy
import emcee
import fit_joint as fj
from run_identity import source_identity
ROOT=Path(__file__).resolve().parent

def main():
    post=fj.load_result('n1_posterior','posterior','physical',1)
    chain=post['chain']
    result={'source_identity':source_identity(),'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,'emcee':emcee.__version__,
            'measurements':int(fj.NDATA),'sound_speed_nodes':1,
            'posterior_samples':len(chain),'best_sampled_chi2':float(post['best_sampled_chi2']),
            'cs2_upper95':float(np.percentile(chain[:,8],95)),
            'sigma8_ref_16_50_84':np.percentile(chain[:,9],[16,50,84]).tolist(),
            'posterior_percentiles_16_50_84':np.percentile(chain,[16,50,84],axis=0).tolist()}
    for name in ('extensions','assumptions','checks','crosscheck_extensions','crosscheck_assumptions'):
        result[name]=json.loads((ROOT/(name+'.json')).read_text())
    result['numerical_validation']=json.loads((ROOT/'figure_data/numerical_validation.json').read_text())
    (ROOT/'new_results.json').write_text(json.dumps(result,indent=2))
    def flatten(value,prefix=''):
        if isinstance(value,dict):
            for key,item in value.items(): yield from flatten(item,prefix+'/'+key)
        elif isinstance(value,list):
            for i,item in enumerate(value): yield from flatten(item,prefix+'/'+str(i))
        else: yield prefix,value
    with (ROOT/'new_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['quantity','value']);writer.writerows(flatten(result))
    print('Written new_results.json and new_results.csv. Inspect logs/converged.log and validation diagnostics.')

if __name__=='__main__': main()
