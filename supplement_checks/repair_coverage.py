"""Retry every failed Monte Carlo fit; retain original records for audit."""
from run_coverage import *
def run(velocity):
    for factor in (1.,.5,0.):
        path=RESULTS/f'coverage_general_{velocity}_{factor:g}.jsonl'
        config=truth(velocity,factor)
        records=[json.loads(line) for line in path.read_text().splitlines()]
        failed=[r for r in records if r['failed'] or r.get('converged_free_starts',1)==0 or ('converged_free_starts' not in r and r.get('failed_optimizer_starts',0)>=2)]
        backup=path.with_suffix('.before_repair.jsonl')
        if not backup.exists():backup.write_bytes(path.read_bytes())
        for bad in failed:
            new=mock_one(config,bad['index'])
            records[records.index(bad)]=new
            print(velocity,factor,bad['index'],'failed',new['failed'],flush=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        summarize(path,config,records)
if __name__=='__main__':run(sys.argv[1])
