"""Fresh numerical run; restart this command to resume completed stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_identity import source_identity

ROOT = Path(__file__).resolve().parent

def commands():
    jobs = [('verify', ['verify_package.py'])]
    jobs += [(s, ['fit_joint.py', s]) for s in ('grid', 'start', 'nodes')]
    jobs += [('main_chain', None)]
    jobs += [(s, ['fit_joint.py', s]) for s in ('converged', 'summary', 'validate')]
    jobs += [('figures_data', ['figures.py', '--data-only']), ('checks', ['checks.py'])]
    jobs += [('ext_'+s, ['extensions.py', s]) for s in
             ('lcdm', 'bestfit', 'profile', 'leaveout', 'grids', 'coverage', 'tied')]
    jobs += [('chain_'+s, None) for s in ('constant', 'constant_any_V', 'tied')]
    jobs += [('ext_'+s, ['extensions.py', s]) for s in ('chains', 'posterior', 'figure', 'early')]
    jobs += [('asm_'+s, ['assumptions.py', s]) for s in
             ('tracer', 'tracer_checks', 'scales', 'window', 'tracer_general', 'figure')]
    jobs += [('crosscheck_extensions', ['crosscheck/crosscheck.py', 'udm_data.txt', 'extensions.json',
                                      'figure_data', 'crosscheck_extensions.json']),
             ('crosscheck_assumptions', ['crosscheck/crosscheck_assumptions.py', 'udm_data.txt',
                                       'assumptions.json', 'figure_data', 'crosscheck_assumptions.json']),
             ('export_results', ['export_results.py'])]
    return jobs

def launch(name, args):
    print('\nRUN:', sys.executable, *args, flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    with (ROOT/'logs'/f'{name}.log').open('a', encoding='utf-8') as log:
        log.write(f'\nInvocation {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: {args!r}\n')
        log.flush()
        process = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            result = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if result:
        raise SystemExit(f'{name} failed (exit {result}); see logs/{name}.log. No completion flag was written.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='print stages without running')
    parser.add_argument('--figures', action='store_true', help='also compile the five paper figures with pdflatex')
    args = parser.parse_args()
    jobs = commands()
    if args.figures:
        jobs += [('paper_figures', ['paper_scripts/paper_figures.py', '.', 'paper_output'])]
    if args.plan:
        for number, (name, cmd) in enumerate(jobs, 1):
            print(f'{number:2d}. {name}: ' + (' '.join(cmd) if cmd else 'resume to target step count'))
        return
    if os.environ.get('UDM_OUTPUT_DIR'):
        raise SystemExit('Unset UDM_OUTPUT_DIR: all scripts in this package use the package directory.')
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(exist_ok=True)
    state_path = ROOT/'pipeline_state.json'
    identity = source_identity()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['source_identity'] != identity:
            raise SystemExit('Inputs or scripts changed since this run. Start from a fresh extracted ZIP.')
    else:
        if any((ROOT/name).exists() for name in ('runs', 'extensions.json', 'assumptions.json', 'extensions_chains')):
            raise SystemExit('Existing numerical outputs without pipeline state. Use a fresh extracted ZIP.')
        state = {'source_identity': identity, 'completed': []}
    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)
    save()
    # Import after the thread settings, and only when an actual run is requested.
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    for name, cmd in jobs:
        if name in state['completed']:
            print('SKIP completed:', name, flush=True)
            continue
        if name == 'main_chain':
            import fit_joint as fj
            path = fj.result_path('n1_mcmc')
            done = int(fj.load_sampler_checkpoint(1)['done']) if path.exists() else 0
            while done < 40000:
                segment = min(5000, 40000-done)
                launch('mcmc', ['fit_joint.py', 'mcmc', str(segment)])
                done = int(fj.load_sampler_checkpoint(1)['done'])
        elif name.startswith('chain_'):
            model = name[len('chain_'):]
            import extensions as ext
            path = ext.chain_file(model)
            done = int(np.load(path)['steps']) if path.exists() else 0
            if done < 12000:
                launch(name, ['extensions.py', 'chain', model, str(12000-done)])
        else:
            launch(name, cmd)
        state['completed'].append(name)
        save()
    print('\nNumerical stages complete. Review convergence, validation and new scientific results before updating the manuscript.')

if __name__ == '__main__':
    main()
