"""The command-line scripts run end to end on tiny problems."""
import glob
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, 'scripts')


def run(args, cwd, ok=True):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_ed_scripts(tmp_path):
    out = run([f'{S}/run_ed.py', '--L', '8', '10', '--beta', '0', '1', '--tmax', '0.5', '--out', 'ed'], tmp_path)
    assert 'D_N' in out.stdout
    ed = json.load(open(tmp_path / 'ed' / 'pbc_L10_J_total_t0.5_dt0.05_b0-1.json'))
    assert abs(ed['C0'][0][0] - 20) < 1e-12 and len(ed['C']) == 2
    run([f'{S}/run_ed.py', '--L', '10', '--op', 'R', '--beta', '0', '--out', 'ed'], tmp_path)
    run([f'{S}/run_ed.py', '--L', '8', '--beta', '0', '0.1', '0.2', '0.3', '0.4', '0.5', '--out', 'long'], tmp_path)
    assert len(glob.glob(str(tmp_path / 'long' / 'pbc_L8_J_total_b0-0.5_n6_*.json'))) == 1
    run([f'{S}/run_ed.py', '--L', '12', '--bc', 'obc', '--probe', 'local', '--beta', '0', '--tmax', '2',
         '--out', 'ed'], tmp_path)
    r = run([f'{S}/drude.py'] + sorted(glob.glob(str(tmp_path / 'ed' / '*.json'))), tmp_path).stdout
    assert 'ED pbc, op = J, probe = total, beta = 0' in r and 'ED pbc, op = R' in r
    assert 'open chain C(t)' in r


def test_tn_scripts(tmp_path):
    f_mpo = str(tmp_path / 'mpo.json')
    run([f'{S}/run_mpo.py', '--route', 'split', '--n', '12', '--chi', '16', '--dt', '0.05', '--tmax', '0.3',
         '--out', f_mpo], tmp_path)
    r = json.load(open(f_mpo))
    assert r['status'] == 'completed' and r['samples'][0]['C_J'] == 20.0 and os.path.exists(f_mpo + '.ckpt.pkl')
    # default names: distinct for different parameters, identical (resume) for a longer tmax
    for chi_rho in ('8', '16'):
        run([f'{S}/run_mpo.py', '--n', '12', '--beta', '0.5', '--chi', '8', '--chi-rho', chi_rho, '--tau-im', '0.05',
             '--tmax', '0.1'], tmp_path)
    assert len(glob.glob(str(tmp_path / 'results' / 'mpo' / '*.json'))) == 2
    o = run([f'{S}/run_mpo.py', '--n', '12', '--beta', '0.5', '--chi', '8', '--chi-rho', '8', '--tau-im', '0.05',
             '--tmax', '0.2'], tmp_path).stdout
    assert 'resumed' in o
    f_r = str(tmp_path / 'mpo_R.json')
    run([f'{S}/run_mpo.py', '--route', 'direct', '--op', 'R', '--n', '12', '--chi', '16', '--tmax', '0.2',
         '--out', f_r], tmp_path)
    assert run([f'{S}/drude.py', f_r], tmp_path, ok=False).returncode != 0      # no C_J series for op R
    assert 'running mean' in run([f'{S}/drude.py', '--key', 'C', f_r], tmp_path).stdout
    f_mps = str(tmp_path / 'mps.json')
    run([f'{S}/run_mps.py', '--route', 'direct', '--L', '10', '--beta', '0.5', '--chi', '16', '--chi-psi', '16',
         '--dt', '0.05', '--dtau', '0.05', '--tmax', '0.2', '--out', f_mps, '--psi-file', 'cache/psi.pkl'], tmp_path)
    assert json.load(open(f_mps))['status'] == 'completed' and (tmp_path / 'cache' / 'psi.pkl').exists()
    out = run([f'{S}/drude.py', f_mpo, f_mps], tmp_path).stdout
    assert 'running mean' in out and 'WARNING' in out                           # runs shorter than one period


def test_imazur_scripts(tmp_path):
    run([f'{S}/run_imazur.py', '--beta', '0', '0.1', '--chi', '4', '--family', 'qlc'], tmp_path)
    f, = glob.glob(str(tmp_path / 'results' / 'imazur' / 'imazur_qlc_chi4_dtau0.05_*.json'))
    r = json.load(open(f))
    assert len(r['points']) == 2 and r['points'][0]['bond'] == 1 and abs(r['points'][0]['JJ'] - 20) < 1e-10
    assert run([f'{S}/run_imazur.py', '--beta', '0.1', '--canon-every', '0'], tmp_path, ok=False).returncode == 2
    assert '"part": "iMPS"' in run([f'{S}/benchmark.py', '--part', 'imps', '--chis', '4'], tmp_path).stdout


def test_make_jobs(tmp_path):
    jobs = run([f'{S}/make_jobs.py', 'run_mpo.py', '--route', 'direct', 'split', '--beta', '0', '1', '--profile'],
               tmp_path).stdout.strip().splitlines()
    assert len(jobs) == 4 and all(x.endswith('--profile') and 'scripts/run_mpo.py' in x for x in jobs)
    assert sum('--beta 1' in x for x in jobs) == 2
    ed = run([f'{S}/make_jobs.py', 'scripts/run_ed.py', '--L', '8', '10', '--beta', '0,1', '--out', 'ed'],
             tmp_path).stdout.strip().splitlines()
    assert len(ed) == 2 and all('--beta 0 1' in x and 'scripts/scripts' not in x for x in ed)
    for line in ed:                                                            # the generated lines run
        argv = shlex.split(line)
        argv[0], argv[1] = sys.executable, os.path.join(S, os.path.basename(argv[1]))
        r = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    w = run([f'{S}/make_jobs.py', './scripts/run_mpo.py', '--beta', '0,1'], tmp_path)
    assert 'warning' in w.stderr and w.stdout.startswith('python3 scripts/run_mpo.py')
    assert run([f'{S}/make_jobs.py', 'run_ed.py', '8', '--L', '8'], tmp_path, ok=False).returncode == 2
