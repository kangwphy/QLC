"""The command-line scripts run end to end on tiny problems."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args, cwd):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_scripts(tmp_path):
    s = os.path.join(ROOT, 'scripts')
    out = run([f'{s}/run_ed.py', '--L', '8', '10', '--beta', '0', '1', '--tmax', '0.5', '--out', str(tmp_path / 'ed')],
              tmp_path)
    assert 'D_N' in out
    ed = json.load(open(tmp_path / 'ed' / 'pbc_L10_J_total.json'))
    assert abs(ed['C0'][0][0] - 20) < 1e-12 and len(ed['C']) == 2
    run([f'{s}/run_ed.py', '--L', '10', '--bc', 'obc', '--probe', 'local', '--beta', '0', '--tmax', '0.2',
         '--out', str(tmp_path / 'ed')], tmp_path)
    f_mpo = str(tmp_path / 'mpo.json')
    run([f'{s}/run_mpo.py', '--route', 'split', '--n', '12', '--chi', '16', '--dt', '0.05', '--tmax', '0.3',
         '--out', f_mpo], tmp_path)
    r = json.load(open(f_mpo))
    assert r['status'] == 'completed' and r['samples'][0]['C_J'] == 20.0 and os.path.exists(f_mpo + '.ckpt.pkl')
    f_mps = str(tmp_path / 'mps.json')
    run([f'{s}/run_mps.py', '--route', 'direct', '--L', '10', '--beta', '0.5', '--chi', '16', '--chi-psi', '16',
         '--dt', '0.05', '--dtau', '0.05', '--tmax', '0.2', '--out', f_mps], tmp_path)
    assert json.load(open(f_mps))['status'] == 'completed'
    out = run([f'{s}/drude.py', f_mpo, f_mps, str(tmp_path / 'ed' / 'pbc_L8_J_total.json'),
               str(tmp_path / 'ed' / 'pbc_L10_J_total.json')], tmp_path)
    assert 'running mean' in out and 'ED pbc' in out
    jobs = run([f'{s}/make_jobs.py', 'run_mpo.py', '--route', 'direct', 'split', '--chi', '64', '128',
                '--beta', '0,1', '--profile'], tmp_path)
    lines = jobs.strip().splitlines()
    assert len(lines) == 4 and all('--beta 0 1' in x and x.endswith('--profile') for x in lines)
