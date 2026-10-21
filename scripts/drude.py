#!/usr/bin/env python3
"""Drude weight from run outputs.

Time series (run_mpo.py / run_mps.py JSON): estimators on C_J(t) (plateau over full tail periods, tail fit,
running mean) and their deviation from D_GHD(beta).  Two runs that differ only in dt can be Richardson-
extrapolated first with --richardson.
ED (run_ed.py JSON): table of D_N(beta) per (bc, op, probe); for the energy current on rings also D_N / D_GHD and
a polynomial extrapolation in 1/N per residue class N mod 3.  Open-chain files with C(t) are analysed as time
series (their D_N vanishes exactly).

examples
  python scripts/drude.py results/mpo/split_auto_n64_b0_chi256_dt0.05_*.json
  python scripts/drude.py --richardson results/mpo/a_dt0.05.json results/mpo/a_dt0.025.json
  python scripts/drude.py results/ed/pbc_L*_J_total_b*.json
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import json
import sys

import numpy as np

from qlc import analysis, theory


def load(f):
    with open(f) as fh:
        return json.load(fh)


def series(r, key):
    """(t, values, is_CJ) of a TN run; key 'C_J' needs the derived C_J series unless the evolved op is J"""
    t = np.array([s['t'] for s in r['samples']])
    if key == 'C_J' and 'C_J' in r['samples'][0]:
        return t, np.array([s['C_J'] for s in r['samples']]), True
    if key == 'C_J' and r.get('op') != 'J':
        sys.exit(f"run with op={r.get('op')!r} has no C_J series; use --key C for the raw correlation")
    return t, np.array([s['C'][0] for s in r['samples']]), r.get('op') == 'J'


def report_series(name, t, C, beta, t_min, is_J):
    ref = theory.drude_ghd(beta) if is_J else None
    e = analysis.drude_estimates(t, C, t_min=t_min, reference=ref)
    print(f'{name}: beta = {beta:g}, t_max = {e["t_max"]:.3f}' + (f', D_GHD = {ref:.8f}' if ref else ''))
    pl, tf = e['plateau'], e['tail_fit']
    rel = lambda d: f'  ({d["rel_dev"]:+.2e})' if 'rel_dev' in d else ''
    if pl['window']:
        print(f'   plateau   D = {pl["D"]:.8f}{rel(pl)}   window [{pl["window"][0]:.3f}, {pl["window"][1]:.3f}], '
              f'{pl["periods"]} periods')
    if np.isfinite(tf['D']):
        print(f'   tail fit  D = {tf["D"]:.8f}{rel(tf)}   |a| = {tf["amp"]:.3g}, rms {tf["rms"]:.1e}, {tf["n"]} points')
    print(f'   running mean (1/t) int C = {e["running_mean"]:.8f}   last C = {e["last"]:.8f}')
    if pl['periods'] == 0:
        print(f'   WARNING: [t_min, t_max] = [{t_min:g}, {e["t_max"]:g}] is shorter than one tail period '
              f'({theory.T_TAIL:.3f}): no reliable estimate; extend tmax or lower --t-min')
    elif 'D' in e:
        print(f'   estimate  D = {e["D"]:.8f} +- {e.get("spread", float("nan")):.1e}')


def report_ed(files, t_min):
    rows = {}
    for f in files:
        r = load(f)
        key = (r['bc'], r.get('op', 'J'), r.get('probe', 'total'),
               r.get('sym', True) if r.get('probe', 'total') == 'local' else True)
        for i, (b, D) in enumerate(zip(r['betas'], r['D'])):
            rows.setdefault(key + (b,), {})[r['L']] = (D[0], f, r, i)
    for (bc, op, probe, sym, b), d in sorted(rows.items()):
        Ns = sorted(d)
        is_ref = bc == 'pbc' and op == 'J' and probe == 'total'
        ref = theory.drude_ghd(b) if is_ref else None
        print(f'ED {bc}, op = {op}, probe = {probe}' + ('' if sym or probe == 'total' else ', nosym') +
              f', beta = {b:g}' + (f'   (D_GHD = {ref:.8f})' if ref else ''))
        for N in Ns:
            D = d[N][0]
            print(f'   L = {N:3d}   D_N = {D:.8f}' + (f'   D_N / D_GHD - 1 = {D / ref - 1:+.3e}' if ref else ''))
        if is_ref and len(Ns) >= 3:
            for r, fit in analysis.extrapolate_size(np.array(Ns), np.array([d[N][0] for N in Ns]), deg=2,
                                                     last=4).items():
                print(f'   N = {r} mod 3: quadratic fit in 1/N over {fit["Ns"]} -> D_inf = {fit["D_inf"]:.6f}')
        if bc == 'obc':
            for N in Ns:
                _, f, r, i = d[N]
                if 'C' in r:
                    t = np.array(r['ts'])
                    C = np.array([c[0] for c in r['C'][i]])
                    report_series(f'   {f} (open chain C(t))', t, C, b, t_min, op == 'J')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='*')
    p.add_argument('--richardson', nargs=2, metavar=('DT', 'DT_HALF'), help='two runs at dt and dt/2')
    p.add_argument('--key', default='C_J', choices=['C_J', 'C'],
                   help='series to analyse: C_J (default) or C (raw correlation of the evolved operator)')
    p.add_argument('--t-min', type=float, default=1.5, help='start of the late-time window (> 0)')
    p.add_argument('--threads', default=1)
    a = p.parse_args()
    if a.t_min <= 0:
        sys.exit('--t-min must be > 0')
    data = {f: load(f) for f in a.files}
    ed = [f for f, r in data.items() if 'samples' not in r]
    if ed:
        report_ed(ed, a.t_min)
    for f, r in data.items():
        if f in ed:
            continue
        t, C, is_J = series(r, a.key)
        report_series(f, t, C, r.get('beta', 0.0), a.t_min, is_J)
    if a.richardson:
        r1, r2 = load(a.richardson[0]), load(a.richardson[1])
        if abs(r1['dt'] / r2['dt'] - 2) > 1e-9:
            sys.exit('--richardson: the second run must have half the dt of the first')
        ignore = {'dt', 'samples', 'status', 'backend', 'backends', 'elapsed_s', 'psi_bond', 'rho_bond',
                  'discarded_init', 'discarded_rho_init', 'n_im_steps', 'tau_im_eff', 'C_J0', 'C_J0_source',
                  'disconnected'}
        diff = {k: (r1.get(k), r2.get(k)) for k in set(r1) | set(r2) if k not in ignore and r1.get(k) != r2.get(k)}
        if diff:
            sys.exit(f'--richardson: the runs differ in more than dt: {diff}')
        t1, C1, is_J = series(r1, a.key)
        t2, C2, _ = series(r2, a.key)
        t, C = analysis.richardson(t1, C1, t2, C2)
        report_series('Richardson (4 C(dt/2) - C(dt)) / 3', t, C, r1.get('beta', 0.0), a.t_min, is_J)


if __name__ == '__main__':
    main()
