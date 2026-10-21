#!/usr/bin/env python3
"""Drude weight from run outputs.

Time series (run_mpo.py / run_mps.py JSON): estimators on C_J(t) (plateau over full tail periods, tail fit,
running mean) and their deviation from D_GHD(beta).  Two runs that differ only in dt can be Richardson-
extrapolated first with --richardson.
ED (run_ed.py JSON): table of D_N(beta) and polynomial extrapolation in 1/N per residue class N mod 3.

examples
  python scripts/drude.py results/mpo/split_auto_n64_b0_chi256_dt0.05.json
  python scripts/drude.py --richardson results/mpo/a_dt0.05.json results/mpo/a_dt0.025.json
  python scripts/drude.py results/ed/pbc_L*_J_total.json
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import json

import numpy as np

from qlc import analysis, theory


def load(f):
    with open(f) as fh:
        return json.load(fh)


def series(r, key):
    t = np.array([s['t'] for s in r['samples']])
    if key in r['samples'][0]:
        return t, np.array([s[key] for s in r['samples']])
    return t, np.array([s['C'][0] for s in r['samples']])


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
    if 'D' in e:
        print(f'   estimate  D = {e["D"]:.8f} +- {e.get("spread", float("nan")):.1e}')


def report_ed(files):
    rows = {}
    for f in files:
        r = load(f)
        for b, D in zip(r['betas'], r['D']):
            rows.setdefault((r['bc'], b), {})[r['L']] = D[0]
    for (bc, b), d in sorted(rows.items()):
        Ns = sorted(d)
        ref = theory.drude_ghd(b)
        print(f'ED {bc}, beta = {b:g}   (D_GHD = {ref:.8f})')
        for N in Ns:
            print(f'   L = {N:3d}   D_N = {d[N]:.8f}   D_N / D_GHD - 1 = {d[N] / ref - 1:+.3e}')
        if bc == 'pbc' and len(Ns) >= 3:
            for r, fit in analysis.extrapolate_size(np.array(Ns), np.array([d[N] for N in Ns]), deg=2, last=4).items():
                print(f'   N = {r} mod 3: quadratic fit in 1/N over {fit["Ns"]} -> D_inf = {fit["D_inf"]:.6f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='*')
    p.add_argument('--richardson', nargs=2, metavar=('DT', 'DT_HALF'), help='two runs at dt and dt/2')
    p.add_argument('--key', default='C_J', help="series to analyse: C_J (default) or C (raw evolved operator)")
    p.add_argument('--t-min', type=float, default=1.5, help='start of the late-time window')
    p.add_argument('--threads', default=1)
    a = p.parse_args()
    ed = [f for f in a.files if 'samples' not in load(f)]
    ts = [f for f in a.files if f not in ed]
    if ed:
        report_ed(ed)
    for f in ts:
        r = load(f)
        t, C = series(r, a.key)
        report_series(f, t, C, r.get('beta', 0.0), a.t_min, a.key == 'C_J' and r.get('op') in ('J', 'R'))
    if a.richardson:
        r1, r2 = load(a.richardson[0]), load(a.richardson[1])
        assert abs(r1['dt'] / r2['dt'] - 2) < 1e-9, 'second run must have half the dt of the first'
        t1, C1 = series(r1, a.key)
        t2, C2 = series(r2, a.key)
        t, C = analysis.richardson(t1, C1, t2, C2)
        report_series('Richardson (4 C(dt/2) - C(dt)) / 3', t, C, r1.get('beta', 0.0), a.t_min,
                      a.key == 'C_J' and r1.get('op') in ('J', 'R'))


if __name__ == '__main__':
    main()
