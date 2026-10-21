#!/usr/bin/env python3
"""Exact diagonalisation: finite-size Drude weight D_N(beta) and current autocorrelation C_N(t).

Rings (--bc pbc) use the reduced representation (block dimension 2^k: N = 24 -> 1024, N = 30 -> 8192) and one
sector per translation orbit.  Open chains (--bc obc) have D_N = 0 exactly; use --probe local for the bulk-like
C(t) = <J(t) j_c> that the MPO / MPS codes compute.  Each sector is cached (beta-independent) in
<out>/sectors/<bc>_L<L>_<op>_<probe>/, so an interrupted run restarts where it stopped and other temperatures
reuse it (a different t grid is a new cache entry).
Output: <out>/<bc>_L<L>_<op>_<probe>[_nosym][_t<tmax>_dt<dt>][_full][_tol<tol>]_b<betas>.json (more than four
betas: b<first>-<last>_n<count>_<hash>).
Memory per block: about 4 D^2 x 16 bytes (D = 8192: ~4 GiB, 16384: ~16 GiB, 32768: ~64 GiB).

examples
  python scripts/run_ed.py --L 12 14 16 18 20 --bc pbc --beta 0 0.5 1 2
  python scripts/run_ed.py --L 24 --beta 0 1 --tmax 20 --dt 0.05
  python scripts/run_ed.py --L 30 --beta 0 --backend gpu                    # GPU eigh (torch)
  python scripts/run_ed.py --L 20 --bc obc --probe local --beta 0 1 --tmax 3 --dt 0.05
"""
import _setup
_setup.setup(default_threads=None)

import argparse
import hashlib
import json
import os
import time

import numpy as np

from qlc.ed import run_ed, result_to_json
from qlc import theory
from qlc.model import PATTERNS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--L', type=int, nargs='+', required=True, help='system size(s) (spins)')
    p.add_argument('--bc', default='pbc', choices=['pbc', 'obc'])
    p.add_argument('--op', default='J', choices=sorted(PATTERNS), help='operator pattern (J = energy current)')
    p.add_argument('--probe', default='total', choices=['total', 'local'])
    p.add_argument('--no-sym', action='store_true', help='local probe at one sublattice only')
    p.add_argument('--beta', type=float, nargs='+', default=[0.0])
    p.add_argument('--tmax', type=float, default=None, help='compute C(t) on [0, tmax]')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--method', default='reduced', choices=['reduced', 'full'])
    p.add_argument('--tol', type=float, default=1e-9, help='relative tolerance for degenerate energies')
    p.add_argument('--chunk', type=_setup.positive_int, default=1024, help='eigenvector columns per chunk (memory)')
    p.add_argument('--out', default='results/ed')
    _setup.add_backend_args(p, None)
    a = p.parse_args()
    be = _setup.make_backend(a)
    ts = None if a.tmax is None else np.round(np.arange(0, a.tmax + 1e-9, a.dt), 12)
    os.makedirs(a.out, exist_ok=True)
    for L in a.L:
        base = f'{a.bc}_L{L}_{a.op}_{a.probe}' + ('_nosym' if a.no_sym and a.probe == 'local' else '')
        bl = '-'.join(f'{b:g}' for b in a.beta)
        if len(a.beta) > 4:                     # long temperature scans: first, last, count and a hash
            bl = f'{a.beta[0]:g}-{a.beta[-1]:g}_n{len(a.beta)}_' + \
                 hashlib.sha1(','.join(map(repr, a.beta)).encode()).hexdigest()[:6]
        tag = (f'_t{a.tmax:g}_dt{a.dt:g}' if ts is not None else '') + ('_full' if a.method == 'full' else '') + \
              (f'_tol{a.tol:g}' if a.tol != 1e-9 else '') + '_b' + bl
        name = base + tag
        t0 = time.time()
        res = run_ed(L, a.bc, op=a.op, probe=a.probe, betas=a.beta, ts=ts, method=a.method, tol=a.tol,
                     sym=not a.no_sym, outdir=os.path.join(a.out, 'sectors', base), be=be, chunk=a.chunk,
                     log=_setup.log)
        if a.op == 'J':
            res['D_GHD'] = [theory.drude_ghd(b) for b in a.beta]
        with open(os.path.join(a.out, name + '.json'), 'w') as f:
            json.dump(result_to_json(res), f, indent=1)
        for b, D, C0 in zip(a.beta, res['D'], res['C0']):
            ref = f'   D_GHD = {theory.drude_ghd(b):.6f}' if a.op == 'J' else ''
            _setup.log(f'{a.bc} L={L} beta={b:g}: D_N = {D.real:.8f}   C(0) = {C0.real:.8f}{ref}')
        _setup.log(f'  -> {os.path.join(a.out, name + ".json")}  ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
