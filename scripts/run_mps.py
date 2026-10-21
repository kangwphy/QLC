#!/usr/bin/env python3
"""Spin-basis purification MPS + TEBD: C(t) = sum_x <j_x(t) j_c>_beta on an open chain of L spins.

An independent check of the h-word MPO (different representation, thermal state and correlation formula).
  --route direct : j_c|psi>, J|psi>   -> C_J(t)
  --route split  : r_c|psi>, R|psi>   -> C_R(t),  C_J = C_R - C_R(0) + C_J(0)  (C_J(0) on the same state)
The thermal state can be cached with --psi-file (reused when L, beta, dtau, chi_psi agree).  Checkpoints as in
run_mpo.py.  Use dt <= 0.025 at beta ~ 1 (the 3-sublattice circuit has a larger method error than the h-word one).

examples
  python scripts/run_mps.py --route split --L 64 --beta 1 --chi 256 --chi-psi 192 --dt 0.025 --tmax 4 \\
      --psi-file results/mps/psi_b1_L64.pkl
  python scripts/run_mps.py --route direct --L 48 --beta 0 --chi 256 --dt 0.025 --tmax 3 --backend gpu
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import os

from qlc.tn.purification import run_purification


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--route', default='split', choices=['direct', 'split'])
    p.add_argument('--op', default=None, help='custom pattern name instead of the route operator')
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--chi', type=int, default=256)
    p.add_argument('--chi-psi', type=int, default=192)
    p.add_argument('--tol', type=float, default=1e-10)
    p.add_argument('--tol-psi', type=float, default=1e-12)
    p.add_argument('--dt', type=float, default=0.025)
    p.add_argument('--dtau', type=float, default=0.0125)
    p.add_argument('--tmax', type=float, default=4.0)
    p.add_argument('--no-sym', action='store_true', help='single-sublattice probe (not recommended)')
    p.add_argument('--psi-file', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--ckpt', default=None, help='checkpoint file (default <out>.ckpt.pkl; "none" to disable)')
    p.add_argument('--ckpt-every', type=int, default=5)
    _setup.add_backend_args(p, 1)
    a = p.parse_args()
    be = _setup.make_backend(a)
    out = a.out or os.path.join('results', 'mps', f'{a.route}_L{a.L}_b{a.beta:g}_chi{a.chi}_dt{a.dt:g}.json')
    ckpt = None if (a.ckpt or '').lower() == 'none' else (a.ckpt or out + '.ckpt.pkl')
    run_purification(route=a.route, L=a.L, beta=a.beta, chi=a.chi, chi_psi=a.chi_psi, tol=a.tol, tol_psi=a.tol_psi,
                     dt=a.dt, dtau=a.dtau, tmax=a.tmax, sym=not a.no_sym, op=a.op, psi_file=a.psi_file, out=out,
                     ckpt=ckpt, ckpt_every=a.ckpt_every, be=be, log=_setup.log)
    _setup.log(f'-> {out}')


if __name__ == '__main__':
    main()
