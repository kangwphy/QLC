#!/usr/bin/env python3
"""Static Mazur bound of the Drude weight on the infinite chain from a purification iMPS (three-site unit cell).

    D(beta) >= M(beta) = b^T G^+ b,   G_ab = <Q_a,Q_b>_beta,   b_a = <J,Q_a>_beta        (per site)

The thermal state e^{-beta H/2}|Phi> is built by imaginary-time iTEBD (fourth-order Suzuki, step --dtau in units of
beta/2, bond --chi); G and b follow from the cell transfer matrix with exact sums over word lengths and separations
(no window).  Families: 'note' = Fig. 2(a) of the note (Q1, Q2, 5 gamma+, 6 lambda+, 6 gamma-), 'qlc' = the default
of run_mazur.py.  M_full uses all charges, M_even the even parts; G is normalized to unit diagonal and eigenvalues
below --rcond are dropped.  The output also checks <H>/N and <H,H> against the exact free-fermion values.
Cost per beta (K charges): K + 2 single-layer preparations (two GMRES solves each) and K(K+3)/2 + 2 transfer-matrix
pairs (one two-layer solve each); see benchmark.py --part imps.  GMRES runs on the host (also with --backend gpu) and
needs (restart + 1)(D_a - 1)(D_b - 1) chi^2 complex numbers for a pair of MPO bonds D_a, D_b: about 12 GB for the Q2 pair
of the 'note' family at chi = 192 and 21 GB at chi = 256.

examples
  python scripts/run_imazur.py --beta 0.5 1 --chi 16 --family qlc                      # quick (~1 min, 1 core)
  python scripts/run_imazur.py --beta 0.1 0.5 1 2 3 4 --chi 64 --backend gpu
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import os

from qlc.tn.imazur import FAMILIES, run_imazur


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--beta', type=float, nargs='+', default=[0.5, 1.0])
    p.add_argument('--chi', type=int, default=64, help='bond cap of the purification')
    p.add_argument('--dtau', type=float, default=0.05, help='S4 step in units of beta/2 (upper bound)')
    p.add_argument('--tol', type=float, default=1e-10, help='relative singular-value cutoff')
    p.add_argument('--canon-every', type=_setup.positive_int, default=1, help='exact canonical form every n S4 steps')
    p.add_argument('--family', default='note', choices=sorted(FAMILIES))
    p.add_argument('--rcond', type=float, default=1e-10, help='relative eigenvalue cutoff of the unit-diagonal G')
    p.add_argument('--gmres-tol', type=float, default=1e-12)
    p.add_argument('--out', default=None)
    _setup.add_backend_args(p, 1)
    a = p.parse_args()
    be = _setup.make_backend(a)
    params = {k: v for k, v in vars(a).items() if k not in ('out', 'backend', 'device', 'threads')}
    params['beta'] = sorted(a.beta)
    out = a.out or os.path.join('results', 'imazur', _setup.run_name(f'imazur_{a.family}_chi{a.chi}_dtau{a.dtau:g}',
                                                                      params) + '.json')
    run_imazur(a.beta, chi=a.chi, dtau=a.dtau, tol=a.tol, charges=a.family, rcond=a.rcond, canon_every=a.canon_every,
               gmres_tol=a.gmres_tol, be=be, log=_setup.log, out=out)
    _setup.log(f'-> {out}')


if __name__ == '__main__':
    main()
