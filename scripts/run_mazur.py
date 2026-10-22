#!/usr/bin/env python3
"""Static Mazur bound of the Drude weight from the quasi-local charges (MPS in the h_S basis, no time evolution).

    D(beta) >= M(beta) = b^T G^+ b,   G_ab = <Q_a, Q_b>_beta,   b_a = <J, Q_a>_beta        (per site)

Charges: even parts Q^{lambda,+}_p (--lam), Q^{gamma,+}_p (--gam), local Q1 (--local), and the odd parts Q^{gamma,-}_p
(--gam-odd) that couple to the even ones at beta > 0.  One imaginary-time trajectory rho = e^{-beta H} (bond --chi,
step <= --tau, 4th-order Suzuki by default: the 2nd-order error biases M upwards) gives every --beta.  M_full uses
all charges, M_even only the even parts; both are compared with D_GHD(beta).  Window: the bulk densities start at
--y0 (default max(24, 16 beta_max), outside the thermal boundary layer of the open window) and need --m - --y0 ~ 200
words for the charge tails (the beta = 0 check against the closed-form kernels is printed).  The eigenvalue cutoff of
G follows the estimated noise of G; M at a scan of cutoffs is stored in the JSON.  GPU recommended (--backend gpu) for chi >= 128.  CPU: 1 BLAS thread by
default (OpenBLAS); with MKL (Linux) --threads 16 helps for chi >= 128.

examples
  python scripts/run_mazur.py --beta 0 0.5 1 --m 120 --chi 64                     # quick
  python scripts/run_mazur.py --beta 0 0.25 0.5 1 1.5 2 --chi 256 --backend gpu
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import os

from qlc.tn.mazur import run_mazur, default_charges


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--beta', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    p.add_argument('--m', type=int, default=None, help='window (words); default y0 + 200')
    p.add_argument('--y0', type=int, default=None, help='left buffer / first word of the bulk densities; default '
                   'max(24, 16 beta_max)')
    p.add_argument('--chi', type=int, default=128, help='bond cap of rho = exp(-beta H)')
    p.add_argument('--tau', type=float, default=0.05, help='imaginary-time step (upper bound)')
    p.add_argument('--order', type=int, default=4, choices=[2, 4], help='Trotter order of the imaginary-time steps')
    p.add_argument('--tol', type=float, default=1e-10, help='relative singular-value cutoff for rho')
    p.add_argument('--lam', type=float, nargs='*', default=[-1.25, -1.6, -2.2, -3.5, -6, -12])
    p.add_argument('--gam', type=float, nargs='*', default=[-0.2, -0.35, -0.5, -0.65, -0.8])
    p.add_argument('--gam-odd', type=float, nargs='*', default=None, help='default: the --gam values')
    p.add_argument('--local', nargs='*', default=['Q1'])
    p.add_argument('--rcond', type=float, default=1e-10, help='relative eigenvalue cutoff of G')
    p.add_argument('--out', default=None)
    _setup.add_backend_args(p, 1)
    a = p.parse_args()
    be = _setup.make_backend(a)
    charges = default_charges(a.lam, a.gam, a.gam_odd, a.local)
    params = {k: v for k, v in vars(a).items() if k not in ('out', 'backend', 'device', 'threads')}
    params['beta'] = sorted(a.beta)
    out = a.out or os.path.join('results', 'mazur', _setup.run_name(f'mazur_chi{a.chi}_tau{a.tau:g}', params)
                                + '.json')
    run_mazur(a.beta, m=a.m, y0=a.y0, chi=a.chi, tau=a.tau, tol=a.tol, charges=charges, rcond=a.rcond, order=a.order,
              be=be, log=_setup.log, out=out)
    _setup.log(f'-> {out}')


if __name__ == '__main__':
    main()
