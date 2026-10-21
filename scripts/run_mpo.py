#!/usr/bin/env python3
"""h-word MPO Heisenberg evolution of the energy current of the infinite FFD chain (open window of n spins).

  --route direct : evolve j_0                      -> C_J(t)
  --route split  : evolve r_0 (J = 3 Q1 + R)       -> C_R(t),  C_J = C_R - C_R(0) + C_J(0)
beta = 0 is exact operator dynamics; beta > 0 uses the thermal factor rho^{1/2} as an h-word MPO.  The Drude
weight is the long-time value of C_J(t): run scripts/drude.py on the output.  The run writes the JSON after every
sample and a checkpoint (<out>.ckpt.pkl) every --ckpt-every samples; rerunning the same command resumes
(tmax may be increased).

examples
  python scripts/run_mpo.py --route split  --n 64 --chi 256 --dt 0.05 --tmax 6
  python scripts/run_mpo.py --route direct --n 64 --chi 256 --dt 0.025 --tmax 6
  python scripts/run_mpo.py --route split  --n 64 --beta 1 --chi 256 --chi-rho 128 --tmax 4
  python scripts/run_mpo.py --route split  --n 96 --chi 512 --backend gpu
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import os

from qlc.tn.heisenberg import run_hword


def default_out(a):
    return os.path.join('results', 'mpo', f'{a.route}_{a.op or "auto"}_n{a.n}_b{a.beta:g}_chi{a.chi}_dt{a.dt:g}.json')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--route', default='split', choices=['direct', 'split'])
    p.add_argument('--op', default=None, help='custom pattern name instead of the route operator (J, R, Q1, J3, H)')
    p.add_argument('--n', type=int, default=64, help='window size (spins); n - 2 words')
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--chi', type=int, default=256)
    p.add_argument('--chi-rho', type=int, default=128, help='bond cap of rho^{1/2} (beta > 0)')
    p.add_argument('--tol', type=float, default=1e-12, help='relative singular-value cutoff')
    p.add_argument('--dt', type=float, default=0.05, help='Trotter step of the folded evolution (t = 2 tau)')
    p.add_argument('--tau-im', type=float, default=0.0125, help='imaginary-time step for rho^{1/2}')
    p.add_argument('--tmax', type=float, default=6.0)
    p.add_argument('--sample-every', type=int, default=1)
    p.add_argument('--x0', type=float, default=12.0, help='shift sum |x| <= x0 + vlc t')
    p.add_argument('--vlc', type=float, default=7.3)
    p.add_argument('--cj0', type=float, default=None, help='C_J(0) for the split route at beta > 0 (default: computed)')
    p.add_argument('--profile', action='store_true', help='store the spatial profile C_x(t)')
    p.add_argument('--out', default=None)
    p.add_argument('--ckpt', default=None, help='checkpoint file (default <out>.ckpt.pkl; "none" to disable)')
    p.add_argument('--ckpt-every', type=int, default=10)
    p.add_argument('--max-rss-gib', type=float, default=None)
    _setup.add_backend_args(p, 1)
    a = p.parse_args()
    be = _setup.make_backend(a)
    out = a.out or default_out(a)
    ckpt = None if (a.ckpt or '').lower() == 'none' else (a.ckpt or out + '.ckpt.pkl')
    run_hword(route=a.route, n=a.n, beta=a.beta, chi=a.chi, tol=a.tol, dt=a.dt, tmax=a.tmax, chi_rho=a.chi_rho,
              tau_im=a.tau_im, sample_every=a.sample_every, x0=a.x0, vlc=a.vlc, op=a.op, cj0=a.cj0, be=be, out=out,
              ckpt=ckpt, ckpt_every=a.ckpt_every, max_rss_gib=a.max_rss_gib, save_profile=a.profile, log=_setup.log)
    _setup.log(f'-> {out}')


if __name__ == '__main__':
    main()
