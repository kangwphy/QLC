#!/usr/bin/env python3
"""Timings of the TN kernels on the chosen backend (one JSON line per measurement).

  --part mpo     one Trotter step + one 21-shift measurement of the h-word MPO at saturated bond chi (random
                 tensors, window n): the steady-state cost of run_mpo.py; --complex for the beta > 0 case
  --part mazur   imaginary-time sweep of rho = exp(-beta H) and one evaluation of the Mazur matrices (G, b) for the
                 default 17 charges at bond chi, window m

examples
  python scripts/benchmark.py --part mpo --chis 64 128 256 --threads 1
  python scripts/benchmark.py --part mpo --chis 256 512 1024 --backend gpu
  python scripts/benchmark.py --part mazur --chis 64 128 256 --m 224 --backend gpu
"""
import _setup
_setup.setup(default_threads=1)

import argparse
import json
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--part', required=True, choices=['mpo', 'mazur'])
    p.add_argument('--chis', type=int, nargs='+', default=[64, 128])
    p.add_argument('--n', type=int, default=64, help='MPO window (spins)')
    p.add_argument('--m', type=int, default=224, help='Mazur window (words)')
    p.add_argument('--complex', action='store_true')
    _setup.add_backend_args(p, 1)
    a = p.parse_args()
    be = _setup.make_backend(a)
    dev = be.name if be.name == 'numpy' else str(be.device)
    if be.name == 'torch' and be.device.type == 'cuda':
        import torch
        dev = torch.cuda.get_device_name(0)
    tag = dict(backend=be.name, device=dev, threads=a.threads)

    def timed(f):
        be.sync()
        t0 = time.perf_counter()
        r = f()
        be.sync()
        return time.perf_counter() - t0, r

    rng = np.random.default_rng(0)
    from qlc.tn.wordmpo import WordMPO, folded_profile
    for chi in a.chis:
        m = (a.n - 2) if a.part == 'mpo' else a.m
        L = m + 4
        ts = []
        for k in range(L):
            bl, br = min(chi, 2 ** min(k, L - k)), min(chi, 2 ** min(k + 1, L - k - 1))
            x = rng.normal(size=(bl, 2, br)) / np.sqrt(2 * bl)          # keeps the norm O(1) on long windows
            ts.append(be.asarray((x + 1j * rng.normal(size=x.shape) / np.sqrt(2 * bl)) / np.sqrt(2)) if a.complex
                      else be.asarray(x, float))
        M = WordMPO(ts, be, 0)
        M.canonicalize(0)
        M.normalize()
        if a.part == 'mpo':
            M.sweep('rot', 0.05, chi, 1e-12)
            t_step, _ = timed(lambda: M.sweep('rot', 0.05, chi, 1e-12))
            t_meas, _ = timed(lambda: folded_profile(M, M, None, 20))
            row = dict(part='h-word MPO step', n=a.n, chi=chi, dtype='complex' if a.complex else 'real',
                       step_s=round(t_step, 3), meas21_s=round(t_meas, 3))
        else:
            from qlc.tn import mazur as mz
            M.sweep('imag_h', 0.025, chi, 1e-12)
            M.normalize()
            t_sweep, _ = timed(lambda: M.sweep('imag_h', 0.025, chi, 1e-12))     # one 2nd-order sweep (order 4: 3)
            M.normalize()
            sums, dens, J = mz.build_operators(mz.default_charges(), m, min(24, m // 8), be)
            t_mat, _ = timed(lambda: mz.mazur_matrices(M, sums, dens, J))
            row = dict(part='Mazur', m=m, chi=chi, charges=len(sums), imag_sweep_s=round(t_sweep, 2),
                       matrices_s=round(t_mat, 2))
        print(json.dumps(dict(tag, **row)), flush=True)


if __name__ == '__main__':
    main()
