"""Exact diagonalisation driver: finite-size Drude weight D_N(beta) and current autocorrelation C_N(t).

    D_N(beta) = sum_{E_m = E_n} p_m |A_mn|^2 / (N Z)            infinite-time average of <A(t) A>_beta / N
    C_N(t)    = Tr(rho A(t) B) / (norm Tr rho)

* bc='pbc' (ring): A = B = total current, norm = N.  The infinite-temperature value tends to the Drude weight
  D(T = inf) = 32 - 243 sqrt3 / (10 pi) = 18.6027 (note, App. E: D_14 = 18.7897, ..., D_30 = 18.6708).
* bc='obc' (open chain): on an open chain J = i[H, P] has no matrix elements inside degenerate blocks, so D_N = 0
  exactly; C_N(t) shows the bulk plateau until the light cone reaches the ends.  probe='local' gives
  C(t) = <J(t) j_c>_beta with j_c the current density in the middle (sym: average over three neighbouring
  sublattices), the quantity the MPO / MPS codes compute on open windows.

Each sector is written to its own .npz file (beta-independent), so an interrupted run restarts where it stopped
and further temperatures reuse the diagonalisations.
"""
import hashlib
import json
import os
import time

import numpy as np

from ..model import Chain, pattern
from .algebra import representations, ReducedStructure, full_representation, is_hermitian
from .spectrum import sector_data, combine

CACHE_VERSION = 2


def _params_key(p):
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]


def pattern_json(op):
    """JSON-safe form of a pattern: {'0,1': [re, im], ...}"""
    return {','.join(map(str, k)): [complex(c).real, complex(c).imag] for k, c in sorted(pattern(op).items())}


def run_ed(L, bc='pbc', op='J', probe='total', betas=(0.0,), ts=None, method='reduced', tol=1e-9,
           sym=True, outdir=None, be=None, chunk=1024, log=print):
    """returns a dict with D(beta), C0(beta) and (if ts is given) C(t); see the module docstring.

    op: pattern name (qlc.model.PATTERNS) or dict {offsets: coefficient}; it must be Hermitian.
    outdir: sector cache.  The cached data do not depend on beta, so further betas reuse every block."""
    from ..backend import get_backend
    be = be or get_backend()
    chain = Chain(L, bc)
    betas = [float(b) for b in betas]
    ts = None if ts is None else np.asarray(ts, dtype=float)
    if probe not in ('total', 'local'):
        raise ValueError("probe must be 'total' or 'local'")
    if int(chunk) < 1:
        raise ValueError('chunk must be a positive integer')
    if not is_hermitian(full_representation(chain).op_terms(op)):
        raise ValueError('the operator pattern is not Hermitian')
    centres = chain.centre_terms(sym, op) if probe == 'local' else None
    norm = float(L) if probe == 'total' else 1.0
    t0 = time.time()
    reps = representations(chain, method, use_orbits=(probe == 'total'))
    info = dict(L=L, bc=bc, op=op if isinstance(op, str) else 'custom', op_pattern=pattern_json(op), probe=probe,
                method=method, n_blocks=len(reps), block_dim=reps[0].dim, centres=centres, norm=norm)
    if method == 'reduced':
        rs = ReducedStructure(chain)
        info.update(k=rs.k, c=rs.c)
    if log:
        log(f'ED {chain}: {len(reps)} block(s) of dimension {reps[0].dim}, op={info["op"]}, probe={probe}, '
            f'backend={be.name}')
    key = _params_key(dict(version=CACHE_VERSION, L=L, bc=bc, op=info['op_pattern'], probe=probe, centres=centres,
                           method=method, tol=tol, ts=None if ts is None else ts.tolist()))
    datas = []
    for rep in reps:
        f = os.path.join(outdir, f'{rep.label}_{key}.npz') if outdir else None
        if f and os.path.exists(f):
            z = np.load(f, allow_pickle=False)
            d = {k: z[k] for k in z.files}
            for k in ('E0', 'weight', 'dim'):
                d[k] = d[k].item()
            d['weight'] = rep.weight
            datas.append(d)
            if log:
                log(f'  {rep.label}: loaded from {f}')
            continue
        A = rep.op_terms(op)
        B = None if probe == 'total' else rep.local_terms(op, centres)
        d = sector_data(rep, A, B, ts=ts, be=be, tol=tol, chunk=chunk, log=log)
        if f:
            os.makedirs(outdir, exist_ok=True)
            tmp = f'{f}.{os.getpid()}.tmp.npz'
            np.savez(tmp, **d)
            os.replace(tmp, f)
        datas.append(d)
    res = combine(datas, norm, betas)
    res.update(info, elapsed_s=time.time() - t0, backend=be.name, tol=tol, sym=sym)
    return res


def result_to_json(res):
    """JSON-serialisable copy (complex -> [re, im], arrays -> lists)"""
    def conv(v):
        if isinstance(v, np.ndarray):
            if np.iscomplexobj(v):
                return [[float(x.real), float(x.imag)] for x in v.ravel()]
            return v.tolist()
        if isinstance(v, (complex, np.complexfloating)):
            return [float(v.real), float(v.imag)]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        return v
    return conv(res)
