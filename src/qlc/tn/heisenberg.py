"""Energy-current autocorrelation of the infinite FFD chain by h-word MPO Heisenberg evolution.

Two routes (note, App. E):

    direct : evolve the current density  j_0 = 2i h_0 h_1 + 4i h_0 h_2                 ->  C_J(t)
    split  : J = 3 Q1 + R with Q1 conserved; evolve  r_0 = i h_0 h_2 - i h_0 h_1         ->  C_R(t),
             C_J(t) = C_R(t) - C_R(0) + C_J(0)          (beta = 0: C_J(t) = 18 + C_R(t) exactly)

The split route evolves only the part of J that is not carried by Q1 (norm 2 instead of 20), so it tolerates
truncation much better at long times.  C(t) = sum_x C_x(t) is summed over shifts |x| <= x0 + v t (light cone,
v = 7.3 > max velocity 7.24) and C_{-x} = C_x by time reversal.  The Drude weight is the long-time value of C_J(t)
(see qlc.analysis).  Open window of n spins (n - 2 words): with folding the operator spreads v t / 2 to each side,
so the light-cone front reaches the window edge at t_edge = (n - 2 - span) / v; window errors of ~1e-4 already
appear around t ~ t_edge / 2 to 2 t_edge / 3, so monitor `edge_weight` (a warning is printed above 1e-4; the note
used n = 64 for t <= 6 and n = 96 for t <= 3.2).  At beta > 0 the window must also hold the thermal factor
(n >= 32 at beta = 1).  Only K-odd operators (currents) are allowed at beta > 0:
for K-even ones <o>_beta != 0 and the sum would contain the disconnected part.
"""
import json
import math
import os
import pickle
import resource
import sys
import time

import numpy as np

from ..model import pattern, STATIC_BETA0
from . import hword
from .mps import imaginary_steps
from .wordmpo import WordMPO, folded_profile, pattern_words, k_parity


def rss_gib():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 2 ** 30 if sys.platform == 'darwin' else v / 2 ** 20


def thermal_factor(n_words, beta, chi_rho, tau_im, tol, be):
    """rho^{1/2} = e^{-beta H / 2} as a normalised WordMPO (None at beta = 0).  The step is adjusted to
    beta / (2 n_steps) <= tau_im so that beta is reached exactly; rho.init_loss is the discarded weight."""
    nst, tau = imaginary_steps(beta, tau_im)
    if nst == 0:
        return None
    rho = WordMPO.identity(n_words, be)
    loss = 0.0
    for _ in range(nst):
        loss += rho.sweep('imag', tau, chi_rho, tol) / rho.norm() ** 2
        rho.normalize()
    rho = rho.real_part()                       # Hermitian: real coefficients, continue in real arithmetic
    loss += rho.compress(chi_rho, tol) / rho.norm() ** 2
    rho.normalize()
    rho.init_loss, rho.n_steps, rho.tau = loss, nst, tau
    return rho


def operator_mpo(pat, n_words, rho, chi, tol, be):
    """o_0 rho^{1/2} with o_0 the pattern density centred in the window (rho None: o_0 itself, real)"""
    k0 = hword.place(pat, n_words)
    if rho is None:
        return WordMPO.from_words(n_words, pattern_words(pat, k0), be, chi, tol)
    return rho.times_pattern(pat, k0, chi, tol)


def static_correlation(op, n_words, rho, chi, tol, xmax, be):
    """C(0) = sum_x <o_x, o_0>_beta for a pattern (static, no time evolution)"""
    pat = pattern(op)
    kappa = k_parity(pat)
    if kappa == 1 and rho is not None:
        raise ValueError('K-even operators at beta > 0 need the connected correlation; not supported')
    M = operator_mpo(pat, n_words, rho, chi, tol, be)
    prof = folded_profile(M, M, rho, min(xmax, n_words - 1), kappa)
    return complex(prof[0] + 2 * sum(prof[1:]))


def run_hword(route='split', n=64, beta=0.0, chi=256, tol=1e-12, dt=0.05, tmax=6.0, chi_rho=128, tau_im=0.0125,
              sample_every=1, x0=12.0, vlc=7.3, op=None, cj0=None, be=None, out=None, ckpt=None, ckpt_every=10,
              max_rss_gib=None, save_profile=False, log=print):
    """Run the h-word MPO evolution; returns the result dict (also written to `out` as JSON after every sample).

    n: spins in the window (n - 2 words); chi / chi_rho: bond caps of the evolved operator / thermal factor;
    tol: relative singular-value cutoff; dt: Trotter step of the folded evolution (t = 2 tau); op: custom
    pattern instead of the route operator (then C_J is not derived); cj0: C_J(0) for the split route at beta > 0
    (default: computed from the same thermal factor).  ckpt: checkpoint file (resumed if it exists).
    """
    from ..backend import get_backend
    be = be or get_backend()
    if route not in ('direct', 'split'):
        raise ValueError("route must be 'direct' or 'split'")
    m = n - 2
    opname = op if op is not None else ('J' if route == 'direct' else 'R')
    pat = pattern(opname)
    kappa = k_parity(pat)
    if kappa == 1 and beta > 0:
        raise ValueError(f'op {opname!r} is K-even: at beta > 0 its correlation has a disconnected part '
                         '<o>^2 per shift; only K-odd operators (currents J, R, Q1, J3) are supported')
    if beta < 0:
        raise ValueError('beta must be >= 0')
    lo, hi = min(min(k) for k in pat), max(max(k) for k in pat)
    params = dict(method='h-word MPO, time folding', route=route, op=opname if isinstance(opname, str) else
                  {','.join(map(str, k)): [complex(v).real, complex(v).imag] for k, v in sorted(pat.items())},
                  n=n, beta=beta, chi=chi, tol=tol, dt=dt, chi_rho=chi_rho, tau_im=tau_im, x0=x0, vlc=vlc,
                  sample_every=sample_every, cj0=cj0, version=2)
    steps = int(round(tmax / 2 / dt))
    need = (hi - lo + 1) + vlc * tmax
    if log and m < need:
        log(f'WARNING: window of {m} words is smaller than span + v tmax = {need:.0f}: the light-cone front reaches '
            f'the edges at t = {(m - (hi - lo + 1)) / vlc:.2f} < tmax (watch edge_weight)')
    if log and beta > 0 and n < 32:
        log('WARNING: at beta ~ 1 the thermal factor needs n >= 32 (n = 24 is off by ~3% in static values)')
    t0 = time.time()
    loss = 0.0
    it0 = 0
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            st = pickle.load(f)
        bad = {k: (st['params'].get(k), v) for k, v in params.items() if st['params'].get(k) != v}
        if bad:
            raise ValueError(f'checkpoint {ckpt} has different parameters: {bad}')
        M = WordMPO.from_numpy(st['M'], be)
        rho0 = None if st['rho0'] is None else WordMPO.from_numpy(st['rho0'], be)
        rho_t = None if st['rho_t'] is None else WordMPO.from_numpy(st['rho_t'], be)
        res, loss, it0 = st['res'], st['loss'], st['it'] + 1
        loss_rho = st.get('loss_rho', 0.0)
        res.pop('status', None)
        res.setdefault('backends', [res.get('backend')])
        res['backends'].append(be.name)
        res['backend'] = be.name
        if log:
            log(f'resumed from {ckpt} at t = {2 * st["it"] * dt:.4f}')
    else:
        rho0 = thermal_factor(m, beta, chi_rho, tau_im, tol, be)
        M = operator_mpo(pat, m, rho0, chi, tol, be)
        rho_t = None if rho0 is None else rho0.copy()
        loss_rho = 0.0
        res = dict(params, backend=be.name, samples=[], discarded_init=float(getattr(M, 'init_loss', 0.0)))
        if rho0 is not None:
            res.update(rho_bond=rho0.bond(), discarded_rho_init=rho0.init_loss, n_im_steps=rho0.n_steps,
                       tau_im_eff=rho0.tau)
        if route == 'split' and op is None:
            if cj0 is not None:
                res['C_J0'], res['C_J0_source'] = float(cj0), 'given'
            elif beta == 0:
                res['C_J0'], res['C_J0_source'] = STATIC_BETA0['J'], 'exact'
            else:
                res['C_J0'] = static_correlation('J', m, rho0, chi, tol, int(math.ceil(x0)), be).real
                res['C_J0_source'] = 'static h-word MPO'
        if log:
            log(f'h-word MPO, route={route}, op={opname if isinstance(opname, str) else "custom"}, n={n}, '
                f'beta={beta}, chi={chi}, dt={dt}, backend={be.name}' +
                (f'; rho^1/2 bond {rho0.bond()}' if rho0 is not None else '') +
                (f'; C_J(0) = {res["C_J0"]:.8f} ({res["C_J0_source"]})' if 'C_J0' in res else '') +
                f'  [{time.time() - t0:.1f}s]')
    warned_cap = warned_edge = False
    for it in range(it0, steps + 1):
        if it > 0:
            loss += M.sweep('rot', dt, chi, tol)
            if rho_t is not None:
                loss_rho += rho_t.sweep('rot', dt, chi_rho, tol)
        if it % sample_every == 0 or it == steps:
            tm = time.time()
            t = 2 * it * dt
            want = int(math.ceil(x0 + vlc * t))
            xmax = min(m - 1, want)
            prof = folded_profile(M, M, rho_t, xmax, kappa)
            C = complex(prof[0] + 2 * sum(prof[1:]))
            row = dict(t=t, C=[C.real, C.imag], xmax=xmax, capped=want > m - 1, edge=abs(prof[-1]),
                       discarded=loss, bond=M.bond(), meas_s=0.0, elapsed_s=0.0, rss_gib=rss_gib())
            if rho_t is None:
                row['edge_weight'] = M.edge_weight()
            else:                                    # occupation of the edge words beyond that of rho^{1/2} itself
                row['edge_weight'] = abs(M.edge_weight() - rho_t.edge_weight())
                row.update(rho_bond=rho_t.bond(), discarded_rho=loss_rho)
            row['meas_s'], row['elapsed_s'] = time.time() - tm, time.time() - t0
            if log and row['edge_weight'] > 1e-4 and not warned_edge:
                warned_edge = True
                log(f'WARNING: t = {t:g}: edge_weight = {row["edge_weight"]:.1e} > 1e-4: the operator has reached '
                    f'the window boundary; use a larger --n for later times')
            if log and row['capped'] and row['edge'] > 1e-6 and not warned_cap:
                warned_cap = True
                log(f'WARNING: t = {t:g}: shift sum capped at the window size while the last shift still '
                    f'contributes {row["edge"]:.1e}')
            if save_profile:
                row['profile'] = [[v.real, v.imag] for v in prof]
            res['samples'].append(row)
            if 'C_J0' in res:
                CR0 = res['samples'][0]['C'][0]
                row['C_J'] = C.real - CR0 + res['C_J0']
            elif op is None:
                row['C_J'] = C.real
            if out:
                _write_json(out, res)
            if log:
                log(json.dumps({k: (round(row[k], 10) if isinstance(row[k], float) else row[k])
                                for k in ('t', 'C', 'C_J', 'bond', 'discarded', 'edge_weight', 'xmax', 'elapsed_s')
                                if k in row}))
            if ckpt and (len(res['samples']) % max(1, ckpt_every) == 0 or it == steps):
                _save_ckpt(ckpt, dict(params=params, M=M.to_numpy(), rho0=None if rho0 is None else rho0.to_numpy(),
                                      rho_t=None if rho_t is None else rho_t.to_numpy(), res=res, loss=loss,
                                      loss_rho=loss_rho, it=it))
            if max_rss_gib and row['rss_gib'] > max_rss_gib:
                res['status'] = 'memory_stop'
                if out:
                    _write_json(out, res)
                return res
    res['status'] = 'completed'
    if out:
        _write_json(out, res)
    return res


def _write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def _save_ckpt(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(obj, f)
    os.replace(path + '.tmp', path)


def result_arrays(res, key='C_J'):
    """(t, values) arrays of a run result; key 'C_J', 'C' (real part) or 'Cim'"""
    t = np.array([s['t'] for s in res['samples']])
    if key == 'C':
        return t, np.array([s['C'][0] for s in res['samples']])
    if key == 'Cim':
        return t, np.array([s['C'][1] for s in res['samples']])
    return t, np.array([s[key] for s in res['samples']])
