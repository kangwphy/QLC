"""Drude-weight estimators from a current autocorrelation C(t), and extrapolations.

D = lim_{tau -> inf} (1/tau) int_0^tau C(t) dt.  On the accessible times C_J(t) - D is an oscillation at the
band-top frequency 3 sqrt3 with an amplitude ~ t^-3 (note, App. F), so

* running_mean     (1/tau) int_0^tau C        biased by  int_0^tau (C - D) / tau  ~  L / (2 tau)   (slow)
* plateau          mean of C over an integer number of tail periods ending at t_max      (default estimator)
* tail_fit         least squares  D + (a cos wt + b sin wt) / t^p  on t >= t_min

and the spread between them is a rough error bar.  `richardson` removes the O(dt^2) Trotter error from two runs
at dt and dt/2; `extrapolate_size` fits D_N in 1/N (per residue class N mod 3, as in the note, App. E).
"""
import math

import numpy as np

from .theory import OMEGA_TAIL, T_TAIL


def running_integral(t, f):
    t, f = np.asarray(t, float), np.asarray(f, float)
    return np.concatenate([[0.0], np.cumsum(np.diff(t) * (f[1:] + f[:-1]) / 2)])


def running_mean(t, C):
    t = np.asarray(t, float)
    I = running_integral(t, C)
    out = np.full_like(t, np.nan)
    out[t > 0] = I[t > 0] / t[t > 0]
    return out


def window_mean(t, C, a, b, n=2001):
    tt = np.linspace(a, b, n)
    y = np.interp(tt, t, C)
    return float(np.trapezoid(y, tt) / (b - a)) if hasattr(np, 'trapezoid') else float(np.trapz(y, tt) / (b - a))


def plateau(t, C, t_min=1.5, t_max=None, period=T_TAIL):
    """mean of C over the longest window [t_max - n period, t_max] with t_max - n period >= t_min, n >= 1"""
    t, C = np.asarray(t, float), np.asarray(C, float)
    b = float(t[-1] if t_max is None else t_max)
    n = int(math.floor((b - t_min) / period + 1e-9))
    if n < 1:
        return dict(D=float('nan'), window=None, periods=0)
    a = b - n * period
    return dict(D=window_mean(t, C, a, b), window=(a, b), periods=n)


def tail_fit(t, C, t_min=1.5, omega=OMEGA_TAIL, power=3.0, extra=()):
    """least squares C = D + (a cos wt + b sin wt) / t^power + sum_k c_k t^-k (k in extra) on t >= t_min"""
    t, C = np.asarray(t, float), np.asarray(C, float)
    m = (t >= t_min) & (t > 0)
    tt, y = t[m], C[m]
    cols = [np.ones_like(tt), np.cos(omega * tt) / tt ** power, np.sin(omega * tt) / tt ** power]
    cols += [tt ** (-k) for k in extra]
    A = np.stack(cols, 1)
    if len(tt) < A.shape[1] + 1:
        return dict(D=float('nan'), rms=float('nan'), n=len(tt))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rms = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return dict(D=float(coef[0]), amp=float(np.hypot(coef[1], coef[2])), rms=rms, n=int(len(tt)))


def drude_estimates(t, C, t_min=1.5, reference=None):
    """all estimators; reference (e.g. theory.drude_ghd(beta)) adds relative deviations"""
    t, C = np.asarray(t, float), np.asarray(C, float)
    out = dict(t_max=float(t[-1]), last=float(C[-1]), running_mean=float(running_mean(t, C)[-1]),
               plateau=plateau(t, C, t_min), tail_fit=tail_fit(t, C, t_min))
    vals = [out['plateau']['D'], out['tail_fit']['D']]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        out['D'] = float(np.mean(vals))
        out['spread'] = float(np.ptp(vals)) if len(vals) > 1 else float('nan')
    if reference is not None:
        out['reference'] = float(reference)
        for k in ('plateau', 'tail_fit'):
            if np.isfinite(out[k]['D']):
                out[k]['rel_dev'] = out[k]['D'] / reference - 1
    return out


def richardson(t1, C1, t2, C2, order=2, ratio=2.0):
    """(r^p C(dt/r) - C(dt)) / (r^p - 1) on the common times; run 1 has dt, run 2 has dt / ratio"""
    t1, C1, t2, C2 = map(lambda a: np.asarray(a, float), (t1, C1, t2, C2))
    common = [x for x in t1 if np.min(np.abs(t2 - x)) < 1e-9]
    i1 = [int(np.argmin(np.abs(t1 - x))) for x in common]
    i2 = [int(np.argmin(np.abs(t2 - x))) for x in common]
    f = ratio ** order
    return np.array(common), (f * C2[i2] - C1[i1]) / (f - 1)


def extrapolate_size(Ns, D, deg=2, by_residue=3, last=4):
    """polynomial fit of D_N in 1/N; per residue class N mod by_residue (0: one fit) using the `last` largest N"""
    Ns, D = np.asarray(Ns), np.asarray(D, float)
    classes = {0: np.arange(len(Ns))} if not by_residue else {r: np.nonzero(Ns % by_residue == r)[0]
                                                              for r in range(by_residue)}
    out = {}
    for r, idx in classes.items():
        idx = idx[np.argsort(Ns[idx])][-last:]
        if len(idx) <= deg:
            continue
        c = np.polyfit(1.0 / Ns[idx], D[idx], deg)
        out[int(r)] = dict(D_inf=float(c[-1]), Ns=Ns[idx].tolist(), coef=c.tolist())
    return out
