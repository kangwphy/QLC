"""Closed forms from the note (energy transport of the periodic FFD chain).

    eps^2(theta) = (1 + 2 cos 2 theta)^3 / (4 cos^2 theta)           lambda band 0 < theta < pi/3 (eps^2 > 0)
    D_GHD(beta)  = int_0^{pi/3} dtheta/pi sech^2(beta eps) (d eps^2 / dtheta)^2            (Eq. 51)
    D(inf) >= M* = 32 - 243 sqrt3 / (10 pi) = 18.6027078413  (Mazur bound of the even charges = D_GHD(0), Eq. 40)
    L(0)         = int_0^{pi/3} dtheta/pi 2 eps^2 Dfrak(theta) = (74 sqrt3 - 95 ln(2 + sqrt3)) / (4 pi) = 0.2435675,
                   Dfrak = 2 sin^3 theta / (cos^2 theta sqrt(4 cos^2 theta - 1))   (Onsager coefficient, Eqs. 57, 75, 76)
    L(beta > 0)  = the same integral with sech^2(beta eps): a MODEL extension (not derived in the note); the h-word
                   TN data of 2026-10 lie about 4.5% above it at beta = 1
    K = 486 sqrt3 / (5 pi) - 52 = 1.5891686  (N (D_N - M*) -> K, Eq. 76)
    Mazur kernels of the even quasi-local charges Q_p^+ (Eq. 36) and the finite-family bound b G^-1 b (Eq. 35).

The tail of C_J(t) - D oscillates at the band-top frequency 2 eps(0) = 3 sqrt3 (period ~1.209).
"""
import math

import numpy as np
from scipy.integrate import quad

SQRT3 = math.sqrt(3.0)
M_STAR = 32 - 243 * SQRT3 / (10 * math.pi)
L_ONSAGER_0 = (74 * SQRT3 - 95 * math.log(2 + SQRT3)) / (4 * math.pi)
K_FINITE_SIZE = 486 * SQRT3 / (5 * math.pi) - 52
W_GAMMA = 243 * SQRT3 / (40 * math.pi) - 3
OMEGA_TAIL = 3 * SQRT3
T_TAIL = 2 * math.pi / OMEGA_TAIL
CJ0_BETA0 = 20.0


def eps2(theta):
    return (1 + 2 * np.cos(2 * theta)) ** 3 / (4 * np.cos(theta) ** 2)


def eps(theta):
    """lambda dispersion (4 cos^2 - 1)^{3/2} / (2 cos), 0 <= theta <= pi/3"""
    c2 = 4 * np.cos(theta) ** 2
    return np.sqrt(np.maximum(c2 - 1, 0.0) ** 3 / c2)


def deps2(theta):
    """d eps^2 / d theta"""
    t = np.tan(theta)
    return -t * (t * t - 3) ** 2 * (t * t + 9) / (2 * (1 + t * t) ** 2)


def velocity(theta):
    """bare velocity w = 2 |eps'| (max 7.24 at theta ~ 0.702)"""
    return np.abs(deps2(theta)) / eps(theta)


def dfrak(theta):
    return 2 * np.sin(theta) ** 3 / (np.cos(theta) ** 2 * np.sqrt(4 * np.cos(theta) ** 2 - 1))


def _sech2(x):
    return 1.0 / np.cosh(x) ** 2


def drude_ghd(beta):
    """D_GHD(beta) (equals M* at beta = 0)"""
    if beta == 0:
        return M_STAR
    f = lambda th: _sech2(beta * eps(th)) * deps2(th) ** 2 / math.pi
    return quad(f, 0, math.pi / 3, limit=400, epsabs=1e-14, epsrel=1e-12)[0]


def onsager_ghd(beta):
    """L(beta) of the energy current; beta = 0: L_ONSAGER_0 (note); beta > 0: model extension (see module doc)"""
    if beta == 0:
        return L_ONSAGER_0
    f = lambda th: _sech2(beta * eps(th)) * 2 * eps(th) ** 2 * dfrak(th) / math.pi
    return quad(f, 0, math.pi / 3, limit=400, epsabs=1e-14, epsrel=1e-12)[0]


# ---------------------------------------------------------------------- Mazur bound at beta = 0 (Eqs. 35, 36)
def _check_p(p):
    if not (p < -1 or -1 < p < 0):
        raise ValueError(f'charge parameter p = {p} outside (-inf, -1) U (-1, 0)')


def mazur_kernel(p, q):
    """<Q_p^+, Q_q^+> per site; gamma family -1 < p < 0, lambda family p < -1 (works with Fractions)"""
    _check_p(p)
    _check_p(q)
    gp, gq = -1 < p < 0, -1 < q < 0
    if gp and gq:
        return p * q * (1 - p * q) / (p * p * q * q - 3 * p * q - p - q)
    if not gp and not gq:
        return (2 * p * q + p + q) / (p * p * q * q - 3 * p * q - p - q)
    g, l = (p, q) if gp else (q, p)
    return g / (l - g)


def mazur_kernel_odd(p, q):
    """<Q_p^{gamma,-}, Q_q^{gamma,-}> per site at beta = 0 (words of odd size): pq sqrt((1+p)(1+q)) / Delta(p, q)"""
    if not (-1 < p < 0 and -1 < q < 0):
        raise ValueError('odd kernel implemented for the gamma family (-1 < p < 0) only')
    return p * q * math.sqrt((1 + p) * (1 + q)) / (p * p * q * q - 3 * p * q - p - q)


def mazur_overlap(p):
    """<J, Q_p^+> per site"""
    _check_p(p)
    return 2 * p if -1 < p < 0 else 2 * (3 * p + 2) / p ** 2


def mazur_bound(ps):
    """b G^-1 b for a finite family of even charges Q_p^+, in exact rational arithmetic.

    The kernel is extremely ill-conditioned (cond ~ 1e15 for 32 charges), so float64 inversion is useless beyond
    ~20 charges; all kernels are rational in p, so the parameters are converted to Fractions."""
    from fractions import Fraction
    ps = [Fraction(p).limit_denominator(10 ** 12) if not isinstance(p, Fraction) else p for p in ps]
    n = len(ps)
    G = [[mazur_kernel(p, q) for q in ps] for p in ps]
    b = [mazur_overlap(p) for p in ps]
    x = list(b)
    A = [row[:] for row in G]
    for c in range(n):                                   # Gaussian elimination, partial pivoting on |entries|
        piv = max(range(c, n), key=lambda r: abs(A[r][c]))
        if A[piv][c] == 0:
            raise ValueError('singular kernel (repeated parameters?)')
        A[c], A[piv] = A[piv], A[c]
        x[c], x[piv] = x[piv], x[c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            if f:
                for k in range(c, n):
                    A[r][k] -= f * A[c][k]
                x[r] -= f * x[c]
    for c in range(n - 1, -1, -1):
        x[c] = (x[c] - sum(A[c][k] * x[k] for k in range(c + 1, n))) / A[c][c]
    return float(sum(bi * xi for bi, xi in zip(b, x)))


def mazur_family(n):
    """the note's family p = -k/(n+1), -(n+1)/k, k = 1..n (Sec. 4.4): 18.18747 (n = 4), 18.59992 (8),
    18.6026931 (12), 18.602707768 (16), 18.6027078406 (24) -> M* = 18.6027078413"""
    from fractions import Fraction
    ps = [Fraction(-k, n + 1) for k in range(1, n + 1)] + [Fraction(-(n + 1), k) for k in range(1, n + 1)]
    return mazur_bound(ps)
