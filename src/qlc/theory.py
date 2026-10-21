"""Closed forms from the note (energy transport of the periodic FFD chain).

    eps^2(theta) = (1 + 2 cos 2 theta)^3 / (4 cos^2 theta)           lambda band 0 < theta < pi/3 (eps^2 > 0)
    D_GHD(beta)  = int_0^{pi/3} dtheta/pi sech^2(beta eps) (d eps^2 / dtheta)^2            (Eq. 51)
    D(inf) >= M* = 32 - 243 sqrt3 / (10 pi) = 18.6027078413  (Mazur bound of the even charges = D_GHD(0), Eq. 40)
    L(beta)      = int_0^{pi/3} dtheta/pi sech^2(beta eps) 2 eps^2 Dfrak(theta),
                   Dfrak = 2 sin^3 theta / (cos^2 theta sqrt(4 cos^2 theta - 1))   (Onsager coefficient, Eqs. 57, 75)
    L(0) = (74 sqrt3 - 95 ln(2 + sqrt3)) / (4 pi) = 0.2435675,   K = 486 sqrt3 / (5 pi) - 52 = 1.5891686 (N D_N - N M*)
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
    """L(beta) of the energy current (beta = 0: L_ONSAGER_0)"""
    if beta == 0:
        return L_ONSAGER_0
    f = lambda th: _sech2(beta * eps(th)) * 2 * eps(th) ** 2 * dfrak(th) / math.pi
    return quad(f, 0, math.pi / 3, limit=400, epsabs=1e-14, epsrel=1e-12)[0]


# ---------------------------------------------------------------------- Mazur bound at beta = 0 (Eqs. 35, 36)
def mazur_kernel(p, q):
    """<Q_p^+, Q_q^+> per site; gamma family -1 < p < 0, lambda family p < -1"""
    gp, gq = -1 < p < 0, -1 < q < 0
    if gp and gq:
        return p * q * (1 - p * q) / (p * p * q * q - 3 * p * q - p - q)
    if not gp and not gq:
        return (2 * p * q + p + q) / (p * p * q * q - 3 * p * q - p - q)
    g, l = (p, q) if gp else (q, p)
    return g / (l - g)


def mazur_overlap(p):
    """<J, Q_p^+> per site"""
    return 2 * p if -1 < p < 0 else 2 * (3 * p + 2) / p ** 2


def mazur_bound(ps, rcond=1e-13):
    """b G^+ b for a finite family of even charges Q_p^+"""
    G = np.array([[mazur_kernel(p, q) for q in ps] for p in ps])
    b = np.array([mazur_overlap(p) for p in ps])
    return float(b @ np.linalg.pinv(G, rcond=rcond) @ b)


def mazur_family(n):
    """the note's family p = -k/(n+1), -(n+1)/k, k = 1..n: 18.18747 (n = 4), 18.59992 (8), ... -> M*"""
    ps = [-k / (n + 1) for k in range(1, n + 1)] + [-(n + 1) / k for k in range(1, n + 1)]
    return mazur_bound(ps)
