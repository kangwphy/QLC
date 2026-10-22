"""Static Mazur bound of the Drude weight from MPS representations of the quasi-local charges (no time evolution).

    D(beta) >= M(beta) = sum_ab <J, Q_a>_beta (G^+)_ab <Q_b, J>_beta,     G_ab = <Q_a, Q_b>_beta        (per site)

with the even parts Q^{lambda,+}_p, Q^{gamma,+}_p (and optionally Q1), plus the odd parts Q^{gamma,-}_p which couple to
the even ones at beta > 0 (note, App. E).  All operators are Hermitian and odd under time reversal, so <Q>_beta = 0
and the connected and plain correlations agree.

Per-site correlations in an open window of m words (bulk density at y0, translation sum over the window):

    G_ab = omega(rho Q_a q_{b,y0}) / omega(rho),        <J, Q_a> = omega(rho J q_{a,y0}) / omega(rho),

rho = e^{-beta H}.  Everything is in the basis of ordered products h_S, where left multiplication is a real sign:
rho has real coefficients and is built by imaginary-time sweeps; one trajectory gives every requested beta.
omega(rho A B) = sum_U rho_U (AB)_U (-1)^{a(U)} with (AB)_U = sum_{S xor T = U} a_S b_T sign(S, T),
sign(S, T) = (-1)^{#(s in S, t in T, s - t in {1, 2})}; one environment sweep with the last two bits of T and U
(16 states) gives it at cost O(m chi^2 D_A D_B).

Truncations: the charge words are cut at the window (choose m so that the tails are negligible: see
`beta0_check`, which compares the window result with the closed-form kernels), the thermal factor by chi / tau.
"""
import math
import time

import numpy as np

from .. import theory
from ..model import pattern
from .charges import charge_part_mps, pattern_mps
from .wordmpo import WordMPO


def _as_complex(be, a):
    return a if be.is_complex(a) else be.to_complex(a)


def omega_product(rho, A, B):
    """omega(rho A B) for word MPS rho, A, B in the h_S basis (rho None: identity)"""
    be = A.be
    if rho is None:
        rho = WordMPO.identity(len(A.A) - 2 * WordMPO.PAD, be)
    E = be.zeros((16, 1, 1, 1))
    E[0, 0, 0, 0] = 1.0
    sgn = {}
    for s in (0, 1):
        for t in (0, 1):
            u = s ^ t
            arr = np.zeros((2, 2, 2, 2))
            for t2 in (0, 1):
                for t1 in (0, 1):
                    for u2 in (0, 1):
                        for u1 in (0, 1):
                            arr[t2, t1, u2, u1] = (-1) ** (s * (t1 + t2) + u * (u1 + u2))
            sgn[s, t] = be.asarray(arr.reshape(2, 2, 2, 2, 1, 1, 1))
    for i in range(len(A.A)):
        R = _as_complex(be, rho.A[i])
        Ai, Bi = _as_complex(be, A.A[i]), _as_complex(be, B.A[i])
        r2, a2, b2 = R.shape[2], Ai.shape[2], Bi.shape[2]
        new = be.zeros((2, 2, 2, 2, r2, a2, b2))
        for s in (0, 1):
            for t in (0, 1):
                u = s ^ t
                Y = E @ Bi[:, t, :]                                   # (16, r, a, b')
                Y = be.permute(Y, (0, 1, 3, 2)) @ Ai[:, s, :]         # (16, r, b', a')
                Y = be.permute(Y, (0, 2, 3, 1)) @ R[:, u, :]          # (16, b', a', r')
                Y = be.permute(Y, (0, 3, 2, 1)).reshape(2, 2, 2, 2, r2, a2, b2)
                Y = (Y * sgn[s, t]).sum(2).sum(0)                     # sum over t2, u2 -> (t1, u1, ...)
                new[:, t, :, u] = new[:, t, :, u] + Y
        E = new.reshape(16, r2, a2, b2)
    return complex(be.to_numpy(E.sum(0))[0, 0, 0])


def _signs(be):
    out = {}
    for s in (0, 1):
        for t in (0, 1):
            u = s ^ t
            arr = np.zeros((2, 2, 2, 2))
            for t2 in (0, 1):
                for t1 in (0, 1):
                    for u2 in (0, 1):
                        for u1 in (0, 1):
                            arr[t2, t1, u2, u1] = (-1) ** (s * (t1 + t2) + u * (u1 + u2))
            out[s, t] = arr
    return out


def _stack(be, mps_list, i):
    """site i of several MPS, zero-padded to common bonds: (K, Dl, 2, Dr)"""
    ts = [m.A[i] for m in mps_list]
    dl, dr = max(t.shape[0] for t in ts), max(t.shape[2] for t in ts)
    out = be.zeros((len(ts), dl, 2, dr))
    for k, t in enumerate(ts):
        out[k, :t.shape[0], :, :t.shape[2]] = _as_complex(be, t)
    return out


def omega_matrix(rho, As, Bs):
    """Omega[a, b] = omega(rho A_a B_b) for lists of h_S-basis MPS, in one environment sweep (rho None: identity).

    Environment E[a, b, t_{j-2}, t_{j-1}, u_{j-2}, u_{j-1}, rho bond, A bond, B bond] (t: bits of B, u = s xor t: bits
    of the product).  At a site with bits (s, t) the weight is (-1)^{s (t_{j-1} + t_{j-2})} (product sign) times
    (-1)^{u (u_{j-1} + u_{j-2})} (omega); the oldest history bits are summed out first (they do not enter A, B), then
    A and B are applied as batched matmuls and the thermal factor is contracted once per u."""
    be = As[0].be
    L = len(As[0].A)
    if rho is None:
        rho = WordMPO.identity(L - 2 * WordMPO.PAD, be)
    KA, KB = len(As), len(Bs)
    E = be.zeros((KA, KB, 2, 2, 2, 2, 1, 1, 1))
    E[:, :, 0, 0, 0, 0, 0, 0, 0] = 1.0
    sgn = {}
    for s_ in (0, 1):
        for u_ in (0, 1):
            arr = np.array([[(-1) ** (s_ * t1 + u_ * u1) for u1 in (0, 1)] for t1 in (0, 1)], dtype=float)
            sgn[s_, u_] = be.asarray(arr.reshape(1, 1, 2, 2, 1, 1, 1), complex)
    for i in range(L):
        R = rho.A[i]
        A_st, B_st = _stack(be, As, i), _stack(be, Bs, i)           # (K, Dl, 2, Dr)
        r1, r2 = R.shape[0], R.shape[2]
        x0, xa = A_st.shape[1], A_st.shape[3]
        y0, yb = B_st.shape[1], B_st.shape[3]
        e00, e01 = E[:, :, 0, :, 0], E[:, :, 0, :, 1]               # [t2, u2] slices: (KA, KB, t1, u1, r, x, y)
        e10, e11 = E[:, :, 1, :, 0], E[:, :, 1, :, 1]
        hs = {(0, 0): e00 + e01 + e10 + e11, (0, 1): e00 - e01 + e10 - e11,
              (1, 0): e00 + e01 - e10 - e11, (1, 1): e00 - e01 - e10 + e11}      # sum (-1)^{s t2 + u u2}
        out_u = {}
        for u in (0, 1):
            per_t = []
            for t in (0, 1):
                s = u ^ t
                F = hs[s, u] * sgn[s, u]                                          # (KA, KB, t1, u1, r, x, y)
                F = be.permute(F, (1, 0, 2, 3, 4, 5, 6)).reshape(KB, KA * 4 * r1 * x0, y0) @ B_st[:, :, t, :]
                F = be.permute(F.reshape(KB, KA, 2, 2, r1, x0, yb), (1, 0, 2, 3, 4, 6, 5))
                F = (F.reshape(KA, KB * 4 * r1 * yb, x0) @ A_st[:, :, s, :]).reshape(KA, KB, 2, 2, r1, yb, xa)
                per_t.append(F)
            acc = be.concat([per_t[0][:, :, :, None], per_t[1][:, :, :, None]], 3)  # (KA,KB,t2'=t1,t1'=t,u2'=u1,r,Y,X)
            Ru = R[:, u, :]
            if be.is_complex(Ru):
                Zu = be.tensordot(acc, Ru, ([5], [0]))                             # (..., Y, X, r2)
            else:                                                                   # real rho: two real GEMMs
                Zu = be.tensordot(acc.real, Ru, ([5], [0])) + 1j * be.tensordot(acc.imag, Ru, ([5], [0]))
            out_u[u] = be.permute(Zu, (0, 1, 2, 3, 4, 7, 6, 5))                    # (KA,KB,t2',t1',u2',r2,X,Y)
        E = be.concat([out_u[0][:, :, :, :, :, None], out_u[1][:, :, :, :, :, None]], 5)
    out = E.reshape(KA, KB, -1).sum(2)
    return be.to_numpy(out)


SUZUKI_S = 1 / (2 - 2 ** (1 / 3))


def thermal_trajectory(m, betas, chi, tau, tol, be, log=None, order=4):
    """yield (beta, rho, discarded) for increasing betas; rho = e^{-beta H} / |.| in the h_S basis (real), each
    segment uses n steps of size (beta_k - beta_{k-1}) / n <= tau.  order 2: one palindromic sweep per step (error
    O(tau^2), which biases M upwards by ~0.25% at beta = 1, tau = 0.05); order 4 (default): Suzuki composition
    S2(s tau) S2((1-2s) tau) S2(s tau), s = 1/(2 - 2^{1/3}), error O(tau^4)."""
    if order not in (2, 4):
        raise ValueError('order must be 2 or 4')
    rho = WordMPO.identity(m, be)
    b_cur, loss = 0.0, 0.0
    for b in sorted(betas):
        if b < 0:
            raise ValueError('beta must be >= 0')
        if b > b_cur:
            n = max(1, int(math.ceil((b - b_cur) / tau - 1e-9)))
            dtau = (b - b_cur) / n
            subs = [dtau] if order == 2 else [SUZUKI_S * dtau, (1 - 2 * SUZUKI_S) * dtau, SUZUKI_S * dtau]
            t0 = time.time()
            for _ in range(n):
                for d in subs:
                    loss += rho.sweep('imag_h', d, chi, tol) / rho.norm() ** 2
                    rho.normalize()
            if log:
                log(f'  rho(beta={b:g}): {n} steps of {dtau:.5f}, bond {rho.bond()}, {time.time() - t0:.1f}s')
            b_cur = b
        yield b, (None if b == 0 else rho), loss


def default_charges(lam=(-1.25, -1.6, -2.2, -3.5, -6.0, -12.0), gam=(-0.2, -0.35, -0.5, -0.65, -0.8),
                    gam_odd=None, local=('Q1',)):
    """charge specs: (name, family, p, parity) or (name, 'pattern', name, +1); gam_odd defaults to gam"""
    gam_odd = gam if gam_odd is None else gam_odd
    out = [(n, 'pattern', n, 1) for n in local]
    out += [(f'lambda+({p:g})', 'lambda', p, 1) for p in lam]
    out += [(f'gamma+({p:g})', 'gamma', p, 1) for p in gam]
    out += [(f'gamma-({p:g})', 'gamma', p, -1) for p in gam_odd]
    return out


def build_operators(charges, m, y0, be):
    """(window sums, densities at y0) for the charges and the energy current J"""
    sums, dens = [], []
    for name, fam, p, par in charges:
        if fam == 'pattern':
            sums.append(pattern_mps(p, m, be, 'sum'))
            dens.append(pattern_mps(p, m, be, 'density', y0))
        else:
            sums.append(charge_part_mps(fam, p, par, m, be, 'sum'))
            dens.append(charge_part_mps(fam, p, par, m, be, 'density', y0))
    J = (pattern_mps('J', m, be, 'sum'), pattern_mps('J', m, be, 'density', y0))
    return sums, dens, J


def mazur_matrices(rho, sums, dens, J):
    """G (K x K) and b (K) per site, with symmetry / reality diagnostics"""
    I = WordMPO.identity(len(sums[0].A) - 2 * WordMPO.PAD, sums[0].be)
    Z = omega_matrix(rho, [I], [I])[0, 0].real
    G = omega_matrix(rho, sums, dens) / Z
    bJ = omega_matrix(rho, [J[0]], dens)[0] / Z
    bQ = np.conj(omega_matrix(rho, sums, [J[1]])[:, 0] / Z)
    diag = dict(G_asym=float(np.abs(G - G.conj().T).max()), G_imag=float(np.abs(G.imag).max()),
                b_imag=float(np.abs(bJ.imag).max()), b_twoway=float(np.abs(bJ - bQ).max()))
    Gs = ((G + G.conj().T) / 2).real
    return Gs, ((bJ + bQ) / 2).real, diag


def mazur_value(G, b, rcond=1e-10):
    """b G^+ b with eigenvalues below rcond * max discarded; returns (value, number of kept eigenvalues).
    Discarding directions is conservative: the result is the Mazur bound of a sub-span of the charges."""
    if len(b) == 0:
        return 0.0, 0
    w, V = np.linalg.eigh(G)
    keep = w > rcond * w.max()
    c = V[:, keep].T @ b
    return float(np.sum(c ** 2 / w[keep])), int(keep.sum())


CUTOFF_SCAN = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12)


def noise_cutoff(G, noise, rcond=1e-10, factor=1.0):
    """relative eigenvalue cutoff above the estimated absolute error `noise` of G (directions with eigenvalues near
    the noise level are not determined by the data)"""
    w_max = float(np.linalg.eigvalsh(G).max())
    return max(rcond, factor * noise / w_max)


def beta0_check(charges, sums, dens, J):
    """window kernels at beta = 0 against the closed forms (note Eqs. 35, 36, 63).  Returns the max relative errors,
    the max absolute error of G (used as the noise level of the window truncation) and the window Mazur value of
    the even lambda / gamma charges against the exact rational one; None if there are no quasi-local charges."""
    idx = [i for i, c in enumerate(charges) if c[1] != 'pattern']
    if not idx:
        return None
    G, b, _ = mazur_matrices(None, sums, dens, J)
    err_G = err_b = abs_G = 0.0
    for i in idx:
        _, fi, pi, si = charges[i]
        if si == 1:
            err_b = max(err_b, abs(b[i] - theory.mazur_overlap(pi)) / abs(theory.mazur_overlap(pi)))
        for j in idx:
            _, fj, pj, sj = charges[j]
            if si != sj:
                exact = 0.0
            elif si == 1:
                exact = theory.mazur_kernel(pi, pj)
            else:
                exact = theory.mazur_kernel_odd(pi, pj) if fi == fj else 0.0
            abs_G = max(abs_G, abs(G[i, j] - exact))
            err_G = max(err_G, abs(G[i, j] - exact) / max(abs(exact), abs(G[i, i] * G[j, j]) ** 0.5))
    out = dict(rel_err_G=float(err_G), rel_err_b=float(err_b), abs_err_G=float(abs_G))
    ev = [i for i in idx if charges[i][3] == 1]
    if ev:
        Mw, _ = mazur_value(G[np.ix_(ev, ev)], b[ev], 1e-14)
        out.update(M_window=Mw, M_exact=theory.mazur_bound([charges[i][2] for i in ev]))
    return out


def run_mazur(betas, m=None, y0=None, chi=128, tau=0.05, tol=1e-10, charges=None, rcond=1e-10, order=4, be=None,
              log=print, out=None, tail=200):
    """Mazur bounds at the given betas from one imaginary-time trajectory; returns the result dict.

    y0 (left buffer, default max(24, 16 beta_max)): the bulk densities must be outside the thermal boundary layer of
    the open window, which grows with beta.  m (default y0 + tail): room for the charge tails to the right.
    The eigenvalue cutoff of G is raised above the estimated noise of G (window tails at beta = 0, asymmetry of G
    at beta > 0); M is also reported for a scan of cutoffs."""
    import json
    import os
    from ..backend import get_backend
    be = be or get_backend()
    charges = charges or default_charges()
    bmax = max(betas)
    y0 = int(math.ceil(max(24, 16 * bmax))) if y0 is None else int(y0)
    m = y0 + tail if m is None else int(m)
    pats = [pattern('J')] + [pattern(c[2]) for c in charges if c[1] == 'pattern']
    span = max(max(max(k) for k in pt) - min(min(k) for k in pt) for pt in pats)
    if y0 < 0 or y0 + span >= m:
        raise ValueError(f'need 0 <= y0 and y0 + {span} < m (y0 = {y0}, m = {m})')
    t0 = time.time()
    sums, dens, J = build_operators(charges, m, y0, be)
    even = [i for i, c in enumerate(charges) if c[3] == 1]
    res = dict(method='Mazur bound from charge MPS (h_S basis, open window)', m=m, y0=y0, chi=chi, tau=tau, tol=tol,
               order=order, rcond=rcond, charges=[c[0] for c in charges], backend=be.name, points=[])
    if log:
        log(f'Mazur: {len(charges)} charges, window m = {m} words, y0 = {y0}, chi = {chi}, tau = {tau} (order {order}), '
            f'backend {be.name}')
        if m - y0 < 200 and charges is not None:
            log(f'  note: m - y0 = {m - y0} < 200; slowly decaying charges (p near -1) may be cut (see beta0_check)')
        if y0 < 16 * bmax:
            log(f'  WARNING: y0 = {y0} < 16 beta_max = {16 * bmax:g}: the bulk densities may feel the window edge')
    chk = beta0_check(charges, sums, dens, J)
    res['beta0_check'] = chk
    if log and chk:
        log('  beta = 0 window vs closed forms: ' + ', '.join(f'{k} = {v:.3e}' if 'err' in k else f'{k} = {v:.10f}'
                                                            for k, v in chk.items()))
    noise0 = chk['abs_err_G'] if chk else 0.0
    for b, rho, loss in thermal_trajectory(m, betas, chi, tau, tol, be, log, order):
        tm = time.time()
        G, bv, diag = mazur_matrices(rho, sums, dens, J)
        w = np.linalg.eigvalsh(G)
        noise = max(noise0, diag['G_asym'])
        rc = noise_cutoff(G, noise, rcond)
        M_full, k_full = mazur_value(G, bv, rc)
        Ge = G[np.ix_(even, even)]
        M_even, k_even = mazur_value(Ge, bv[even], max(rcond, noise_cutoff(Ge, noise, rcond)) if even else rc)
        D = theory.drude_ghd(b)
        scan = {f'{c:g}': [mazur_value(G, bv, c)[0], mazur_value(Ge, bv[even], c)[0]] for c in CUTOFF_SCAN}
        row = dict(beta=b, M_full=M_full, M_even=M_even, D_GHD=D, ratio_full=M_full / D, ratio_even=M_even / D,
                   cutoff=rc, kept=[k_full, k_even], cutoff_scan=scan, eig_min_rel=float(w.min() / w.max()),
                   noise=noise, rho_bond=None if rho is None else rho.bond(), rho_discarded=loss,
                   G=G.tolist(), b=bv.tolist(), meas_s=time.time() - tm, elapsed_s=time.time() - t0, **diag)
        res['points'].append(row)
        if log:
            log(f'beta = {b:g}: M_full = {M_full:.8f} (/D_GHD = {M_full / D:.6f}), M_even = {M_even:.8f} '
                f'(/D_GHD = {M_even / D:.6f}), D_GHD = {D:.8f}; cutoff {rc:.1e} (kept {k_full}/{len(bv)}), '
                f'G_asym {diag["G_asym"]:.1e}, b_twoway {diag["b_twoway"]:.1e}  [{row["meas_s"]:.1f}s]')
            bmaxabs = max(abs(bv).max(), 1e-300)
            if diag['b_twoway'] > 1e-5 * bmaxabs or diag['G_asym'] > 1e-6 * w.max():
                log(f'  WARNING: G / b not converged in the window (G_asym {diag["G_asym"]:.1e}, b_twoway '
                    f'{diag["b_twoway"]:.1e}): increase y0 and m')
            if M_even > M_full * (1 + 1e-9):
                log('  WARNING: M_even > M_full: inconsistent Gram matrix (window or truncation)')
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out + '.tmp', 'w') as f:
                json.dump(res, f, indent=1)
            os.replace(out + '.tmp', out)
    return res
