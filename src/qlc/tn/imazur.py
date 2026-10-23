"""Static Mazur bound on the infinite chain from the purification iMPS: no window (an independent method; it shares with
the window code qlc.tn.mazur only the charge rules, the default family, mazur_value and its cutoff list).

Per-site connected correlation of two translation sums A, B (MPOs W0 + mu W1 of qlc.tn.pauli_mpo):

    <A, B>_beta = (1/3) d_a d_b log Lambda(mu_a, mu_b) |_0,

Lambda = dominant eigenvalue of the cell transfer matrix E(mu_a, mu_b) of the canonical iMPS with A in the layer next
to the bra and B next to the ket (tr prod_x (W0 + mu W1) = sum_k mu^k {k separated words}, so the mixed derivative of
log <psi|.|psi> is the connected <A B>).  At mu = 0 the right fixed point is r = identity on the IDLE x IDLE block, and
E_0 is block triangular in the MPO states (a word never returns to IDLE at mu = 0).  Perturbation theory:

    d_a Lambda = <l|E_a|r>,    d_a d_b Lambda = <l|E_ab|r> + <l|E_a G E_b|r> + <l|E_b G E_a|r>,
    G = Q (1 - E_0)^{-1} Q,  Q = 1 - |r><l|,

with the left fixed point l = (l_0 = S^2 on IDLE x IDLE, in-word blocks from linear solves).  The (IDLE, word) blocks
of l and G E_b|r> are single-layer quantities, solved once per operator; the (word, word) block of l is solved per
pair.  The solves (GMRES) perform all sums over word lengths and separations exactly.  The derivatives are applied in
one pass with flagged MPOs [[W0, W1], [0, W0]] (the flag records that W1 was used).  The Mazur value is formed as in
the reference computation: G normalized to unit diagonal, eigenvalues below rcond x the largest dropped.
"""
import json
import os
import time

import numpy as np

from .. import theory
from .imps import ThermalIMPS
from .mazur import CUTOFF_SCAN, default_charges, mazur_value
from .pauli_mpo import SumMPO, charge_mpo, pattern_mpo

# Fig. 2(a) of the note (lax_drude.tex Table tab:ft-odd): 13 even + 6 odd charges
NOTE_CHARGES = ([('Q1', 'pattern', 'Q1', 1), ('Q2', 'pattern', 'Q2', 1)]
                + [(f'gamma+({p:g})', 'gamma', p, 1) for p in (-0.3, -0.5, -0.7, -0.85, -0.94)]
                + [(f'lambda+({p:g})', 'lambda', p, 1) for p in (-1.06, -1.1, -1.25, -1.6, -2.2, -3.5)]
                + [(f'gamma-({p:g})', 'gamma', p, -1) for p in (-0.2, -0.4, -0.6, -0.75, -0.85, -0.94)])
FAMILIES = {'note': NOTE_CHARGES, 'qlc': default_charges()}


def charge_spec_mpo(spec):
    name, fam, p, par = spec
    return pattern_mpo(p, name) if fam == 'pattern' else charge_mpo(fam, p, par, name)


class Operator:
    """an MPO on the backend: W0, the flagged [[W0, W1], [0, W0]] (state index = flag * D + state), and the
    single-layer data of one CellTM (tm, d1, l, y; set by CellTM.prepare)"""

    def __init__(self, mpo, be):
        self.name, self.D = mpo.name, mpo.D
        Wf = np.zeros((2 * mpo.D, 2 * mpo.D, 2, 2), complex)
        Wf[:mpo.D, :mpo.D] = Wf[mpo.D:, mpo.D:] = mpo.W0
        Wf[:mpo.D, mpo.D:] = mpo.W1
        self.W0, self.Wf = be.asarray(mpo.W0), be.asarray(Wf)
        self.tm = self.d1 = self.l = self.y = None


class CellTM:
    """two-layer cell transfer matrices of a canonical ThermalIMPS (cell = sites 0, 1, 2)"""

    def __init__(self, state, gmres_tol=1e-12):
        be = self.be = state.be
        self.B = [b.reshape(b.shape[0], 2, 2, b.shape[2]) for b in state.B]                   # real
        self.D = self.B[0].shape[0]
        self.l0 = be.asarray(np.diag(state.S[2] ** 2))
        self.gmres_tol = gmres_tol
        self.I0 = be.asarray(SumMPO.identity().W0)
        self.solves = 0

    # ------------------------------------------------------------------ one site / one cell
    def _real_dot(self, x, B, axes, b_first=False):
        """tensordot of a complex x with the real state tensor B as two real products"""
        be = self.be
        if b_first:
            return be.tensordot(B, be.real_part(x), axes) + 1j * be.tensordot(B, x.imag, axes)
        return be.tensordot(be.real_part(x), B, axes) + 1j * be.tensordot(x.imag, B, axes)

    def _site_right(self, x, B, Wa, Wb):
        """x[i,j,a,b] -> sum B[a,s,c,a'] Wb[j,j',t,s] Wa[i,i',u,t] B[b,u,c,b'] x[i',j',a',b']"""
        be = self.be
        t = self._real_dot(x, B, ([3], [2]), b_first=True)        # (a, s, c, i', j', b')
        t = be.tensordot(Wb, t, ([1, 3], [4, 1]))                  # (j, t, a, c, i', b')
        t = be.tensordot(Wa, t, ([1, 3], [4, 1]))                  # (i, u, j, a, c, b')
        return self._real_dot(t, B, ([1, 4, 5], [1, 2, 3]))       # (i, j, a, b)

    def _site_left(self, l, B, Wa, Wb):
        """the transpose map: l[i,j,a,b] -> l'[i',j',a',b']"""
        be = self.be
        t = self._real_dot(l, B, ([2], [0]))                       # (i, j, b, s, c, a')
        t = be.tensordot(t, Wb, ([1, 3], [0, 3]))                  # (i, b, c, a', j', t)
        t = be.tensordot(t, Wa, ([0, 5], [0, 3]))                  # (b, c, a', j', i', u)
        t = self._real_dot(t, B, ([0, 5, 1], [0, 1, 2]))           # (a', j', i', b')
        return be.permute(t, (2, 1, 0, 3))

    def right(self, x, Wa, Wb):
        for k in (2, 1, 0):
            x = self._site_right(x, self.B[k], Wa, Wb)
        return x

    def left(self, l, Wa, Wb):
        for k in (0, 1, 2):
            l = self._site_left(l, self.B[k], Wa, Wb)
        return l

    # ------------------------------------------------------------------ linear algebra helpers
    def dot(self, l, x):
        return complex(self.be.to_numpy((l * x).sum()))

    def _vec(self, da, db, block=None, value=None):
        x = self.be.zeros((da, db, self.D, self.D))
        if block is not None:
            x[block] = value
        return x

    def solve(self, matvec, rhs):
        """GMRES for matvec(x) = rhs on backend arrays of the shape of rhs"""
        from scipy.sparse.linalg import LinearOperator, gmres
        be, shape = self.be, rhs.shape
        b = be.to_numpy(rhs).ravel()
        if not np.any(b):
            return be.zeros(shape)
        n = b.size
        op = LinearOperator((n, n), dtype=complex,
                            matvec=lambda v: be.to_numpy(matvec(be.asarray(v.reshape(shape)))).ravel())
        kw = dict(atol=0.0, restart=min(n, 60), maxiter=500)
        try:
            x, info = gmres(op, b, rtol=self.gmres_tol, **kw)
        except TypeError:                                                   # scipy < 1.12
            x, info = gmres(op, b, tol=self.gmres_tol, **kw)
        self.solves += 1
        if info != 0:
            raise RuntimeError(f'GMRES did not converge (info {info}, size {n})')
        return be.asarray(x.reshape(shape))

    # ------------------------------------------------------------------ single operator and pairs
    def prepare(self, op):
        """left fixed point blocks op.l, d1 = d Lambda / d mu, op.y = G E_d|r> (single layer, op in layer a)"""
        D, W0, I0 = op.D, op.W0, self.I0
        l = self._vec(D, 1, (0, 0), self.l0)
        if D > 1:
            def mv(z):
                x = self._vec(D, 1)
                x[1:] = z
                return z - self.left(x, W0, I0)[1:]
            l[1:] = self.solve(mv, self.left(l, W0, I0)[1:])
        r = self._vec(D, 1, (0, 0), self.be.eye(self.D))
        v = self.right(self._vec(2 * D, 1, (D, 0), self.be.eye(self.D)), op.Wf, I0)[:D]       # E_d |r>
        op.d1 = self.dot(l, v)
        y = self.solve(lambda y: y - self.right(y, W0, I0) + r * self.dot(l, y), v - r * op.d1)
        op.tm, op.l, op.y = self, l[:, 0], (y - r * self.dot(l, y))[:, 0]
        return op

    def pair(self, A, B):
        """<A B> - <A><B> per site (A next to the bra), both prepared by this CellTM"""
        if A.tm is not self or B.tm is not self:
            raise ValueError('operators must be prepared by this CellTM')
        Da, Db = A.D, B.D
        l = self._vec(Da, Db)
        l[:, 0], l[0, :] = A.l, B.l
        if Da > 1 and Db > 1:
            def mv(z):
                x = self._vec(Da, Db)
                x[1:, 1:] = z
                return z - self.left(x, A.W0, B.W0)[1:, 1:]
            l[1:, 1:] = self.solve(mv, self.left(l, A.W0, B.W0)[1:, 1:])
        x = self._vec(2 * Da, 2 * Db, (Da, Db), self.be.eye(self.D))      # flags (1, 1): E_ab |r>
        x[Da, :Db], x[:Da, Db] = B.y, A.y                                   # (1, 0): E_a G E_b|r>; (0, 1)
        d2 = self.dot(l, self.right(x, A.Wf, B.Wf)[:Da, :Db])
        return (d2 - A.d1 * B.d1) / 3


def unit_mazur(G, b, rcond):
    """b^T G^+ b with G normalized to unit diagonal and eigenvalues below rcond * max dropped"""
    if len(b) == 0:
        return 0.0, 0
    d = 1 / np.sqrt(np.diag(G))
    return mazur_value(G * np.outer(d, d), b * d, rcond)


def measure(state, charges, rcond=1e-10, gmres_tol=1e-12):
    """G, b, M and checks for a canonical ThermalIMPS"""
    tm = CellTM(state, gmres_tol)
    be = state.be
    t0 = time.time()
    ops = [tm.prepare(Operator(charge_spec_mpo(c), be)) for c in charges]
    J = tm.prepare(Operator(pattern_mpo('J'), be))
    H = tm.prepare(Operator(pattern_mpo('H'), be))
    K = len(ops)
    G = np.zeros((K, K), complex)
    for a in range(K):
        for c in range(a, K):
            G[a, c] = tm.pair(ops[a], ops[c])
            G[c, a] = np.conj(G[a, c])
    b = np.array([tm.pair(J, o) for o in ops])
    even = [i for i, c in enumerate(charges) if c[3] == 1]
    beta = state.beta
    D = theory.drude_ghd(beta)
    e_ex, hh_ex = theory.thermal_energy(beta)
    Gr, br = G.real, b.real
    M_full, k_full = unit_mazur(Gr, br, rcond)
    M_even, k_even = unit_mazur(Gr[np.ix_(even, even)], br[even], rcond)
    scan = {f'{c:g}': [unit_mazur(Gr, br, c)[0], unit_mazur(Gr[np.ix_(even, even)], br[even], c)[0]]
            for c in CUTOFF_SCAN}
    return dict(beta=beta, M_full=M_full, M_even=M_even, D_GHD=D, ratio_full=M_full / D, ratio_even=M_even / D,
                kept=[k_full, k_even], cutoff_scan=scan, G=Gr.tolist(), b=br.tolist(),
                G_imag=float(np.abs(G.imag).max()), b_imag=float(np.abs(b.imag).max()),
                b_odd=float(max([abs(b[i]) for i in range(K) if i not in even], default=0.0)),
                energy=(H.d1 / 3).real, energy_h0=state.local_energy(), energy_exact=e_ex,
                HH=tm.pair(H, H).real, HH_exact=hh_ex, JJ=tm.pair(J, J).real,
                bond=state.bond(), discarded=state.discarded, solves=tm.solves, meas_s=time.time() - t0)


def run_imazur(betas, chi=64, dtau=0.05, tol=1e-10, charges='note', rcond=1e-10, canon_every=1, gmres_tol=1e-12,
               be=None, log=print, out=None):
    """Mazur bounds at the given betas from one imaginary-time trajectory of the purification iMPS"""
    from ..backend import get_backend
    be = be or get_backend()
    if isinstance(charges, str):
        charges = FAMILIES[charges]
    res = dict(method='Mazur bound from the purification iMPS (3-site cell, transfer-matrix sums)', chi=chi,
               dtau=dtau, tol=tol, canon_every=canon_every, rcond=rcond, gmres_tol=gmres_tol,
               charges=[c[0] for c in charges], backend=be.name, points=[])
    if log:
        log(f'iMPS Mazur: {len(charges)} charges, chi = {chi}, dtau = {dtau} (beta/2 units, S4), backend {be.name}')
    state = ThermalIMPS(be)
    t0 = time.time()
    for beta in sorted(set(betas)):
        ts = time.time()
        state.evolve(max(beta - state.beta, 0.0), dtau, chi, tol, canon_every)
        state.beta = beta                                                   # no drift from repeated additions
        row = measure(state, charges, rcond, gmres_tol)
        row.update(evolve_s=time.time() - ts - row['meas_s'], elapsed_s=time.time() - t0)
        res['points'].append(row)
        if log:
            log(f'beta = {beta:g}: M_full/D_GHD = {row["ratio_full"]:.7f}, M_even/D_GHD = {row["ratio_even"]:.6f} '
                f'(D_GHD = {row["D_GHD"]:.6f}); E/N - exact = {row["energy"] - row["energy_exact"]:.1e}, '
                f'<H,H> - exact = {row["HH"] - row["HH_exact"]:.1e}; bond {row["bond"]}, '
                f'discarded {row["discarded"]:.1e}  [{row["evolve_s"]:.1f} + {row["meas_s"]:.1f} s]')
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out + '.tmp', 'w') as f:
                json.dump(res, f, indent=1)
            os.replace(out + '.tmp', out)
    return res
