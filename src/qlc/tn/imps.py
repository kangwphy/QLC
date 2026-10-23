"""Thermal state of the infinite FFD chain as an infinite purification MPS (iMPS) with a three-site unit cell.

|psi(beta)> = e^{-beta H/2} |Phi>, |Phi> = Bell pairs between each spin and an ancilla (local dimension 4, index
2 p + a, as in qlc.tn.purification).  The terms h_j with j = r mod 3 commute and act on disjoint triples, so the
evolution is split into layers L_r(t) = prod_{j = r mod 3} (cosh t - sinh t h_j);
    S2(t) = L0(t/2) L1(t/2) L2(t) L1(t/2) L0(t/2),    S4(t) = S2(p t)^2 S2((1 - 4p) t) S2(p t)^2,  p = 1/(4 - 4^{1/3})
(the scheme of the reference computation, lax_drude.tex Methods (a)); adjacent layers on one sublattice are merged.
The step is in units of beta/2: n = ceil(dbeta / (2 dtau)) steps of dbeta / (2n).

Gauge: right-canonical site tensors B_k (shape (D_l, 4, D_r)) and Schmidt values S_k on the bond to the right of
site k.  A layer is applied with a Hastings-type update (no inverse Schmidt values); every `canon_every` S4 steps and
at the end of each evolution the exact canonical form is restored from the dominant fixed points of the cell
transfer matrix (the Schmidt values come from eigh of the left fixed point, so values below ~1e-8 of the largest are
not resolved; the next layer truncates them).  All tensors are real.  `discarded` accumulates the truncated weight
relative to the norm at each truncation.
"""
import math

import numpy as np

from ..model import I2, X2, Z2
from .mps import imaginary_steps, keep_rank

P4 = 1 / (4 - 4 ** (1 / 3))
H3 = np.kron(np.kron(np.kron(Z2, I2), np.kron(Z2, I2)), np.kron(X2, I2)).real      # h_0 on a triple, (2p + a)^3


def s4_schedule(t):
    """[(sublattice r, time)] of one S4(t) step, adjacent layers on the same sublattice merged"""
    out = []
    for c in (P4, P4, 1 - 4 * P4, P4, P4):
        for r, x in ((0, c * t / 2), (1, c * t / 2), (2, c * t), (1, c * t / 2), (0, c * t / 2)):
            if out and out[-1][0] == r:
                out[-1] = (r, out[-1][1] + x)
            else:
                out.append((r, x))
    return out


def _dominant(apply, D, v0, be):
    """dominant eigenpair of a real linear map on D x D matrices; the eigenvector symmetrized, trace 1"""
    if D <= 12:
        M = np.stack([be.to_numpy(apply(be.asarray(e.reshape(D, D), float))).ravel() for e in np.eye(D * D)], 1)
        w, V = np.linalg.eig(M)
        k = int(np.argmax(np.abs(w)))
        lam, v = w[k].real, V[:, k].real
    else:
        from scipy.sparse.linalg import LinearOperator, eigs
        op = LinearOperator((D * D, D * D), dtype=float,
                            matvec=lambda x: be.to_numpy(apply(be.asarray(x.reshape(D, D), float))).ravel())
        w, V = eigs(op, k=1, which='LM', v0=v0.ravel(), tol=1e-14, maxiter=100000)
        lam, v = w[0].real, V[:, 0].real
    v = v.reshape(D, D)
    v = (v + v.T) / 2
    return v / np.trace(v), lam


class ThermalIMPS:
    """|psi(beta)> = e^{-beta H/2}|Phi> on the infinite chain; cell = sites 0, 1, 2 (spins 3n, 3n+1, 3n+2)"""

    def __init__(self, be):
        bell = np.zeros((1, 4, 1))
        bell[0, 0, 0] = bell[0, 3, 0] = 1 / math.sqrt(2)
        self.be = be
        self.B = [be.asarray(bell, float) for _ in range(3)]
        self.S = [np.ones(1) for _ in range(3)]
        self.beta = 0.0
        self.discarded = 0.0
        self._gates = {}

    def bond(self):
        return max(len(s) for s in self.S)

    def cell(self):
        """B_0 B_1 B_2 as (D, 64, D)"""
        be = self.be
        c = be.tensordot(be.tensordot(self.B[0], self.B[1], ([2], [0])), self.B[2], ([3], [0]))
        return c.reshape(c.shape[0], 64, c.shape[-1])

    def _gate(self, t):
        if t not in self._gates:
            G = math.cosh(t) * np.eye(64) - math.sinh(t) * H3
            self._gates[t] = self.be.asarray(G.reshape([4] * 6), float)
        return self._gates[t]

    # ------------------------------------------------------------------ one layer: Hastings-type update
    def _layer(self, r, t, chi, tol):
        """gates on the triples (r, r+1, r+2) of all cells; the bond left of site r is kept"""
        be = self.be
        k0, k1, k2 = r, (r + 1) % 3, (r + 2) % 3
        Dl, Dr = self.B[k0].shape[0], self.B[k2].shape[2]
        th = be.tensordot(be.tensordot(self.B[k0], self.B[k1], ([2], [0])), self.B[k2], ([3], [0]))
        th = be.permute(be.tensordot(self._gate(t), th, ([3, 4, 5], [1, 2, 3])), (3, 0, 1, 2, 4))
        th = th.reshape(Dl * 4, 16 * Dr)
        lam = be.asarray(np.repeat(self.S[k2], 4), float)
        _, sa, V1 = be.svd(lam[:, None] * th)
        sa = np.asarray(be.host(sa))
        sa = sa / np.linalg.norm(sa)
        n1 = keep_rank(sa, chi, tol)
        _, sb, V2 = be.svd((be.asarray(sa[:n1], float)[:, None] * V1[:n1]).reshape(n1 * 4, 4 * Dr))
        sb = np.asarray(be.host(sb))
        sb = sb / np.linalg.norm(sb)
        n2 = keep_rank(sb, chi, tol)
        B2 = V2[:n2].reshape(n2, 4, Dr)
        B1 = be.tensordot(V1[:n1].reshape(n1, 4, 4 * Dr), V2[:n2], ([2], [1]))             # V1 V2^T
        B12 = be.tensordot(B1, B2, ([2], [0])).reshape(n1, 16 * Dr)
        B0 = th @ B12.T
        self.B[k0] = (B0 / be.norm(lam[:, None] * B0)).reshape(Dl, 4, n1)
        self.B[k1], self.B[k2] = B1, B2
        self.S[k0] = sa[:n1] / np.linalg.norm(sa[:n1])
        self.S[k1] = sb[:n2] / np.linalg.norm(sb[:n2])
        self.discarded += float((sa[n1:] ** 2).sum() + (sb[n2:] ** 2).sum())

    # ------------------------------------------------------------------ exact canonical form
    def canonicalize(self):
        """right-canonical B_k, diagonal Schmidt values S_k on all bonds, <psi|psi> = 1 per cell"""
        be = self.be
        cell = self.cell()
        D = cell.shape[0]
        R, eta = _dominant(lambda x: be.tensordot(be.tensordot(cell, x, ([2], [0])), cell, ([1, 2], [1, 2])),
                           D, np.eye(D) / D, be)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, w.max() * 1e-14, None)
        B = [be.tensordot(be.asarray((V / np.sqrt(w)).T, float), self.B[0], ([1], [0])), self.B[1],
             be.tensordot(self.B[2], be.asarray(V * np.sqrt(w), float), ([2], [0]))]       # X^-1 B0, B1, B2 X
        L = None
        for k in (2, 1, 0):                                      # right-orthonormalize site by site (LQ)
            if L is not None:
                B[k] = be.tensordot(B[k], L, ([2], [0]))
            q, rr = be.qr(B[k].reshape(B[k].shape[0], -1).T)
            B[k], L = q.T.reshape(q.shape[1], 4, -1), rr.T
        B[2] = be.tensordot(B[2], L, ([2], [0])) / math.sqrt(eta)   # L = sqrt(eta) x orthogonal: a cell gauge
        self.B = B
        cell = self.cell()
        D = cell.shape[0]
        Lb, _ = _dominant(lambda x: be.tensordot(be.tensordot(cell, x, ([0], [0])), cell, ([0, 2], [1, 0])),
                          D, np.eye(D) / D, be)
        for k in (2, 0, 1):                                      # bonds 2|0, 0|1, 1|2: diagonalize the left env
            d, U = np.linalg.eigh(Lb)
            d, U = np.clip(d[::-1], 0, None), np.ascontiguousarray(U[:, ::-1])
            self.S[k] = np.sqrt(d / d.sum())
            Ub = be.asarray(U, float)
            B[(k + 1) % 3] = be.tensordot(Ub.T, B[(k + 1) % 3], ([1], [0]))
            B[k] = be.tensordot(B[k], Ub, ([2], [0]))
            nxt = B[(k + 1) % 3]
            Lb = be.to_numpy(be.tensordot(be.tensordot(nxt, be.asarray(np.diag(self.S[k] ** 2), float), ([0], [0])),
                                          nxt, ([0, 2], [1, 0])))

    # ------------------------------------------------------------------ evolution and a local check
    def evolve(self, dbeta, dtau, chi, tol=1e-10, canon_every=1):
        """beta -> beta + dbeta with S4 steps of dbeta / (2n) <= dtau (units of beta / 2)"""
        if canon_every < 1:
            raise ValueError('canon_every must be >= 1')
        n, t = imaginary_steps(dbeta, dtau)                      # raises for dbeta < 0
        sched = s4_schedule(t)
        for step in range(n):
            for r, x in sched:
                self._layer(r, x, chi, tol)
            if (step + 1) % canon_every == 0 or step == n - 1:
                self.canonicalize()
        self.beta += dbeta

    def local_energy(self):
        """<h_j> on the cell's first triple (one sublattice; the Trotter circuit breaks the three-site symmetry)"""
        return self.expect3(H3)

    def expect3(self, op):
        """<psi|O|psi> for a 64 x 64 operator O on the cell's spins 0, 1, 2 (needs the canonical form)"""
        be = self.be
        th = be.asarray(self.S[2], float)[:, None, None] * self.cell()
        Oth = be.permute(be.tensordot(be.asarray(op, float), th, ([1], [1])), (1, 0, 2))
        return float(be.to_numpy((Oth * th).sum()))
