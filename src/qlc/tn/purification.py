"""Spin-basis purification MPS + TEBD: C(t) = sum_x <j_x(t) j_c>_beta on an open chain of L spins.

Independent of the h-word code (different representation, thermal state and correlation formula):

* purification: each site = physical spin (x) ancilla spin (local dimension 4, index 2 p + a);
  |psi> = e^{-beta H_P / 2} |Phi>, |Phi> = product of Bell pairs (imaginary-time TEBD, normalised).
* real time with the ancilla evolving backwards, calH = H_P - H_A (H real, calH |psi> = 0):
      C(t) = <A(tau)|B(tau)>,  |B(tau)> = e^{-i calH tau} o_c |psi>,  |A(tau)> = e^{+i calH tau} O |psi>,  t = 2 tau,
  with O = sum_x o_x on the chain and o_c the density in the middle (sym: average over the three sublattices
  c, c+1, c+2, which removes most of the method error of the 3-periodic Trotter circuit).
* Trotter: terms with j = r mod 3 commute; U = U0(dt/2) U1(dt/2) U2(dt) U1(dt/2) U0(dt/2), 3-site 64 x 64 gates.

Routes as for the h-word MPO: direct (o = j, C_J) or split (o = r, C_J = C_R - C_R(0) + C_J(0), with C_J(0)
measured on the same thermal state).  At beta = 0 the purification is the vectorised operator, so this is a
Pauli-basis (local dimension 4) operator evolution.  Ported from ffd_purif/purif_tebd.py of the 2026-10-20 package.
"""
import json
import os
import pickle
import time

import numpy as np

from ..model import Chain, pattern, pattern_pauli_terms, I2, X2, Z2
from .mps import MPS, keep_rank

D = 4


def _phys(o):
    return np.kron(o, I2)


def _anc(o):
    return np.kron(I2, o)


_HP3 = np.kron(np.kron(_phys(Z2), _phys(Z2)), _phys(X2))
_HA3 = np.kron(np.kron(_anc(Z2), _anc(Z2)), _anc(X2))
_E64 = np.eye(64)


def gate_real(dt):
    """exp(-i dt (h_P - h_A)) on three sites"""
    with np.errstate(all='ignore'):
        G = (np.cos(dt) * _E64 - 1j * np.sin(dt) * _HP3) @ (np.cos(dt) * _E64 + 1j * np.sin(dt) * _HA3)
    assert np.all(np.isfinite(G)) and np.allclose(G @ G.conj().T, _E64, atol=1e-13)
    return G


def gate_imag(dtau):
    """exp(-dtau h_P)"""
    return np.cosh(dtau) * _E64 - np.sinh(dtau) * _HP3


class Purification(MPS):
    def layer(self, r, G, chi, tol):
        """G on all triples (j, j+1, j+2), j = r mod 3, sweeping from the side nearest the centre"""
        starts = list(range(r, len(self.A) - 2, 3))
        if not starts:
            return 0.0
        lost = 0.0
        if abs(self.center - starts[0]) <= abs(self.center - starts[-1] - 2):
            for i in starts:
                lost += self.apply_gate(i, G, 3, chi, tol, 'right')
        else:
            for i in reversed(starts):
                lost += self.apply_gate(i, G, 3, chi, tol, 'left')
        return lost

    def step(self, Gh, Gf, chi, tol):
        lost = 0.0
        for r, G in ((0, Gh), (1, Gh), (2, Gf), (1, Gh), (0, Gh)):
            lost += self.layer(r, G, chi, tol)
        return lost

    @classmethod
    def bell(cls, L, be):
        t = np.zeros((1, D, 1), complex)
        t[0, 0, 0] = t[0, 3, 0] = 1 / np.sqrt(2)
        return cls([be.asarray(t) for _ in range(L)], be, 0)


def thermal_state(L, beta, dtau, chi, tol, be):
    psi = Purification.bell(L, be)
    if beta > 0:
        Gh, Gf = be.asarray(gate_imag(dtau / 2)), be.asarray(gate_imag(dtau))
        for _ in range(int(round(beta / 2 / dtau))):
            psi.step(Gh, Gf, chi, tol)
            psi.normalize()
    return psi


def operator_mpo(pat, L, starts, weight=1.0):
    """finite-state MPO (NumPy) of weight * sum_{x in starts} o_x on the physical spins; W[k]: (wl, wr, 4, 4).
    Placement x is kept only if its whole support fits in the chain."""
    terms = []
    for c, ops in pattern_pauli_terms(pat):
        span = max(ops) + 1
        terms.append((c * weight, [_phys(ops.get(s, I2)) for s in range(span)]))
    nst = 2 + sum(len(o) - 1 for _, o in terms)
    F = nst - 1
    Ws = []
    for k in range(L):
        W = np.zeros((nst, nst, D, D), complex)
        W[0, 0] = np.eye(D)
        W[F, F] = np.eye(D)
        s0 = 1
        for c, ops in terms:
            m = len(ops)
            if k in starts and k + m <= L:
                if m == 1:
                    W[0, F] += c * ops[0]
                else:
                    W[0, s0] += c * ops[0]
            for j in range(1, m):
                src = s0 + j - 1
                dst = F if j == m - 1 else s0 + j
                W[src, dst] += ops[j]
            s0 += m - 1
        if k == 0:
            W = W[:1]
        if k == L - 1:
            W = W[:, F:F + 1]
        Ws.append(W)
    return Ws


def apply_mpo(psi, Ws, chi, tol):
    """MPO |psi> as a compressed MPS; site-by-site SVD (cutoff 1e-12, cap 2 chi) keeps memory low"""
    be = psi.be
    L = len(psi.A)
    p = psi.copy()
    p.move_center(0)
    out = []
    Cm = be.asarray([[1.0]])
    for k in range(L):
        W = be.asarray(Ws[k])
        a = p.A[k]
        l, d, r = a.shape
        t = be.einsum('uvpq,lqr->luprv', W, a).reshape(l * W.shape[0], d, r * W.shape[1])
        t = be.tensordot(Cm, t, (1, 0))
        if k < L - 1:
            nl = t.shape[0]
            u, s, v = be.svd(t.reshape(nl * d, -1))
            kk = keep_rank(s, 2 * chi, 1e-12)
            out.append(u[:, :kk].reshape(nl, d, kk))
            Cm = s[:kk, None] * v[:kk]
        else:
            out.append(t)
    m = Purification(out, be, L - 1)
    for k in range(L - 1, 0, -1):
        l, d, r = m.A[k].shape
        u, s, v = be.svd(m.A[k].reshape(l, d * r))
        kk = keep_rank(s, chi, tol)
        m.A[k] = v[:kk].reshape(kk, d, r)
        m.A[k - 1] = be.tensordot(m.A[k - 1], u[:, :kk] * s[:kk], (2, 0))
    m.center = 0
    return m


def centres(L, sym=True):
    """first spins of the probe densities: the middle of the chain (same convention as qlc.ed probe='local')"""
    return Chain(L, 'obc').centre_terms(sym)


def run_purification(route='split', L=64, beta=1.0, chi=256, chi_psi=192, tol=1e-10, tol_psi=1e-12, dt=0.025,
                     dtau=0.0125, tmax=4.0, sym=True, op=None, psi_file=None, out=None, ckpt=None, ckpt_every=5,
                     be=None, log=print):
    """Purification TEBD run; returns the result dict (written to `out` after every sample)."""
    from ..backend import get_backend
    be = be or get_backend()
    if route not in ('direct', 'split'):
        raise ValueError("route must be 'direct' or 'split'")
    opname = op if op is not None else ('J' if route == 'direct' else 'R')
    pat = pattern(opname)
    cs = centres(L, sym)
    params = dict(method='spin-basis purification TEBD', route=route,
                  op=opname if isinstance(opname, str) else str(pat), L=L, beta=beta, chi=chi, chi_psi=chi_psi,
                  tol=tol, tol_psi=tol_psi, dt=dt, dtau=dtau, centres=cs, sym=int(sym))
    nsteps = int(round(tmax / 2 / dt))
    t0 = time.time()
    lostA = lostB = 0.0
    n0 = 0
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            st = pickle.load(f)
        bad = {k: (st['params'].get(k), v) for k, v in params.items() if st['params'].get(k) != v}
        if bad:
            raise ValueError(f'checkpoint {ckpt} has different parameters: {bad}')
        A, B = Purification.from_numpy(st['A'], be), Purification.from_numpy(st['B'], be)
        res, lostA, lostB, n0 = st['res'], st['lostA'], st['lostB'], st['n'] + 1
        res.pop('status', None)
        if log:
            log(f'resumed from {ckpt} at t = {2 * st["n"] * dt:.4f}')
    else:
        psi = None
        if psi_file and os.path.exists(psi_file):
            with open(psi_file, 'rb') as f:
                d = pickle.load(f)
            assert (d['L'], d['beta'], d['dtau'], d['chi_psi']) == (L, beta, dtau, chi_psi), 'psi_file parameters differ'
            psi = Purification.from_numpy(d, be)
            if log:
                log(f'loaded thermal state from {psi_file}')
        if psi is None:
            psi = thermal_state(L, beta, dtau, chi_psi, tol_psi, be)
            if psi_file:
                with open(psi_file, 'wb') as f:
                    pickle.dump(dict(L=L, beta=beta, dtau=dtau, chi_psi=chi_psi, **psi.to_numpy()), f)
        B = apply_mpo(psi, operator_mpo(pat, L, set(cs), 1.0 / len(cs)), chi, tol)
        A = apply_mpo(psi, operator_mpo(pat, L, set(range(L))), chi, tol)
        res = dict(params, psi_bond=psi.bond(), samples=[])
        if route == 'split' and op is None:
            Jtot = apply_mpo(psi, operator_mpo('J', L, set(range(L))), chi, tol)
            jc = apply_mpo(psi, operator_mpo('J', L, set(cs), 1.0 / len(cs)), chi, tol)
            res['C_J0'] = Jtot.overlap(jc).real
            res['C_J0_source'] = 'static, same thermal state'
        if log:
            log(f'purification, route={route}, L={L}, beta={beta}: psi bond {psi.bond()}, A bond {A.bond()}, '
                f'B bond {B.bond()}, centres {cs}, backend={be.name}' +
                (f', C_J(0) = {res["C_J0"]:.8f}' if 'C_J0' in res else '') + f'  [{time.time() - t0:.0f}s]')
    Ghp, Gfp = be.asarray(gate_real(dt / 2)), be.asarray(gate_real(dt))
    Ghm, Gfm = be.asarray(gate_real(-dt / 2)), be.asarray(gate_real(-dt))
    for n in range(n0, nsteps + 1):
        if n > 0:
            lostB += B.step(Ghp, Gfp, chi, tol)
            lostA += A.step(Ghm, Gfm, chi, tol)
        Ct = A.overlap(B)
        row = dict(t=2 * n * dt, C=[Ct.real, Ct.imag], bondA=A.bond(), bondB=B.bond(), lostA=lostA, lostB=lostB,
                   elapsed_s=time.time() - t0)
        res['samples'].append(row)
        if 'C_J0' in res:
            row['C_J'] = Ct.real - res['samples'][0]['C'][0] + res['C_J0']
        elif op is None:
            row['C_J'] = Ct.real
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out + '.tmp', 'w') as f:
                json.dump(res, f, indent=1)
            os.replace(out + '.tmp', out)
        if ckpt and ckpt_every > 0 and (n % ckpt_every == 0 or n == nsteps):
            os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
            with open(ckpt + '.tmp', 'wb') as f:
                pickle.dump(dict(params=params, A=A.to_numpy(), B=B.to_numpy(), n=n, res=res, lostA=lostA,
                                 lostB=lostB), f)
            os.replace(ckpt + '.tmp', ckpt)
        if log:
            log(json.dumps(dict(t=round(row['t'], 4), C=round(Ct.real, 9), C_J=round(row.get('C_J', np.nan), 9),
                                Cim=f'{Ct.imag:.1e}', bonds=[row['bondA'], row['bondB']],
                                lost=[f'{lostA:.1e}', f'{lostB:.1e}'], el=round(row['elapsed_s']))))
    res['status'] = 'completed'
    if out:
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)
    return res
