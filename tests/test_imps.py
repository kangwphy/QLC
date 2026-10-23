"""Purification iMPS and the transfer-matrix Mazur bound: spin-basis MPOs against dense operators, beta = 0 against
the closed forms (including charges with p next to -1), pairs against the dense cell transfer matrix, the canonical
form, fourth-order imaginary time, finite beta against exact free-fermion values and the two-layer TBA Gram matrix
(quad_mazur/finiteT of the 2026-10 package)."""
import numpy as np
import pytest

from conftest import backends
from qlc import theory
from qlc.backend import get_backend
from qlc.model import Chain, dense_operator, pauli_dense
from qlc.tn.charges import words_of
from qlc.tn.imazur import CellTM, Operator, charge_spec_mpo, measure, run_imazur
from qlc.tn.imps import ThermalIMPS, s4_schedule
from qlc.tn.pauli_mpo import charge_mpo, pattern_mpo

BE = get_backend()


def dense_words(mpo, L):
    """the single words of an IDLE-form MPO inside an open chain of L spins (flag MPO [[W0, W1], [0, W0]])"""
    D = mpo.D
    W = np.zeros((2 * D, 2 * D, 2, 2), complex)
    W[:D, :D] = W[D:, D:] = mpo.W0
    W[:D, D:] = mpo.W1
    T = np.zeros((2 * D, 1, 1), complex)
    T[0, 0, 0] = 1
    for _ in range(L):
        a, b = T.shape[1:]
        T = np.tensordot(T, W, (0, 0)).transpose(2, 0, 3, 1, 4).reshape(2 * D, 2 * a, 2 * b)
    idx = np.array([int(format(n, f'0{L}b')[::-1], 2) for n in range(1 << L)])        # spin s = bit s
    return T[D][np.ix_(idx, idx)]


@pytest.mark.parametrize('name', ['H', 'J', 'R', 'Q1', 'Q2', 'J3'])
def test_pattern_mpo_dense(name):
    ch = Chain(9, 'obc')
    assert np.abs(dense_words(pattern_mpo(name), 9) - dense_operator(ch, ch.placements(name))).max() < 1e-14


@pytest.mark.parametrize('family,p', [('lambda', -1.7), ('lambda', -1.06), ('gamma', -0.4), ('gamma', -0.94)])
@pytest.mark.parametrize('parity', [1, -1, 0])
def test_charge_mpo_dense(family, p, parity):
    ch = Chain(9, 'obc')
    ref = sum(c * pauli_dense(ch.word_pauli(list(S)), 9) for S, c in words_of(family, p, 7, parity).items())
    mpo = charge_mpo(family, p, parity)
    assert np.abs(dense_words(mpo, 9) - ref).max() < 1e-14
    assert mpo.D == (7 if parity else 4)


def test_beta0_closed_forms():
    """Bell state (bond 1): kernels, odd kernel and overlaps exact, also for slowly decaying charges (p near -1)"""
    tm = CellTM(ThermalIMPS(BE))
    specs = [('l1', 'lambda', -1.03, 1), ('l2', 'lambda', -3.5, 1), ('g1', 'gamma', -0.97, 1), ('g2', 'gamma', -0.3, 1),
             ('o1', 'gamma', -0.97, -1), ('o2', 'gamma', -0.3, -1)]
    ops = [tm.prepare(Operator(charge_spec_mpo(s), BE)) for s in specs]
    for (_, fa, pa, sa), A in zip(specs, ops):
        for (_, fb, pb, sb), B in zip(specs, ops):
            ex = 0.0 if sa != sb else (theory.mazur_kernel(pa, pb) if sa == 1 else theory.mazur_kernel_odd(pa, pb))
            assert abs(tm.pair(A, B) - ex) < 1e-10 * max(1, abs(ex))
    J, H, Q1, Q2 = (tm.prepare(Operator(pattern_mpo(n), BE)) for n in ('J', 'H', 'Q1', 'Q2'))
    for (_, f, p, s), A in zip(specs, ops):
        assert abs(tm.pair(J, A) - (theory.mazur_overlap(p) if s == 1 else 0)) < 1e-10
    vals = [tm.pair(J, J), tm.pair(H, H), H.d1, tm.pair(Q1, Q1), tm.pair(Q2, Q2), tm.pair(Q1, Q2), tm.pair(J, Q2)]
    assert np.allclose(vals, [20, 1, 0, 2, 6, 1, 4], atol=1e-11)


def test_beta0_mazur_family():
    ch = [(f'g{p}', 'gamma', p, 1) for p in (-0.2, -0.5)] + [(f'l{p}', 'lambda', p, 1) for p in (-1.6, -3.5)]
    r = measure(ThermalIMPS(BE), ch)
    assert abs(r['M_full'] - theory.mazur_bound([-0.2, -0.5, -1.6, -3.5])) < 1e-10
    assert r['energy'] == pytest.approx(0, abs=1e-12) and r['HH'] == pytest.approx(1, abs=1e-12)


def test_pair_against_dense_eigenvalue():
    """(1/3) d_a d_b log Lambda of the dense two-layer cell matrix (Cauchy formula on a small circle) at finite beta,
    for a pair with <A>, <B> != 0 (H, lambda-) and a T-odd pair"""
    st = ThermalIMPS(BE)
    st.evolve(0.5, 0.1, 3)
    B = [BE.to_numpy(b).reshape(b.shape[0], 2, 2, b.shape[2]) for b in st.B]

    def lam(Wa, Wb):
        E = [np.einsum('ascx,ikut,jlts,bucy->aijbxkly', b, Wa, Wb, b, optimize=True) for b in B]
        E = [e.reshape(int(np.sqrt(e.size)), -1) for e in E]
        w = np.linalg.eigvals(E[0] @ E[1] @ E[2])
        return w[np.argmin(abs(w - 1))]

    N, rho = 12, 0.03
    om = rho * np.exp(2j * np.pi * np.arange(N) / N)
    tm = CellTM(st)
    for ma, mb in ((pattern_mpo('H'), charge_mpo('lambda', -1.6, -1)), (pattern_mpo('J'), pattern_mpo('Q1'))):
        f = np.array([[np.log(lam(ma.W0 + x * ma.W1, mb.W0 + y * mb.W1)) for y in om] for x in om])
        dense = (f / np.outer(om, om)).sum() / N ** 2 / 3
        A, Bo = tm.prepare(Operator(ma, BE)), tm.prepare(Operator(mb, BE))
        assert abs(tm.pair(A, Bo) - dense) < 1e-11


def test_transfer_matrix_adjoint():
    st = ThermalIMPS(BE)
    st.evolve(0.4, 0.1, 6)
    tm = CellTM(st)
    A, B = Operator(charge_mpo('gamma', -0.5, 1), BE), Operator(pattern_mpo('Q1'), BE)
    rng = np.random.default_rng(1)
    shape = (A.D, B.D, tm.D, tm.D)
    l, x = (BE.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)) for _ in range(2))
    lhs, rhs = tm.dot(tm.left(l, A.W0, B.W0), x), tm.dot(l, tm.right(x, A.W0, B.W0))
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)


@pytest.mark.parametrize('chi,canon_every', [(12, 1), (24, 1), (24, 5)])         # dense and ARPACK fixed points
def test_canonical_form(chi, canon_every):
    st = ThermalIMPS(BE)
    st.evolve(0.6, 0.1, chi, canon_every=canon_every)
    for k in range(3):
        B = BE.to_numpy(st.B[k])
        assert np.abs(np.einsum('asb,csb->ac', B, B) - np.eye(B.shape[0])).max() < 1e-10       # right-isometric
        Lk = np.einsum('asb,a,asc->bc', B, st.S[k - 1] ** 2, B)                                   # left env moves
        assert np.abs(Lk - np.diag(st.S[k] ** 2)).max() < 1e-10


def test_s4_schedule():
    s = s4_schedule(1.0)
    assert len(s) == 21 and s[0][0] == 0
    assert all(abs(sum(t for r, t in s if r == k) - 1) < 1e-14 for k in range(3))


def test_fourth_order_energy():
    """E/N error (from <h_0> on the cell's first triple) falls ~16x per halving of the step"""
    e_ex = theory.thermal_energy(0.4)[0]
    err = []
    for dtau in (0.1, 0.05):
        st = ThermalIMPS(BE)
        st.evolve(0.4, dtau, 24)
        err.append(st.local_energy() - e_ex)
    assert err[0] / err[1] > 12 and abs(err[1]) < 1e-6


def test_finite_beta_tba():
    """beta = 1: Q1/Q2 Gram entries, the slow gamma(-0.2) diagonal and current overlaps against the TBA / bare values
    (the window code is off by 1.6e-4 relative on the gamma(-0.2) diagonal here)"""
    st = ThermalIMPS(BE)
    st.evolve(1.0, 0.05, 32)                    # chi = 24: up to 5e-5 (truncation); chi = 32: < 3e-6
    tm = CellTM(st)
    Q1, Q2, J, H = (tm.prepare(Operator(pattern_mpo(n), BE)) for n in ('Q1', 'Q2', 'J', 'H'))
    g = tm.prepare(Operator(charge_mpo('gamma', -0.2, 1), BE))
    got = [tm.pair(Q1, Q1), tm.pair(Q1, Q2), tm.pair(Q2, Q2), tm.pair(g, g), tm.pair(J, Q1), tm.pair(J, Q2)]
    ref = [0.4919407, -0.4135226, 1.0078367, 0.13651106, 1.0185264, -0.7708450]
    assert max(abs(x / y - 1) for x, y in zip(got, ref)) < 1e-5
    e, hh = theory.thermal_energy(1.0)
    assert abs(H.d1 / 3 - e) < 1e-6 and abs(tm.pair(H, H) - hh) < 1e-6


@pytest.mark.parametrize('name', backends())
def test_backends_agree(name):
    be = get_backend(name, 'cpu' if name == 'torch' else None)
    out = []
    for b in (BE, be):
        st = ThermalIMPS(b)
        st.evolve(0.3, 0.1, 16)
        tm = CellTM(st)
        A, B = (tm.prepare(Operator(m, b)) for m in (charge_mpo('lambda', -1.6, 1), pattern_mpo('J')))
        out.append([tm.pair(A, A), tm.pair(B, A), st.local_energy()])
    assert np.allclose(out[0], out[1], atol=1e-10)


def test_run_imazur(tmp_path):
    ch = [('Q1', 'pattern', 'Q1', 1), ('g', 'gamma', -0.5, 1), ('go', 'gamma', -0.5, -1)]
    r = run_imazur([0.0, 0.2], chi=8, dtau=0.1, charges=ch, log=None, out=str(tmp_path / 'r.json'))
    p0, p1 = r['points']
    assert (tmp_path / 'r.json').exists() and p0['bond'] == 1 and p1['bond'] > 1
    assert abs(p0['G'][0][0] - 2) < 1e-12 and abs(p0['b'][0] - 6) < 1e-12 and p0['b_odd'] < 1e-12
    assert p1['ratio_full'] < 1 and p1['ratio_even'] <= p1['ratio_full'] + 1e-12 and p1['G_imag'] < 1e-10
    with pytest.raises(ValueError):
        ThermalIMPS(BE).evolve(0.1, 0.1, 4, canon_every=0)
    r = run_imazur([0.3, 0.9, 0.9], chi=4, dtau=0.1, charges=ch[:1], log=None)          # beta accumulates without drift
    assert [p['beta'] for p in r['points']] == [0.3, 0.9]
