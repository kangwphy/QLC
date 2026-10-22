"""Charge MPS and the static Mazur bound: automata against word enumeration and the note, contractions against
dense matrices of the same imaginary-time circuit, beta = 0 against the closed-form kernels."""
import numpy as np
import pytest

from conftest import backends
from dense_ref import Window
from qlc import theory
from qlc.backend import get_backend
from qlc.model import pauli_dense
from qlc.tn import hword, mazur as mz
from qlc.tn.charges import charge_mps, charge_part_mps, pattern_mps, words_of

BE = get_backend()


def as_vector(words, m, first=None):
    v = np.zeros(2 ** m, complex)
    for S, c in words.items():
        if first is None or S[0] == first:
            v[sum(1 << (m - 1 - j) for j in S)] = c
    return v


@pytest.mark.parametrize('family,p', [('lambda', -1.7), ('lambda', -3.0), ('gamma', -0.4), ('gamma', -0.75)])
@pytest.mark.parametrize('parity', [1, -1, 0])
def test_automaton_matches_enumeration(family, p, parity):
    m = 9
    M = charge_mps(family, p, m, BE) if parity == 0 else charge_part_mps(family, p, parity, m, BE)
    assert np.abs(M.coefficients() - as_vector(words_of(family, p, m, parity), m)).max() < 1e-13
    D = (charge_mps(family, p, m, BE, 'density', 3) if parity == 0 else
         charge_part_mps(family, p, parity, m, BE, 'density', 3))
    assert np.abs(D.coefficients() - as_vector(words_of(family, p, m, parity), m, first=3)).max() < 1e-13


def test_note_coefficients():
    """note, Sec. 3.3 / Remark 3.3 / App. B"""
    p = -1.7
    W = words_of('lambda', p, 8)
    assert np.isclose(W[(0, 1)], 1j / p) and np.isclose(W[(0, 2)], 1j * (1 + p) / p ** 2)
    assert np.isclose(W[(0, 1, 3, 4)], 1j / p ** 2) and np.isclose(W[(0,)], -np.sqrt(-1 - p) / p)
    assert np.isclose(W[(0, 2, 4)], -(1 + p) * np.sqrt(-1 - p) / p ** 3)
    p = -0.4
    W = words_of('gamma', p, 8)
    assert np.isclose(W[(0, 1)], 1j * p) and np.isclose(W[(0, 1, 2, 3)], 1j * p * (1 + p))
    assert np.isclose(W[(0, 1, 3, 4)], 1j * p ** 2) and np.isclose(W[(0, 1, 2)], -1j * p * np.sqrt(1 + p))
    assert np.isclose(W[(0, 1, 2, 3, 4)], -1j * p * (1 + p) ** 1.5)


def test_hermitian_and_k_odd():
    """in the Hermitian B basis (c_B = i^{-a(S)} c_h) every coefficient is real: the charges are Hermitian"""
    for fam, p in (('lambda', -2.0), ('gamma', -0.5)):
        for S, c in words_of(fam, p, 9).items():
            cb = c * (1j) ** (-hword.a_count(S))
            assert abs(cb.imag) < 1e-12 * max(1, abs(cb))


def test_odd_kernel_closed_form():
    """<Q^{gamma,-}_p, Q^{gamma,-}_q> = pq sqrt((1+p)(1+q)) / Delta(p, q), from the word rules (window m = 100)"""
    ps = (-0.3, -0.45, -0.6)
    ch = [(f'g-{p}', 'gamma', p, -1) for p in ps]
    sums, dens, J = mz.build_operators(ch, 100, 4, BE)
    G = mz.omega_matrix(None, sums, dens).real
    for i, p in enumerate(ps):
        for j, q in enumerate(ps):
            assert abs(G[i, j] / theory.mazur_kernel_odd(p, q) - 1) < 1e-5      # window tail at p = -0.6


def dense_h(W, vec):
    X = np.zeros((W.D, W.D), complex)
    for i, c in enumerate(vec):
        if abs(c) > 1e-15:
            X += c * pauli_dense(W.chain.word_pauli(W.word_tuple(i)), W.L)
    return X


def test_contractions_match_dense_circuit():
    """omega(rho A B) with rho from the same real imaginary-time circuit, against dense matrices"""
    W = Window(9)
    m = W.m
    ch = mz.default_charges(lam=(-1.7,), gam=(-0.3,), local=('Q1',))
    sums, dens, J = mz.build_operators(ch, m, 2, BE)
    rho = None
    for _, rho, _ in mz.thermal_trajectory(m, [0.6], 10 ** 6, 0.1, 1e-15, BE, order=2):
        pass
    P = np.linalg.matrix_power(W.imag_sweep(0.1), 6)                 # the same 2nd-order circuit, dense
    rv = rho.coefficients()
    rd = np.array([np.trace(pauli_dense(W.chain.word_pauli(W.word_tuple(i)), W.L).conj().T @ P) / W.D
                   for i in range(len(W.words))])
    assert np.abs(rv / np.linalg.norm(rv) - rd / np.linalg.norm(rd)).max() < 1e-13
    I = mz.WordMPO.identity(m, BE)
    Zm, Zd = mz.omega_product(rho, I, I), np.trace(P)
    ops = sums + [J[0]]
    Om = mz.omega_matrix(rho, ops, dens + [J[1]]) / Zm
    for a, A in enumerate(ops):
        Ad = dense_h(W, A.coefficients())
        for b, B in enumerate(dens + [J[1]]):
            ref = np.trace(P @ Ad @ dense_h(W, B.coefficients())) / Zd
            assert abs(Om[a, b] - ref) < 1e-12
            assert abs(mz.omega_product(rho, A, B) / Zm - ref) < 1e-12


def test_beta0_closed_forms_and_exact_mazur():
    ps = (-2.2, -3.5, -0.35, -0.5)
    ch = mz.default_charges(lam=ps[:2], gam=ps[2:], gam_odd=(-0.35,), local=())
    sums, dens, J = mz.build_operators(ch, 110, 6, BE)
    chk = mz.beta0_check(ch, sums, dens, J)
    assert chk['rel_err_b'] < 1e-12 and chk['rel_err_G'] < 1e-6
    G, b, diag = mz.mazur_matrices(None, sums, dens, J)
    even = [0, 1, 2, 3]
    M, _ = mz.mazur_value(G[np.ix_(even, even)], b[even])
    assert abs(M - theory.mazur_bound(ps)) < 1e-6
    assert abs(b[4]) < 1e-12 and np.abs(G[4, :4]).max() < 1e-12      # odd part decouples at beta = 0
    assert diag['G_asym'] < 1e-12 and diag['b_twoway'] < 1e-12


def test_local_charge_q1():
    sums, dens, J = mz.build_operators([('Q1', 'pattern', 'Q1', 1)], 20, 5, BE)
    G, b, _ = mz.mazur_matrices(None, sums, dens, J)
    assert abs(G[0, 0] - 2) < 1e-12 and abs(b[0] - 6) < 1e-12           # <Q1, Q1> = 2, <J, Q1> = 6


def test_finite_beta_bound():
    """beta = 0.5, small window and fast-decaying charges, 4th-order imaginary time at tau = 0.025: a lower bound
    within 1e-4 of D_GHD (tau -> 0 value 1.000008 at fixed cutoff); odd parts help; stable in the cutoff"""
    ch = mz.default_charges(lam=(-1.6, -2.2, -3.5, -6.0), gam=(-0.2, -0.35, -0.5), local=('Q1',))
    r = mz.run_mazur([0.5], m=64, y0=16, chi=48, tau=0.025, charges=ch, log=None)
    pt = r['points'][0]
    D = pt['D_GHD']
    assert pt['M_even'] <= pt['M_full'] + 1e-9 and pt['ratio_even'] < 0.996
    assert 0.9995 < pt['ratio_full'] < 1.0001
    scan = [v[0] / D for v in pt['cutoff_scan'].values()]
    assert max(scan[2:]) - min(scan[2:]) < 1e-5 and max(scan) < 1.0001


def test_fourth_order_imaginary_time():
    """order 4: error of e^{-beta H} decreases ~16x per halving of tau (order 2: 4x)"""
    from scipy.linalg import expm
    W = Window(9)
    ex = expm(-1.0 * W.H)
    rd = np.array([np.trace(pauli_dense(W.chain.word_pauli(W.word_tuple(i)), W.L).conj().T @ ex) / W.D
                   for i in range(len(W.words))]).real
    rd /= np.linalg.norm(rd)
    for order, lo in ((2, 3.5), (4, 12.0)):
        err = []
        for tau in (0.1, 0.05):
            rho = None
            for _, rho, _ in mz.thermal_trajectory(W.m, [1.0], 10 ** 6, tau, 1e-15, BE, order=order):
                pass
            v = rho.coefficients().real
            err.append(np.abs(v / np.linalg.norm(v) - rd).max())
        assert err[0] / err[1] > lo


def test_run_mazur_validation():
    with pytest.raises(ValueError):
        mz.run_mazur([0.5], m=20, y0=19, chi=8, log=None)
    r = mz.run_mazur([0.0], m=24, y0=4, chi=8, charges=[('Q1', 'pattern', 'Q1', 1)], log=None)
    assert r['beta0_check'] is None and abs(r['points'][0]['M_full'] - 18) < 1e-10      # <J,Q1>^2/<Q1,Q1> = 36/2


@pytest.mark.parametrize('name', backends())
def test_backends_agree(name):
    be = get_backend(name, 'cpu' if name == 'torch' else None)
    ch = mz.default_charges(lam=(-1.7,), gam=(-0.3,), local=('Q1',))
    out = []
    for b_ in (BE, be):
        sums, dens, J = mz.build_operators(ch, 16, 4, b_)
        for _, rho, _ in mz.thermal_trajectory(16, [0.5], 32, 0.05, 1e-12, b_):
            pass
        out.append(mz.mazur_matrices(rho, sums, dens, J))
    assert np.abs(out[0][0] - out[1][0]).max() < 1e-10 and np.abs(out[0][1] - out[1][1]).max() < 1e-10
