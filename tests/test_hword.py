"""h-word MPO: local maps, overlaps and the full evolution against dense matrices of the same Trotter circuit."""
import json
import os

import numpy as np
import pytest
from scipy.linalg import expm

from conftest import backends, REF_DIR
from dense_ref import Window
from qlc.backend import get_backend
from qlc.tn import hword
from qlc.tn.wordmpo import WordMPO, shifted_overlap, pattern_words, k_parity
from qlc.tn.heisenberg import run_hword, thermal_factor, static_correlation
from qlc.model import pattern

W = Window(9)
BE = get_backend()
RNG = np.random.default_rng(7)


def mpo_from_vector(c, be=BE):
    return WordMPO.from_words(W.m, {W.word_tuple(i): complex(x) for i, x in enumerate(c) if x != 0}, be)


def rand_vec(real=False):
    v = RNG.normal(size=len(W.words))
    return v if real else v + 1j * RNG.normal(size=len(W.words))


def test_basis_orthonormal_hermitian():
    for i in (0, 5, 77, 127):
        B = W.operator(np.eye(len(W.words))[i])
        assert np.allclose(B, B.conj().T) and np.allclose(B @ B, np.eye(W.D))
        assert np.allclose(W.coeffs(B), np.eye(len(W.words))[i])


def test_left_multiplication():
    c = rand_vec()
    X = W.operator(c)
    for w in (0, 3, W.m - 1):
        M = mpo_from_vector(c).left_mult(w)
        assert np.allclose(M.coefficients(), W.coeffs(W.h[w] @ X), atol=1e-12)


@pytest.mark.parametrize('real', [True, False])
def test_rotation_gate(real):
    c = rand_vec(real)
    X = W.operator(c)
    for w in (0, 2, W.m - 1):
        M = mpo_from_vector(c)
        G = M._gate('rot', 0.137)
        M.apply_gate(w, G, 5, 10 ** 6, 1e-15)
        U = expm(1j * 0.137 * W.h[w])
        assert np.allclose(M.coefficients(), W.coeffs(U @ X @ U.conj().T), atol=1e-12)
        assert M.is_complex() == (not real)


def test_imag_gate_and_thermal_factor():
    c = rand_vec()
    M = mpo_from_vector(c)
    M.make_complex()
    M.apply_gate(3, M._gate('imag', 0.2), 5, 10 ** 6, 1e-15)
    assert np.allclose(M.coefficients(), W.coeffs(expm(-0.2 * W.h[3]) @ W.operator(c)), atol=1e-12)
    beta, tau = 0.8, 0.1
    rho = thermal_factor(W.m, beta, 10 ** 6, tau, 1e-15, BE)
    assert not rho.is_complex()
    P = np.linalg.matrix_power(W.imag_sweep(tau), int(round(beta / 2 / tau)))
    ref = W.coeffs(P)
    v = rho.coefficients()
    assert np.allclose(v / np.linalg.norm(v), ref / np.linalg.norm(ref), atol=1e-12)
    ex = W.coeffs(expm(-beta / 2 * W.H))                       # Trotter error only
    assert np.abs(v / np.linalg.norm(v) - ex / np.linalg.norm(ex)).max() < 5e-3


@pytest.mark.parametrize('signed', [True, False])
def test_shifted_overlap(signed):
    a, b = rand_vec(), rand_vec()
    A, B = mpo_from_vector(a), mpo_from_vector(b)
    for x in range(4):
        dense = np.sum(W.shifted(a, x) * b * (W.signs if signed else 1))
        assert abs(shifted_overlap(A, B, x, signed) - dense) < 1e-10


def test_real_part():
    c = rand_vec()
    M = mpo_from_vector(c)
    R = M.real_part()
    assert not R.is_complex()
    assert np.allclose(R.coefficients(), c.real, atol=1e-12)


def test_pattern_words_and_parity():
    assert pattern_words(pattern('J'), 3) == {(3, 4): 2, (3, 5): 4}
    assert pattern_words(pattern('R'), 0) == {(0, 1): -1, (0, 2): 1}
    assert k_parity(pattern('J')) == -1 and k_parity(pattern('J3')) == -1 and k_parity(pattern('H')) == 1
    I = WordMPO.identity(W.m, BE)
    for name in ('J', 'R', 'J3', 'H'):
        pat = pattern(name)
        if max(max(k) for k in pat) >= W.m:
            continue
        k0 = hword.place(pat, W.m)
        a = I.times_pattern(pat, k0, 10 ** 6, 1e-15).coefficients()
        b = WordMPO.from_words(W.m, pattern_words(pat, k0), BE).coefficients()
        assert np.allclose(a, b, atol=1e-12) and np.allclose(b, W.coeffs(W.density(pat, k0)), atol=1e-12)


@pytest.mark.parametrize('route,beta', [('direct', 0.0), ('split', 0.0), ('direct', 0.6), ('split', 0.6)])
def test_evolution_matches_dense_circuit(route, beta):
    """no truncation: run_hword reproduces the dense same-circuit emulation to rounding"""
    dt, tau, tmax = 0.1, 0.1, 0.6
    r = run_hword(route=route, n=W.L, beta=beta, chi=10 ** 6, chi_rho=10 ** 6, tol=1e-15, dt=dt, tau_im=tau,
                  tmax=tmax, x0=100, log=None)
    pat = pattern('J' if route == 'direct' else 'R')
    k0 = hword.place(pat, W.m)
    rho = None
    if beta > 0:
        rho = np.linalg.matrix_power(W.imag_sweep(tau), int(round(beta / 2 / tau)))
    M = W.density(pat, k0) @ (rho if rho is not None else np.eye(W.D))
    V = W.sweep_unitary(dt)
    rt = rho
    for k, s in enumerate(r['samples']):
        if k:
            M = V @ M @ V.conj().T
            rt = None if rt is None else V @ rt @ V.conj().T
        C, _ = W.folded(M, M, rt)
        assert abs(complex(*s['C']) - C) < 1e-10 * max(1, abs(C)), (s['t'], s['C'], C)
    if route == 'split' and beta > 0:                           # C_J(0) from the same thermal factor
        Mj = W.density(pattern('J'), hword.place(pattern('J'), W.m)) @ rho
        CJ0, _ = W.folded(Mj, Mj, rho)
        assert abs(r['C_J0'] - CJ0.real) < 1e-10


def test_split_equals_direct_beta0():
    """same physics up to an O(dt^2) term (the Trotter circuit conserves Q1 only approximately); at chi = 64 the
    direct route starts to lose accuracy after t ~ 0.8 while the split route stays accurate"""
    kw = dict(n=20, chi=64, dt=0.05, tmax=0.8, sample_every=4, log=None)
    a = run_hword(route='direct', **kw)
    b = run_hword(route='split', **kw)
    for s1, s2 in zip(a['samples'], b['samples']):
        assert abs(s1['C_J'] - s2['C_J']) < 2e-3
    assert b['samples'][0]['C_J'] == 20.0


def test_checkpoint_resume(tmp_path):
    kw = dict(route='split', n=14, chi=32, dt=0.05, log=None)
    full = run_hword(tmax=0.6, **kw)
    ck = str(tmp_path / 'r.ckpt')
    run_hword(tmax=0.3, ckpt=ck, ckpt_every=1, **kw)
    res = run_hword(tmax=0.6, ckpt=ck, ckpt_every=1, **kw)
    assert len(res['samples']) == len(full['samples'])
    for s1, s2 in zip(res['samples'], full['samples']):
        assert abs(s1['C'][0] - s2['C'][0]) < 1e-12


@pytest.mark.parametrize('name', backends())
def test_backends_agree(name):
    be = get_backend(name, 'cpu' if name == 'torch' else None)
    for beta in (0.0, 0.5):
        kw = dict(route='split', n=12, beta=beta, chi=24, chi_rho=16, dt=0.05, tau_im=0.05, tmax=0.4, log=None)
        a = run_hword(**kw)
        b = run_hword(be=be, **kw)
        for s1, s2 in zip(a['samples'], b['samples']):
            assert abs(s1['C_J'] - s2['C_J']) < 1e-9


@pytest.mark.slow
def test_bulk_reference_beta0():
    """C_J(t) of the infinite chain (N = 96, chi = 512, dt-extrapolated) for t <= 0.8"""
    ref = np.loadtxt(os.path.join(REF_DIR, 'CJ_beta0.txt'))
    for route in ('direct', 'split'):
        r = run_hword(route=route, n=24, chi=64, dt=0.025, tmax=0.8, sample_every=4, log=None)
        for s in r['samples']:
            i = int(np.argmin(np.abs(ref[:, 0] - s['t'])))
            assert abs(ref[i, 0] - s['t']) < 1e-9
            assert abs(s['C_J'] - ref[i, 1]) < 5e-4          # dt^2 Trotter error at dt = 0.025


@pytest.mark.slow
def test_bulk_static_beta1():
    """<J, J>_beta / N and <R, R>_beta / N at beta = 1 (exact ring values 4.290727, 2.607040); windows n >= 32
    are needed at beta = 1 (n = 24 is off by 3%)"""
    rho = thermal_factor(30, 1.0, 64, 0.0125, 1e-12, BE)
    assert abs(static_correlation('J', 30, rho, 128, 1e-12, 12, BE).real - 4.290727) < 1e-3
    assert abs(static_correlation('R', 30, rho, 128, 1e-12, 12, BE).real - 2.607040) < 1e-3


@pytest.mark.slow
def test_bulk_reference_beta1():
    """both routes against the 2026-10 h-word production reference (n = 168, chi = 256, dt = 0.05), t <= 0.3;
    smaller windows / bonds (n = 36, chi = 96) are already off by 3e-4"""
    with open(os.path.join(REF_DIR, 'tn_reference.json')) as f:
        ref = json.load(f)['hword_pair_b1_chi256_n168_dt0.05_tmax6']
    for route in ('direct', 'split'):
        r = run_hword(route=route, n=40, beta=1.0, chi=128, chi_rho=64, dt=0.05, tmax=0.3, log=None)
        for i, s in enumerate(r['samples']):
            assert abs(s['t'] - ref['t'][i]) < 1e-9 and abs(s['C_J'] - ref['C'][i]) < 3e-5
