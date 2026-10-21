"""Purification TEBD (no truncation) against dense matrices of the same Trotter circuit."""
import numpy as np
import pytest

from conftest import backends
from qlc.backend import get_backend
from qlc.model import Chain, dense_operator, pauli_dense
from qlc.tn.purification import run_purification, centres

L, BETA, DTAU, DT, TMAX = 8, 1.0, 0.05, 0.05, 0.6


def circuit(h, f_half, f_full):
    D = h[0].shape[0]
    out = np.eye(D, dtype=complex)
    for r, f in ((0, f_half), (1, f_half), (2, f_full), (1, f_half), (0, f_half)):
        for j in range(r, len(h), 3):
            out = f(h[j]) @ out
    return out


def dense_reference(route, sym):
    ch = Chain(L, 'obc')
    h = [pauli_dense(ch.term_pauli(j), L) for j in range(ch.n_terms)]
    E = np.eye(1 << L)
    Uim = circuit(h, lambda x: np.cosh(DTAU / 2) * E - np.sinh(DTAU / 2) * x, lambda x: np.cosh(DTAU) * E - np.sinh(DTAU) * x)
    Ure = circuit(h, lambda x: np.cos(DT / 2) * E - 1j * np.sin(DT / 2) * x, lambda x: np.cos(DT) * E - 1j * np.sin(DT) * x)
    M = np.linalg.matrix_power(Uim, int(round(BETA / 2 / DTAU)))
    op = 'J' if route == 'direct' else 'R'
    cs = centres(L, sym)
    O = dense_operator(ch, ch.placements(op))
    oc = sum(dense_operator(ch, ch.local_placements(op, c)) for c in cs) / len(cs)
    nrm = np.trace(M.conj().T @ M).real
    A, B = M.conj().T @ O, oc @ M
    W = np.eye(1 << L, dtype=complex)
    out = []
    for k in range(int(round(TMAX / 2 / DT)) + 1):
        if k:
            W = Ure @ (Ure @ W)
        out.append(np.trace(A @ W @ B @ W.conj().T) / nrm)
    J = dense_operator(ch, ch.placements('J'))
    jc = sum(dense_operator(ch, ch.local_placements('J', c)) for c in cs) / len(cs)
    return np.array(out), np.trace(M.conj().T @ J @ jc @ M).real / nrm


@pytest.mark.parametrize('route,sym', [('direct', True), ('split', True), ('direct', False)])
def test_matches_dense_circuit(route, sym):
    r = run_purification(route, L=L, beta=BETA, chi=4096, chi_psi=4096, tol=1e-15, tol_psi=1e-15, dt=DT, dtau=DTAU,
                         tmax=TMAX, sym=sym, log=None)
    ref, cj0 = dense_reference(route, sym)
    got = np.array([complex(*s['C']) for s in r['samples']])
    assert np.abs(got - ref).max() < 1e-10
    if route == 'split':
        assert abs(r['C_J0'] - cj0) < 1e-10
        assert abs(r['samples'][0]['C_J'] - cj0) < 1e-10


def test_checkpoint_resume(tmp_path):
    kw = dict(route='split', L=10, beta=0.5, chi=32, chi_psi=32, dt=0.05, dtau=0.05, log=None)
    full = run_purification(tmax=0.5, **kw)
    ck, psi = str(tmp_path / 'c.pkl'), str(tmp_path / 'psi.pkl')
    run_purification(tmax=0.2, ckpt=ck, ckpt_every=1, psi_file=psi, **kw)
    res = run_purification(tmax=0.5, ckpt=ck, ckpt_every=1, psi_file=psi, **kw)
    assert len(res['samples']) == len(full['samples'])
    assert max(abs(a['C'][0] - b['C'][0]) for a, b in zip(res['samples'], full['samples'])) < 1e-12
    again = run_purification(tmax=0.2, psi_file=psi, **kw)           # cached thermal state
    assert abs(again['samples'][-1]['C'][0] - full['samples'][2]['C'][0]) < 1e-12


@pytest.mark.parametrize('name', backends())
def test_backends_agree(name):
    be = get_backend(name, 'cpu' if name == 'torch' else None)
    kw = dict(route='split', L=10, beta=0.5, chi=24, chi_psi=24, dt=0.05, dtau=0.05, tmax=0.3, log=None)
    a = run_purification(**kw)
    b = run_purification(be=be, **kw)
    for s1, s2 in zip(a['samples'], b['samples']):
        assert abs(s1['C_J'] - s2['C_J']) < 1e-9
