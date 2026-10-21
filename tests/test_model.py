import itertools

import numpy as np
import pytest

from qlc.model import Chain, dense_operator, dense_hamiltonian, pauli_dense, pattern, pattern_pauli_terms, PATTERNS
from qlc.model import X2, Y2, Z2


@pytest.mark.parametrize('L,bc', [(7, 'pbc'), (8, 'pbc'), (8, 'obc')])
def test_terms_square_and_anticommute(L, bc):
    ch = Chain(L, bc)
    h = [pauli_dense(ch.term_pauli(j), L) for j in range(ch.n_terms)]
    for i, j in itertools.product(range(ch.n_terms), repeat=2):
        prod = h[i] @ h[j]
        if i == j:
            assert np.allclose(prod, np.eye(1 << L))
            continue
        d = (j - i) % L if bc == 'pbc' else abs(j - i)
        anti = d in (1, 2, L - 1, L - 2) if bc == 'pbc' else d in (1, 2)
        assert np.allclose(prod, -h[j] @ h[i] if anti else h[j] @ h[i])


def test_current_is_polarisation_derivative():
    """J = i [H, P], P = sum_j j h_j, on an open chain"""
    ch = Chain(8, 'obc')
    H = dense_hamiltonian(ch)
    P = sum(j * pauli_dense(ch.term_pauli(j), 8) for j in range(ch.n_terms))
    J = dense_operator(ch, ch.placements('J'))
    assert np.allclose(J, 1j * (H @ P - P @ H))
    assert np.allclose(J, J.conj().T)


def test_split_patterns():
    ch = Chain(9, 'pbc')
    J = dense_operator(ch, ch.placements('J'))
    Q1 = dense_operator(ch, ch.placements('Q1'))
    R = dense_operator(ch, ch.placements('R'))
    H = dense_hamiltonian(ch)
    assert np.allclose(J, 3 * Q1 + R)
    assert np.allclose(H @ Q1, Q1 @ H)                      # Q1 is conserved on the ring
    D, N = 1 << 9, 9
    assert abs(np.trace(J @ J).real / (N * D) - 20) < 1e-12
    assert abs(np.trace(R @ R).real / (N * D) - 2) < 1e-12
    assert abs(np.trace(Q1 @ R)) / (N * D) < 1e-12


def test_pattern_pauli_terms_current():
    """j_x = 2 Z_x Y_{x+2} X_{x+3} + 4 Z_x Z_{x+1} Y_{x+2} Z_{x+3} X_{x+4}"""
    terms = pattern_pauli_terms('J')
    got = sorted([(round(c.real, 12), round(c.imag, 12), tuple(sorted(o))) for c, o in terms])
    assert got == [(2.0, 0.0, (0, 2, 3)), (4.0, 0.0, (0, 1, 2, 3, 4))]
    for c, ops in terms:
        if len(ops) == 3:
            assert np.allclose(ops[0], Z2) and np.allclose(ops[2], Y2) and np.allclose(ops[3], X2)


def test_obc_placements_fit():
    ch = Chain(7, 'obc')
    for c, idx in ch.placements('J'):
        assert max(idx) < ch.n_terms
    assert len(ch.placements('J')) == (ch.n_terms - 1) + (ch.n_terms - 2)
    assert set(PATTERNS) >= {'H', 'J', 'R', 'Q1', 'J3'}
    assert pattern('J') == PATTERNS['J']
