"""h-word basis for operators of the FFD chain and the local maps acting on it.

Basis:  B_S = i^{a(S)} prod_{j in S} h_j  (ordered product), a(S) = number of pairs in S at distance 1 or 2.
B_S is Hermitian, B_S^2 = 1 and omega(B_S B_T) = delta_ST (omega = normalised trace), so an operator is a vector of
real (Hermitian operators) or complex coefficients c_S over occupation bits n_j in {0, 1}: an MPS of local
dimension 2 ("word MPO").  Compared with the Pauli basis (local dimension 4) this halves the local dimension and
removes the redundant spin degrees of freedom.

Every map built from h_c changes only bit c and depends on the bits c-2..c+2.  On these five bits
(b0..b4, b0 most significant, centre b2 = index bit 4):

    left multiplication   h_c B_S = phi(S) B_{S xor c},   phi = i^{-(1-2 b2)(l+r)} (-1)^l,  l = b0+b1, r = b3+b4
    Heisenberg rotation   e^{i d h_c} B_S e^{-i d h_c} = B_S                                    (l + r even)
                                                      = cos(2d) B_S + sin(2d) (i phi) B_{S xor c}   (l + r odd)
    imaginary time        e^{-tau h_c} X = cosh(tau) X - sinh(tau) h_c X
    time reversal         K B_S K = (-1)^{a(S)} B_S        (h_j real in the sigma^z basis)
"""
import itertools
import math

import numpy as np

BITS = list(itertools.product((0, 1), repeat=5))
CENTRE = 4                       # index bit of b2


def _phi(b):
    l, r = b[0] + b[1], b[3] + b[4]
    return (1j) ** (-(1 - 2 * b[2]) * (l + r)) * (-1) ** l


def left_mult_matrix():
    """32 x 32 matrix of X -> h_c X"""
    M = np.zeros((32, 32), complex)
    for src, b in enumerate(BITS):
        M[src ^ CENTRE, src] = _phi(b)
    return M


def rotation_matrix(delta):
    """32 x 32 real matrix of X -> e^{i delta h_c} X e^{-i delta h_c}"""
    R = np.eye(32, dtype=complex)
    c, s = math.cos(2 * delta), math.sin(2 * delta)
    for src, b in enumerate(BITS):
        if (b[0] + b[1] + b[3] + b[4]) % 2 == 0:
            continue
        coeff = 1j * _phi(b)
        assert abs(coeff.imag) < 1e-12
        R[src, src] = c
        R[src ^ CENTRE, src] = s * coeff.real
    return R


LMULT = left_mult_matrix()


def imag_matrix(tau):
    """X -> e^{-tau h_c} X"""
    return math.cosh(tau) * np.eye(32) - math.sinh(tau) * LMULT


def a_count(S):
    """a(S) for a sorted collection of word indices"""
    S = sorted(S)
    return sum(1 for p, q in itertools.combinations(S, 2) if q - p in (1, 2))


def place(pat, n_words):
    """base word k0 such that the density sum_P c_P h_{k0+P...} sits in the middle of a window of n_words words"""
    lo = min(min(k) for k in pat)
    hi = max(max(k) for k in pat)
    k0 = (n_words - 1 - (hi - lo)) // 2 - lo
    if k0 + lo < 0 or k0 + hi > n_words - 1:
        raise ValueError(f'window of {n_words} words too small for an operator spanning {hi - lo + 1} words')
    return k0
