"""Dense reference implementations for the tests (small open chains, full 2^L matrices)."""
import itertools

import numpy as np
from scipy.linalg import expm

from qlc.model import Chain, pauli_dense, pattern, popcount
from qlc.tn import hword


class Window:
    """open chain of L spins with its m = L - 2 words, dense matrices and h-word coefficients"""

    def __init__(self, L):
        self.L, self.m, self.D = L, L - 2, 1 << L
        self.chain = Chain(L, 'obc')
        self.h = [pauli_dense(self.chain.term_pauli(j), L) for j in range(self.m)]
        self.H = sum(self.h)
        self.words = list(itertools.product((0, 1), repeat=self.m))       # word 0 most significant
        n = np.arange(self.D, dtype=np.int64)
        self._basis = []
        for w in self.words:
            S = [j for j in range(self.m) if w[j]]
            x, z, p = self.chain.word_pauli(S)
            ph = (1j) ** (hword.a_count(S) + p)                             # B_S = ph X^x Z^z
            self._basis.append((x, ph * (1 - 2 * (popcount(n & z) & 1)), n))
        self.signs = np.array([(-1) ** hword.a_count([j for j in range(self.m) if w[j]]) for w in self.words])

    def operator(self, c):
        """dense matrix sum_S c_S B_S of a coefficient vector"""
        X = np.zeros((self.D, self.D), complex)
        for ci, (x, sg, n) in zip(c, self._basis):
            if ci != 0:
                X[n ^ x, n] += ci * sg
        return X

    def word_tuple(self, i):
        return tuple(j for j in range(self.m) if self.words[i][j])

    def coeffs(self, X):
        """c_S = omega(B_S X) = Tr(B_S X) / D for all words"""
        out = []
        for x, sg, n in self._basis:
            out.append(np.sum(sg * X[n, n ^ x]) / self.D)      # Tr(X^x Z^z X) = sum_m (-1)^{z m} X[m, m^x]
        return np.array(out)

    def density(self, pat, k0):
        """dense matrix of the pattern density placed at word k0"""
        out = np.zeros((self.D, self.D), complex)
        for offs, c in pattern(pat).items():
            P = np.eye(self.D, dtype=complex)
            for o in offs:
                P = P @ self.h[k0 + o]
            out += c * P
        return out

    def sweep_unitary(self, step):
        """V of the palindromic rotation sweep: X -> V X V^dag"""
        V = np.eye(self.D, dtype=complex)
        seq = [(w, step / 2) for w in range(self.m - 1)] + [(self.m - 1, step)] + \
              [(w, step / 2) for w in range(self.m - 2, -1, -1)]
        for w, d in seq:
            V = expm(1j * d * self.h[w]) @ V
        return V

    def imag_sweep(self, tau):
        P = np.eye(self.D, dtype=complex)
        seq = [(w, tau / 2) for w in range(self.m - 1)] + [(self.m - 1, tau)] + \
              [(w, tau / 2) for w in range(self.m - 2, -1, -1)]
        for w, d in seq:
            P = expm(-d * self.h[w]) @ P
        return P

    def shifted(self, c, x):
        """coefficients a_{S - x} as a vector over S (zero where S - x leaves the window)"""
        T = c.reshape([2] * self.m)
        out = np.zeros_like(T)
        out[(0,) * x + (slice(None),) * (self.m - x)] = T[(slice(None),) * (self.m - x) + (0,) * x]
        return out.reshape(-1)

    def folded(self, Ma, Mb, rho, kappa=-1):
        """C = sum_x C_x with C_x = kappa conj(sum_S Ma_{S-x} Mb_S sign(S)) / Z_x, all shifts"""
        a, b = self.coeffs(Ma), self.coeffs(Mb)
        r = None if rho is None else self.coeffs(rho)
        prof = []
        for x in range(self.m):
            den = 1.0 if r is None else np.sum(self.shifted(r, x) * r).real
            prof.append(kappa * np.conj(np.sum(self.shifted(a, x) * b * self.signs)) / den)
        return prof[0] + 2 * sum(prof[1:]), prof
