"""The FFD chain (Fendley's free fermions in disguise): local terms, currents, boundary conditions.

    H = sum_j h_j,          h_j = Z_j Z_{j+1} X_{j+2}
    J = 2i sum_j (h_j h_{j+1} + 2 h_j h_{j+2})                 energy current, J = i[H, P] with P = sum_j j h_j
    J = 3 Q1 + R,           Q1 = i sum_j (h_j h_{j+1} + h_j h_{j+2})      (conserved)
                            R  = i sum_j (h_j h_{j+2} - h_j h_{j+1})      (<Q1, R> = 0 at beta = 0)

Spins are labelled 0..L-1, terms h_j by their first spin j.  Periodic ring ('pbc'): j = 0..L-1, spins mod L.
Open chain ('obc'): j = 0..L-3.

Operators are written in the h_j ("h-words").  A *pattern* is a dict {offsets: coefficient} standing for the
translation sum  sum_j sum_P c_P h_{j+P_1} h_{j+P_2} ...  (ordered product).  On an open chain a placement j is kept
only when all its terms exist; on a ring the term indices are taken mod L.

Pauli strings are triples (x, z, p) of two bitmasks and a phase exponent: i^p X^x Z^z, with X^x Z^z meaning
prod_s X_s^{x_s} prod_s Z_s^{z_s}; bit s of a mask is spin s.
"""
import numpy as np

PATTERNS = {
    'H': {(0,): 1.0},
    'J': {(0, 1): 2j, (0, 2): 4j},
    'R': {(0, 1): -1j, (0, 2): 1j},
    'Q1': {(0, 1): 1j, (0, 2): 1j},
    # current of the third local charge H_3 = H^3 - 3 H I_2 + 3 I_3 (note, App. F)
    'J3': {(0, 1): 6j, (0, 2): 14j, (0, 1, 3, 4): 2j, (0, 1, 3, 5): 2j, (0, 2, 3, 5): 4j, (0, 2, 4, 5): 4j,
           (0, 2, 4, 6): 4j},
}

# exact infinite-temperature static values <A, A> per site (word shapes of distinct translation classes are orthonormal)
STATIC_BETA0 = {'J': 20.0, 'R': 2.0, 'Q1': 2.0, 'H': 1.0}


def pattern(op):
    """a pattern from a name in PATTERNS or a dict"""
    if isinstance(op, str):
        return dict(PATTERNS[op])
    return {tuple(k): complex(v) for k, v in op.items()}


def pattern_norm2(pat):
    """<A, A> per site at beta = 0 for a pattern whose words are distinct translation classes"""
    return float(sum(abs(c) ** 2 for c in pat.values()))


def popcount(a):
    """vectorised popcount of a non-negative int64 array, as int64 (np.bitwise_count, numpy >= 2, returns uint8)"""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(a).astype(np.int64)
    a = np.asarray(a, dtype=np.int64).copy()
    out = np.zeros(a.shape, dtype=np.int64)
    while np.any(a):
        out += a & 1
        a >>= 1
    return out


def _pc(x):
    return bin(x).count('1')


def pauli_mul(A, B):
    """(i^p1 X^x1 Z^z1)(i^p2 X^x2 Z^z2) = i^(p1+p2+2|z1&x2|) X^(x1^x2) Z^(z1^z2)"""
    x1, z1, p1 = A
    x2, z2, p2 = B
    return (x1 ^ x2, z1 ^ z2, (p1 + p2 + 2 * _pc(z1 & x2)) % 4)


def pauli_anticommute(A, B):
    return (_pc(A[0] & B[1]) + _pc(A[1] & B[0])) % 2 == 1


class Chain:
    """FFD chain of L spins with boundary condition 'pbc' (ring) or 'obc' (open chain)."""

    def __init__(self, L, bc='pbc'):
        bc = bc.lower()
        if bc not in ('pbc', 'obc'):
            raise ValueError("bc must be 'pbc' or 'obc'")
        if bc == 'pbc' and L < 5:
            raise ValueError('ring needs L >= 5')
        if bc == 'obc' and L < 3:
            raise ValueError('open chain needs L >= 3')
        self.L, self.bc = L, bc
        self.n_terms = L if bc == 'pbc' else L - 2

    def __repr__(self):
        return f'Chain(L={self.L}, bc={self.bc!r})'

    def term_pauli(self, j):
        """h_j = Z_j Z_{j+1} X_{j+2} as (x, z, p)"""
        L = self.L
        return (1 << ((j + 2) % L), (1 << (j % L)) | (1 << ((j + 1) % L)), 0)

    def placements(self, pat):
        """list of (coefficient, [term indices in product order]) of the translation sum of a pattern"""
        pat = pattern(pat)
        out = []
        for j in range(self.n_terms):
            for offs, c in pat.items():
                idx = [j + o for o in offs]
                if self.bc == 'pbc':
                    idx = [i % self.L for i in idx]
                elif max(idx) >= self.n_terms:
                    continue
                out.append((c, idx))
        return out

    def local_placements(self, pat, j):
        """the density of a pattern at term j (open chain: only the terms that exist)"""
        pat = pattern(pat)
        out = []
        for offs, c in pat.items():
            idx = [j + o for o in offs]
            if self.bc == 'pbc':
                idx = [i % self.L for i in idx]
            elif min(idx) < 0 or max(idx) >= self.n_terms:
                continue
            out.append((c, idx))
        return out

    def centre_terms(self, sym=True):
        """term index (indices) of a local probe in the middle of the chain, c = L//2 - 2; sym: c, c+1, c+2"""
        c = max(0, self.L // 2 - 2)
        return [c, c + 1, c + 2] if sym else [c]

    def word_pauli(self, idx):
        """ordered product of h_j, j in idx, as a Pauli string (x, z, p)"""
        P = (0, 0, 0)
        for j in idx:
            P = pauli_mul(P, self.term_pauli(j))
        return P


# ---------------------------------------------------------------------- dense matrices (validation, small L)
I2 = np.eye(2, dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z2 = np.diag([1.0, -1.0]).astype(complex)


def pauli_dense(P, L):
    """dense 2^L x 2^L matrix of a Pauli string (x, z, p); basis index n has spin s in bit s"""
    x, z, p = P
    n = np.arange(1 << L, dtype=np.int64)
    sign = 1 - 2 * (popcount(n & z) & 1)
    M = np.zeros((1 << L, 1 << L), dtype=complex)
    M[n ^ x, n] = (1j ** p) * sign
    return M


def dense_operator(chain, terms):
    """dense matrix of a list of (coefficient, [term indices]) on the full 2^L space"""
    D = 1 << chain.L
    M = np.zeros((D, D), dtype=complex)
    for c, idx in terms:
        M += c * pauli_dense(chain.word_pauli(idx), chain.L)
    return M


def dense_hamiltonian(chain):
    return dense_operator(chain, chain.placements('H'))


def pattern_pauli_terms(pat, offset_span=None):
    """spin-basis form of one density of a pattern placed at term 0 of an open chain:
    list of (coefficient, {spin offset: 2x2 matrix}) with spin offsets >= 0."""
    pat = pattern(pat)
    span = offset_span or (max(max(k) for k in pat) + 3)
    ch = Chain(span + 2, 'obc')
    out = []
    for offs, c in pat.items():
        x, z, p = ch.word_pauli(list(offs))
        coef = c * (1j ** p)
        ops = {}
        for s in range(ch.L):
            xs, zs = (x >> s) & 1, (z >> s) & 1
            if xs and zs:                     # X Z = -i Y
                ops[s] = Y2
                coef *= -1j
            elif xs or zs:
                ops[s] = X2 if xs else Z2
        out.append((coef, ops))
    return out
