"""Representations of the h-algebra by signed XOR permutations.

In every representation used here a word of the h_j acts on a basis |n>, n in [0, 2^k), as

    |n>  ->  phase * (-1)^{|b & n|} |n ^ a>          ("term" (a, b, phase))

* full representation: the physical spin space, k = L; a word acts as its Pauli string X^a Z^b.
* reduced representation: the h_j generate an algebra whose irreducible blocks have dimension 2^k with
  n_terms = 2k + c, c = dimension of the radical of the commutation form.  A symplectic basis {a_i, b_i, z_l} of the
  word space gives the algebra isomorphism P(a_i) -> X_i, P(b_i) -> Z_i, P(z_l) -> s_l = +-1 on k logical qubits.
  Each sector s in {+-1}^c is one irreducible block, and every sector occurs with the same multiplicity
  2^(L-k-c) in the spin space, so  Tr_spin(A) = 2^(L-k-c) sum_s Tr_s(A).  Normalised quantities such as
  Tr(rho J(t) J) / Tr(rho) need only the blocks.  On a ring, sectors related by translation have identical
  spectra and current matrix elements, so one sector per translation orbit is enough (weight = orbit size).

  Sizes (ring): L = 24 -> 2^10, L = 30 -> 2^13, L = 33 -> 2^15 (instead of 2^L).

Generalised from q2_finiteT/finiteT_numerics/ringred.py and secorbits.py of the 2026-10-20 package.
"""
import itertools
import numpy as np

from ..model import Chain, pauli_mul, popcount, _pc


def _herm(x, z):
    """Hermitian Pauli string with the given masks: i^{|x&z|} X^x Z^z"""
    return (x, z, _pc(x & z) % 4)


def opmul(o1, o2):
    """(p X^a Z^b)(p' X^a' Z^b') = p p' (-1)^{|b&a'|} X^{a^a'} Z^{b^b'} for terms (a, b, phase)"""
    a1, b1, p1 = o1
    a2, b2, p2 = o2
    return (a1 ^ a2, b1 ^ b2, p1 * p2 * (-1) ** _pc(b1 & a2))


class Representation:
    """One block: images (a_j, b_j, phase_j) of the generators h_j on 2^k states."""

    def __init__(self, chain, k, gens, label='full', weight=1, sector=None):
        self.chain, self.k, self.gens = chain, k, gens
        self.dim = 1 << k
        self.label, self.weight, self.sector = label, weight, sector

    def word(self, idx):
        o = (0, 0, 1.0 + 0j)
        for j in idx:
            o = opmul(o, self.gens[j])
        return o

    def terms(self, placements):
        """merge a list of (coefficient, [term indices]) into distinct terms (a, b, phase)"""
        acc = {}
        for c, idx in placements:
            a, b, p = self.word(idx)
            acc[(a, b)] = acc.get((a, b), 0) + c * p
        return [(a, b, p) for (a, b), p in acc.items() if abs(p) > 1e-14]

    def op_terms(self, pat):
        return self.terms(self.chain.placements(pat))

    def local_terms(self, pat, centres):
        pl = []
        for c in centres:
            pl += [(coef / len(centres), idx) for coef, idx in self.chain.local_placements(pat, c)]
        return self.terms(pl)

    # ------------------------------------------------------------------ matrices
    def dense(self, terms, be):
        """dense matrix sum_t phase_t (-1)^{|b_t & n|} |n ^ a_t><n| on the backend"""
        D = self.dim
        n = np.arange(D, dtype=np.int64)
        M = be.zeros((D, D))
        for a, b, p in terms:
            vals = p * (1 - 2 * (popcount(n & b) & 1))
            be.add_at(M, be.asarray(n ^ a, int), be.asarray(n, int), be.asarray(vals))
        return M

    def applier(self, terms, be):
        """precomputed gather form: (O V)[m] = sum_t phase_t (-1)^{|b_t & (m^a_t)|} V[m ^ a_t]"""
        D = self.dim
        m = np.arange(D, dtype=np.int64)
        src, sgn = [], []
        for a, b, p in terms:
            s = m ^ a
            src.append(be.asarray(s, int))
            sgn.append(be.asarray(p * (1 - 2 * (popcount(s & b) & 1)))[:, None])
        return list(zip(src, sgn))


def apply_terms(app, V, be):
    """O V for an applier (list of (source rows, signs)); an empty operator gives zeros"""
    out = be.zeros(tuple(V.shape))
    for src, sgn in app:
        out = out + sgn * be.take_rows(V, src)
    return out


def is_hermitian(terms):
    """terms (a, b, phase) of sum phase X^a Z^b: Hermitian iff phase = conj(phase) (-1)^{|a & b|} for every term"""
    from ..model import _pc
    return all(abs(p - np.conj(p) * (-1) ** _pc(a & b)) <= 1e-12 * max(1.0, abs(p)) for a, b, p in terms)


# ---------------------------------------------------------------------- full representation
def full_representation(chain):
    gens = []
    for j in range(chain.n_terms):
        x, z, p = chain.term_pauli(j)
        gens.append((x, z, 1j ** p))
    return Representation(chain, chain.L, gens, label='full', weight=1)


# ---------------------------------------------------------------------- reduced representation
class ReducedStructure:
    """symplectic decomposition of the word space of the h_j of a chain"""

    def __init__(self, chain):
        self.chain = chain
        M = chain.n_terms
        hs = [chain.term_pauli(j) for j in range(M)]
        self._check_independent(hs, chain.L)
        Om = [0] * M
        for i in range(M):
            for j in range(M):
                if i != j and (_pc(hs[i][0] & hs[j][1]) + _pc(hs[i][1] & hs[j][0])) % 2:
                    Om[i] |= 1 << j
        self.Om = Om

        def omega(u, v):
            s = 0
            while u:
                i = (u & -u).bit_length() - 1
                s += _pc(Om[i] & v)
                u &= u - 1
            return s % 2
        self.omega = omega

        pool = [1 << j for j in range(M)]
        A, B, Zc = [], [], []
        while pool:
            u = pool.pop(0)
            idx = next((t for t, w in enumerate(pool) if omega(u, w)), None)
            if idx is None:
                Zc.append(u)
                continue
            w = pool.pop(idx)
            A.append(u)
            B.append(w)
            pool = [v ^ (u if omega(v, w) else 0) ^ (w if omega(v, u) else 0) for v in pool]
        self.k, self.c = len(A), len(Zc)
        assert 2 * self.k + self.c == M
        for i in range(self.k):
            for j in range(self.k):
                assert omega(A[i], B[j]) == (i == j) and omega(A[i], A[j]) == 0 and omega(B[i], B[j]) == 0
            for z in Zc:
                assert omega(A[i], z) == 0 and omega(B[i], z) == 0
        for z in Zc:
            for g in range(M):
                assert omega(z, 1 << g) == 0
        self.A, self.B, self.Zc = A, B, Zc
        self.PA = [_herm(*self.word_pauli(u)[:2]) for u in A]
        self.PB = [_herm(*self.word_pauli(u)[:2]) for u in B]
        self.PZ = [_herm(*self.word_pauli(u)[:2]) for u in Zc]
        # elimination table for the central part
        self._zrows = []
        for l, z in enumerate(Zc):
            v, comb = z, 1 << l
            for pb, pv, pcmb in self._zrows:
                if (v >> pb) & 1:
                    v ^= pv
                    comb ^= pcmb
            assert v
            self._zrows.append(((v & -v).bit_length() - 1, v, comb))
        self.gens = []
        for j in range(M):
            e = 1 << j
            alpha = beta = 0
            r = e
            for i in range(self.k):
                if omega(e, B[i]):
                    alpha |= 1 << i
                    r ^= A[i]
                if omega(e, A[i]):
                    beta |= 1 << i
                    r ^= B[i]
            gamma = self.zsolve(r)
            P = (0, 0, 0)
            for i in range(self.k):
                if (alpha >> i) & 1:
                    P = pauli_mul(P, self.PA[i])
                if (beta >> i) & 1:
                    P = pauli_mul(P, self.PB[i])
            for l in range(self.c):
                if (gamma >> l) & 1:
                    P = pauli_mul(P, self.PZ[l])
            assert P[0] == hs[j][0] and P[1] == hs[j][1]
            self.gens.append((alpha, beta, gamma, 1j ** ((hs[j][2] - P[2]) % 4)))

    @staticmethod
    def _check_independent(hs, L):
        rows = []
        for x, z, _ in hs:
            v = x | (z << L)
            for r in rows:
                v = min(v, v ^ r)
            assert v, 'the h_j are not independent Pauli strings'
            rows.append(v)

    def word_pauli(self, u):
        P = (0, 0, 0)
        j = 0
        while u:
            if u & 1:
                P = pauli_mul(P, self.chain.term_pauli(j))
            u >>= 1
            j += 1
        return P

    def zsolve(self, r):
        comb = 0
        for pb, pv, pcmb in self._zrows:
            if (r >> pb) & 1:
                r ^= pv
                comb ^= pcmb
        assert r == 0
        return comb

    @property
    def multiplicity_log2(self):
        return self.chain.L - self.k - self.c

    def sectors(self):
        return list(itertools.product([1, -1], repeat=self.c))

    def representation(self, sector, weight=1):
        gens = []
        for alpha, beta, gamma, phase in self.gens:
            s = 1
            for l in range(self.c):
                if (gamma >> l) & 1:
                    s *= sector[l]
            gens.append((alpha, beta, phase * s))
        si = self.sectors().index(tuple(sector))
        return Representation(self.chain, self.k, gens, label=f'sector{si}', weight=weight, sector=tuple(sector))

    # ------------------------------------------------------------------ translation orbits (ring only)
    def orbits(self):
        """sectors grouped into translation orbits (ring); open chain: every sector on its own"""
        secs = self.sectors()
        if self.chain.bc != 'pbc':
            return [[s] for s in secs]
        N = self.chain.n_terms
        rel = []
        for l in range(self.c):
            u = self.Zc[l]
            zs = ((u << 1) | (u >> (N - 1))) & ((1 << N) - 1)
            comb = self.zsolve(zs)
            P = (0, 0, 0)
            for m in range(self.c):
                if (comb >> m) & 1:
                    P = pauli_mul(P, self.PZ[m])
            Ps = _herm(*self.word_pauli(zs)[:2])
            assert P[0] == Ps[0] and P[1] == Ps[1]
            rel.append((comb, (-1) ** (((Ps[2] - P[2]) % 4) // 2)))

        def prod(sp, comb):
            v = 1
            for m in range(len(sp)):
                if (comb >> m) & 1:
                    v *= sp[m]
            return v
        img = {}
        for s in secs:
            cands = [sp for sp in secs if all(phi * prod(sp, comb) == s[l] for l, (comb, phi) in enumerate(rel))]
            assert len(cands) == 1
            img[s] = cands[0]
        seen, orbs = set(), []
        for s in secs:
            if s in seen:
                continue
            orb = [s]
            x = img[s]
            while x != s:
                orb.append(x)
                x = img[x]
            seen |= set(orb)
            orbs.append(orb)
        return orbs

    def representations(self, use_orbits=True):
        """one representation per translation orbit (weight = orbit size), or every sector (weight 1).
        Orbits may only be used for translation-invariant operators (not for a local probe)."""
        if not use_orbits:
            return [self.representation(s) for s in self.sectors()]
        return [self.representation(o[0], weight=len(o)) for o in self.orbits()]


def reduced_representations(chain, use_orbits=True):
    return ReducedStructure(chain).representations(use_orbits)


def representations(chain, method='reduced', use_orbits=True):
    if method == 'reduced':
        return reduced_representations(chain, use_orbits)
    if method == 'full':
        return [full_representation(chain)]
    raise ValueError("method must be 'reduced' or 'full'")


def describe(L, bc='pbc'):
    """(k, c, number of sector orbits, block dimension) of a chain"""
    rs = ReducedStructure(Chain(L, bc))
    return dict(L=L, bc=bc, k=rs.k, c=rs.c, dim=1 << rs.k, sectors=1 << rs.c, orbits=len(rs.orbits()))
