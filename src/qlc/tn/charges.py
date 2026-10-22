"""Quasi-local charges of the periodic FFD chain as exact MPS in the basis of ordered products h_S.

Type I words of the note (Sec. 3.3; the Type II words of span N-1 vanish per site in the infinite chain): a word is a
sequence of blocks of consecutive h's, adjacent blocks separated by exactly one empty site, and its coefficient is
a product of block weights,

    lambda family, p < -1      blocks of length 1 or 2:  c(S) = i prod_blocks w(b),  w(1) = i sqrt(-1-p)/p,  w(2) = 1/p
    gamma family, -1 < p < 0   blocks of length b >= 2:  c(S) = i prod_blocks (-1)^b p (1+p)^{b/2-1}

(e.g. h_j: -sqrt(-1-p)/p, h_j h_{j+1}: i/p, h_j h_{j+2}: i(1+p)/p^2 for lambda; h_j h_{j+1}: ip,
h_j h_{j+1} h_{j+2}: -ip sqrt(1+p) for gamma).  The even / odd parts Q^{a,+-} keep the words of even / odd size;
since (-1)^{|S|} = prod_blocks (-1)^b they are (Q[w] +- Q[(-1)^b w]) / 2.

The word structure is a 5-state automaton along the chain (before / first site of a block / longer block / one
empty site after a block / after the word), so every charge is an MPS of bond dimension 5 (10 for a parity part):
`kind='sum'` gives the translation sum over the window, `kind='density'` the density whose first word sits at y0.
Sites: the same layout as WordMPO (two empty padding bits on each side of the m words).
"""
import math

import numpy as np

from ..model import pattern
from .wordmpo import WordMPO, _sort_word

PRE, B1, B2, GAP, POST = range(5)


def _block_rules(family, p):
    """(w_end1, w_end2, s_ext): block weights put on the transitions of the automaton"""
    if family == 'lambda':
        if not p < -1:
            raise ValueError('lambda family needs p < -1')
        return 1j * math.sqrt(-1 - p) / p, 1 / p, None
    if family == 'gamma':
        if not -1 < p < 0:
            raise ValueError('gamma family needs -1 < p < 0')
        return None, p, -math.sqrt(1 + p)
    raise ValueError("family must be 'lambda' or 'gamma'")


def _automaton(m, trans_at, n_states, init, final, be, dtype=complex):
    """MPS from per-site transition lists [(i, bit, j, weight)] on the m active words (pads: bit 0, identity)"""
    S = n_states
    tensors = []
    pad = np.zeros((S, 2, S), dtype=dtype)
    for i in range(S):
        pad[i, 0, i] = 1
    sites = [pad, pad] + [None] * m + [pad, pad]
    for k in range(m):
        T = np.zeros((S, 2, S), dtype=dtype)
        for i, bit, j, w in trans_at(k):
            T[i, bit, j] += w
        sites[k + 2] = T
    v0 = np.zeros(S, dtype=dtype)
    v0[init] = 1
    f = np.asarray(final, dtype=dtype)
    sites[0] = np.tensordot(v0, sites[0], (0, 0))[None]
    sites[-1] = np.tensordot(sites[-1], f, (2, 0))[:, :, None]
    return WordMPO([be.asarray(t, complex if dtype is complex else float) for t in sites], be, 0)


def charge_mps(family, p, m, be, kind='sum', y0=None, eta=1):
    """Q^{family}_p (all words) with an extra factor eta per occupied site (eta = -1: Q[(-1)^b w]), h_S basis"""
    w1, w2, s_ext = _block_rules(family, p)

    def trans(k):
        out = []
        if kind == 'sum' or k < y0:
            out.append((PRE, 0, PRE, 1))
        if kind == 'sum' or k == y0:
            out.append((PRE, 1, B1, 1j * eta))          # overall prefactor i on the first site of the word
        out.append((B1, 1, B2, eta))
        if w1 is not None:
            out.append((B1, 0, GAP, w1))
        if s_ext is not None:
            out.append((B2, 1, B2, eta * s_ext))
        out.append((B2, 0, GAP, w2))
        out.append((GAP, 1, B1, eta))
        out.append((GAP, 0, POST, 1))
        out.append((POST, 0, POST, 1))
        return out
    if kind == 'density' and not 0 <= y0 < m:
        raise ValueError('y0 outside the window')
    final = [0, w1 or 0, w2, 1, 1]
    return _automaton(m, trans, 5, PRE, final, be)


def charge_part_mps(family, p, parity, m, be, kind='sum', y0=None):
    """even (parity=+1) or odd (parity=-1) part (Q[w] + parity Q[(-1)^b w]) / 2"""
    a = charge_mps(family, p, m, be, kind, y0, eta=1)
    b = charge_mps(family, p, m, be, kind, y0, eta=-1)
    out = a.add(b, 0.5, 0.5 * parity)
    out.compress(10 ** 9, 1e-14)
    return out


def pattern_mps(pat, m, be, kind='sum', y0=None):
    """sum_P c_P h_{k0+P...} (ordered product) as an h_S-basis MPS: translation sum over the window or the density
    with base word y0; offsets are shifted to start at 0"""
    pat = pattern(pat)
    terms = []
    for offs, c in pat.items():
        lo = min(offs)
        sign, S = _sort_word([o - lo for o in offs])
        if S:
            terms.append((complex(c) * sign, tuple(x - S[0] for x in S)))
    # states: PRE, POST, then one chain per term for relative positions 1..span
    S_total = 2
    chains = []
    for c, S in terms:
        span = S[-1]
        chains.append((c, set(S), span, S_total))
        S_total += span
    POST_ = 1

    def trans(k):
        out = []
        if kind == 'sum' or k < y0:
            out.append((0, 0, 0, 1))
        out.append((POST_, 0, POST_, 1))
        for c, occ, span, base in chains:
            if kind == 'sum' or k == y0:
                out.append((0, 1, base if span else POST_, c))
            for r in range(1, span + 1):
                st = base + r - 1
                nxt = POST_ if r == span else st + 1
                out.append((st, 1 if r in occ else 0, nxt, 1))
        return out
    final = np.zeros(S_total)
    final[POST_] = 1
    return _automaton(m, trans, S_total, 0, final, be)


def words_of(family, p, m, parity=0):
    """reference enumeration: {sorted word tuple: h_S coefficient} of all Type I words in m words (parity 0: all,
    +1 / -1: even / odd size), first word anywhere"""
    import itertools
    w1, w2, s_ext = _block_rules(family, p)
    out = {}
    for bits in itertools.product((0, 1), repeat=m):
        S = tuple(j for j in range(m) if bits[j])
        if not S:
            continue
        if parity and (-1) ** len(S) != parity:
            continue
        blocks, gaps, cur = [], [], 1
        for a, b in zip(S, S[1:]):
            if b == a + 1:
                cur += 1
            else:
                blocks.append(cur)
                gaps.append(b - a - 1)
                cur = 1
        blocks.append(cur)
        if any(g != 1 for g in gaps):
            continue
        c = 1j
        ok = True
        for b in blocks:
            if family == 'lambda':
                if b == 1:
                    c *= w1
                elif b == 2:
                    c *= w2
                else:
                    ok = False
            else:
                if b < 2:
                    ok = False
                else:
                    c *= (-1) ** b * p * (1 + p) ** (b / 2 - 1)
        if ok:
            out[S] = c
    return out
