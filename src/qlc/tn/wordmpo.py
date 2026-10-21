"""Operators of the FFD chain as MPS over h-word occupation bits ("word MPO") on an open window.

A window of L spins carries the m = L - 2 words h_0..h_{m-1}; the MPO has m + 4 sites: two vacuum bits on each side
(always 0) so that every word is the centre of a 5-site block.  Word w sits on MPO site w + 2.

Correlations use time folding and translation (rho commutes with H, H is real):

    C_x(t) = omega(rho a_x(t/2) b_0(-t/2)) / omega(rho) = kappa < T_x M_a(tau), K M_b(tau) K >_HS / Z_x,   tau = t/2,
    M(tau) = e^{iH tau} (o_0 rho^{1/2}) e^{-iH tau},      Z_x = < T_x rho^{1/2}, rho^{1/2} >,

with kappa = -1 for K-odd operators (currents: K j K = -j) and +1 for K-even ones (energy density).  In
coefficients  C_x = kappa conj( sum_S Ma_{S-x} Mb_S (-1)^{a(S)} ) / Z_x  (`shifted_overlap`).  At beta = 0,
rho^{1/2} = 1 and Z_x = 1 exactly; at beta > 0 Z_x divides out the window-edge mismatch of the shifted thermal
factor.  Generalised from remainder/tn/thermal_word.py of the 2026-10-20 package.

At beta = 0 the evolved current has real coefficients (Hermitian operator, Hermitian basis, real rotation gates),
and rho^{1/2} is Hermitian, so both are kept in real arithmetic; M = o_0 rho^{1/2} at beta > 0 is complex.
"""
from .mps import MPS
from . import hword


def _sort_word(offs):
    """ordered product h_{w1} h_{w2} ... = sign * prod_{s in S} h_s (sorted S, h^2 = 1)"""
    w = list(offs)
    sign = 1
    for i in range(len(w)):                         # bubble sort with anticommutation signs
        for j in range(len(w) - 1 - i):
            if w[j] > w[j + 1]:
                if w[j] - w[j + 1] in (1, 2):
                    sign = -sign
                w[j], w[j + 1] = w[j + 1], w[j]
    out = []
    for x in w:
        if out and out[-1] == x:
            out.pop()
        else:
            out.append(x)
    return sign, tuple(out)


def pattern_words(pat, k0):
    """{sorted word tuple S: coefficient of B_S} of the density sum_P c_P h_{k0+P...}"""
    words = {}
    for offs, c in pat.items():
        sign, S = _sort_word([k0 + o for o in offs])
        coef = c * sign * (1j) ** (-hword.a_count(S))      # prod h_S = i^{-a(S)} B_S
        words[S] = words.get(S, 0) + coef
    return {S: c for S, c in words.items() if abs(c) > 1e-15}


def k_parity(pat):
    """kappa = +1 / -1 if the pattern is even / odd under time reversal K (Hermitian patterns only)"""
    par = set()
    for S, c in pattern_words(pat, 0).items():
        a = hword.a_count(S)
        if abs(c.imag) > 1e-12 * abs(c):
            raise ValueError('pattern is not Hermitian (complex B_S coefficient)')
        par.add(a % 2)
    if len(par) != 1:
        raise ValueError('pattern mixes K-even and K-odd words; split it')
    return 1 if par.pop() == 0 else -1


class WordMPO(MPS):
    PAD = 2

    @classmethod
    def identity(cls, n_words, be):
        e0 = be.asarray([1.0, 0.0], float).reshape(1, 2, 1)
        return cls([be.copy(e0) for _ in range(n_words + 2 * cls.PAD)], be, 0)

    @classmethod
    def from_words(cls, n_words, words, be, chi=10 ** 9, tol=1e-14):
        """sum_S c_S B_S from {word tuple: coefficient} (real tensors if all coefficients are real)"""
        real = all(abs(complex(c).imag) == 0 for c in words.values())
        dt = float if real else complex
        n = n_words + 2 * cls.PAD
        total = None
        for S, c in words.items():
            occ = set(w + cls.PAD for w in S)
            ts = []
            for k in range(n):
                v = [0.0, 1.0] if k in occ else [1.0, 0.0]
                ts.append(be.asarray(v, dt).reshape(1, 2, 1))
            ts[0] = ts[0] * (complex(c).real if real else complex(c))
            m = cls(ts, be, 0)
            total = m if total is None else total.add(m)
        total.init_loss = total.compress(chi, tol)
        return total

    @property
    def n_words(self):
        return len(self.A) - 2 * self.PAD

    def _gate(self, name, x):
        cache = self.__dict__.setdefault('_gates', {})
        real = name == 'rot' and not self.is_complex()
        key = (name, round(x, 15), real)
        if key not in cache:
            G = {'rot': hword.rotation_matrix, 'imag': hword.imag_matrix}[name](x)
            cache[key] = self.be.asarray(G.real if real else G, float if real else complex)
        return cache[key]

    def left_mult(self, word, tol=1e-14):
        """X -> h_word X (exact up to numerical zeros)"""
        self.make_complex()
        G = self.__dict__.get('_lmult')
        if G is None:
            G = self._lmult = self.be.asarray(hword.LMULT)
        self.apply_gate(word, G, 5, 10 ** 9, tol, 'right')
        return self

    def times_pattern(self, pat, k0, chi, tol=1e-13):
        """sum_P c_P (h_{k0+P_1} h_{k0+P_2} ...) X  as a new compressed WordMPO"""
        total, loss = None, 0.0
        for offs, c in pat.items():
            m = self.copy()
            for o in reversed(offs):
                m.left_mult(k0 + o)
            m.scale(complex(c))
            total = m if total is None else total.add(m)
            loss += total.compress(chi, tol)
        total.init_loss = loss
        return total

    def edge_weight(self):
        """occupation weight of the outermost words, max(left, right), relative to the squared norm.
        At beta = 0 this measures how much of the evolved operator has reached the window boundary."""
        be = self.be
        out = []
        for site in (self.PAD, self.PAD + self.n_words - 1):
            self.move_center(site)
            A = self.A[site]
            n2 = be.norm(A) ** 2
            out.append(be.norm(A[:, 1, :]) ** 2 / n2 if n2 > 0 else 0.0)
        return max(out)

    def sweep(self, kind, step, chi, tol):
        """palindromic second-order product of the 5-site maps of all words; kind 'rot' (Heisenberg e^{iH step})
        or 'imag' (left multiplication by e^{-step H}).  Returns the discarded weight."""
        if kind == 'imag':
            self.make_complex()
        last = len(self.A) - 5
        half, full = self._gate(kind, step / 2), self._gate(kind, step)
        loss = 0.0
        for j in range(last):
            loss += self.apply_gate(j, half, 5, chi, tol, 'right')
        loss += self.apply_gate(last, full, 5, chi, tol, 'left')
        for j in range(last - 1, -1, -1):
            loss += self.apply_gate(j, half, 5, chi, tol, 'left')
        return loss

    def coefficients(self):
        """dense coefficient vector over the m words (2^m entries, word 0 most significant); small windows only"""
        be = self.be
        T = self.A[0]
        for a in self.A[1:]:
            T = be.tensordot(T, a, (-1, 0))
        T = be.to_numpy(T).reshape([2] * len(self.A))
        idx = (0, 0) + (slice(None),) * self.n_words + (0, 0)
        return T[idx].reshape(-1)


def shifted_overlap(a, b, x, signed=True):
    """sum_S a_{S-x} b_S [(-1)^{a(S)} if signed]  (no complex conjugation; vacuum padding), x >= 0.

    The environment carries the last two occupation bits (4 states) to count pairs at distance 1 and 2."""
    be = a.be
    cplx = a.is_complex() or b.is_complex()
    P = (lambda t: be.to_complex(t) if not be.is_complex(t) else t) if cplx else (lambda t: t)
    n = len(a.A)
    env = be.zeros((4, 1, 1), complex if cplx else float)
    env[0, 0, 0] = 1.0

    def recombine(T0, T1):
        # new history (b_{i-1}, p) from hist h = 2 b_{i-2} + b_{i-1}; sign (-1)^{p (b_{i-2} + b_{i-1})}
        e0 = T0[0] + T0[2]
        e2 = T0[1] + T0[3]
        if T1 is None:
            z = e0 * 0
            return be.concat([e0[None], z[None], e2[None], z[None]], 0)
        if signed:
            e1, e3 = T1[0] - T1[2], T1[3] - T1[1]
        else:
            e1, e3 = T1[0] + T1[2], T1[3] + T1[1]
        return be.concat([e0[None], e1[None], e2[None], e3[None]], 0)

    for i in range(n + x):
        if i < x:                                   # a is vacuum here
            env = recombine(env @ P(b.A[i][:, 0, :]), None)
        elif i >= n:                                # b is vacuum here
            env = recombine(P(a.A[i - x][:, 0, :]).T @ env, None)
        else:
            A, B = P(a.A[i - x]), P(b.A[i])
            T0 = A[:, 0, :].T @ (env @ B[:, 0, :])
            T1 = A[:, 1, :].T @ (env @ B[:, 1, :])
            env = recombine(T0, T1)
    return complex(be.to_numpy(env.sum(0))[0, 0])


def folded_profile(Ma, Mb, rho, xmax, kappa=-1):
    """v_x = kappa conj(shifted_overlap(Ma, Mb, x)) / Z_x for x = 0..xmax (rho None: beta = 0, Z_x = 1)"""
    out = []
    for x in range(xmax + 1):
        den = 1.0 if rho is None else shifted_overlap(rho, rho, x, signed=False).real
        out.append(kappa * shifted_overlap(Ma, Mb, x, True).conjugate() / den)
    return out
