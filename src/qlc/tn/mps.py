"""Open-boundary MPS with an explicit orthogonality centre, k-site gates and SVD truncation (backend aware).

Tensors have shape (left bond, d, right bond).  Used with d = 2 for operators in the h-word basis (word MPO,
5-site gates) and with d = 4 for spin-basis purifications (3-site gates).  Tensors are real (float64) when the
represented coefficients are real, which makes the SVDs about 3x cheaper; see WordMPO.
"""
import math


def keep_rank(s, chi, tol):
    """number of singular values kept: those above tol * s[0], at most chi, at least one"""
    s0 = float(s[0])
    if s0 <= 0:
        return 1
    return max(1, min(int(chi), int((s > tol * s0).sum())))


def imaginary_steps(beta, dtau):
    """(number of steps, step) reaching beta / 2 exactly with steps <= dtau"""
    if beta < 0:
        raise ValueError('beta must be >= 0')
    if beta == 0:
        return 0, 0.0
    n = max(1, int(math.ceil(beta / (2 * dtau) - 1e-9)))
    return n, beta / (2 * n)


def truncation(be, s, chi, tol):
    """(kept rank, discarded weight sum of s^2) from one host copy of the singular values"""
    s_h = be.to_numpy(s)
    rk = keep_rank(s_h, chi, tol)
    return rk, float((s_h[rk:] ** 2).sum())


class MPS:
    def __init__(self, tensors, be, center=0):
        self.A = list(tensors)
        self.be = be
        self.center = center

    # ------------------------------------------------------------------ basics
    def __len__(self):
        return len(self.A)

    @property
    def d(self):
        return self.A[0].shape[1]

    def copy(self):
        return type(self)([self.be.copy(a) for a in self.A], self.be, self.center)

    def bond(self):
        return max(a.shape[2] for a in self.A)

    def bonds(self):
        return [a.shape[2] for a in self.A[:-1]]

    def is_complex(self):
        return self.be.is_complex(self.A[0])

    def make_complex(self):
        if not self.is_complex():
            self.A = [self.be.to_complex(a) for a in self.A]
        return self

    def move_center(self, to):
        be = self.be
        while self.center < to:
            k = self.center
            l, d, r = self.A[k].shape
            q, R = be.qr(self.A[k].reshape(l * d, r))
            self.A[k] = q.reshape(l, d, q.shape[1])
            self.A[k + 1] = be.tensordot(R, self.A[k + 1], (1, 0))
            self.center += 1
        while self.center > to:
            k = self.center
            l, d, r = self.A[k].shape
            q, R = be.qr(self.A[k].reshape(l, d * r).T)
            self.A[k] = q.T.reshape(q.shape[1], d, r)
            self.A[k - 1] = be.tensordot(self.A[k - 1], R.T, (2, 0))
            self.center -= 1

    def canonicalize(self, to=None):
        """bring an MPS of unknown gauge into canonical form with the centre at `to` (default: last site)"""
        self.center = 0
        self.move_center(len(self.A) - 1)
        if to is not None:
            self.move_center(to)

    def norm(self):
        return self.be.norm(self.A[self.center])

    def normalize(self):
        """divide by the norm (canonical form required); returns log(norm)"""
        nrm = self.norm()
        self.A[self.center] = self.A[self.center] / nrm
        return math.log(nrm)

    def scale(self, c):
        if isinstance(c, complex) and c.imag != 0:
            self.make_complex()
        elif isinstance(c, complex):
            c = c.real
        self.A[self.center] = self.A[self.center] * c

    # ------------------------------------------------------------------ gates
    def apply_gate(self, i, G, k, chi, tol, direction='right'):
        """apply a d^k x d^k matrix G (same dtype as the tensors) on sites i..i+k-1 and split back by SVDs;
        returns the discarded weight.  direction 'right' leaves the centre at i+k-1, 'left' at i."""
        be = self.be
        if not (i <= self.center <= i + k - 1):
            self.move_center(i if self.center < i else i + k - 1)
        d = self.d
        th = self.A[i]
        for q in range(1, k):
            th = be.tensordot(th, self.A[i + q], (-1, 0))
        l, r = th.shape[0], th.shape[-1]
        th = th.reshape(l, d ** k, r)
        th = be.einsum('ab,lbr->lar', G, th)
        loss = 0.0
        if direction == 'right':
            left, tail = l, th
            for q in range(k - 1):
                u, s, v = be.svd(tail.reshape(left * d, -1))
                rk, w = truncation(be, s, chi, tol)
                loss += w
                self.A[i + q] = u[:, :rk].reshape(left, d, rk)
                tail = s[:rk, None] * v[:rk]
                left = rk
            self.A[i + k - 1] = tail.reshape(left, d, r)
            self.center = i + k - 1
        else:
            right, tail = r, th
            for q in range(k - 1, 0, -1):
                u, s, v = be.svd(tail.reshape(-1, d * right))
                rk, w = truncation(be, s, chi, tol)
                loss += w
                self.A[i + q] = v[:rk].reshape(rk, d, right)
                tail = u[:, :rk] * s[:rk]
                right = rk
            self.A[i] = tail.reshape(l, d, right)
            self.center = i
        return loss

    def compress(self, chi, tol):
        """optimal sequential SVD truncation (canonical sweep); returns the discarded weight"""
        be = self.be
        self.canonicalize()
        loss = 0.0
        for k in range(len(self.A) - 1, 0, -1):
            l, d, r = self.A[k].shape
            u, s, v = be.svd(self.A[k].reshape(l, d * r))
            rk, w = truncation(be, s, chi, tol)
            loss += w
            self.A[k] = v[:rk].reshape(rk, d, r)
            self.A[k - 1] = be.tensordot(self.A[k - 1], u[:, :rk] * s[:rk], (2, 0))
        self.center = 0
        return loss

    # ------------------------------------------------------------------ algebra
    def add(self, other, ca=1.0, cb=1.0):
        """ca * self + cb * other as a new MPS with summed bonds (canonicalised, not truncated)"""
        be = self.be
        cplx = self.is_complex() or other.is_complex() or isinstance(ca, complex) or isinstance(cb, complex)
        P = (lambda a: be.to_complex(a) if not be.is_complex(a) else a) if cplx else (lambda a: a)
        out = []
        n = len(self.A)
        for k in range(n):
            A, B = P(self.A[k]), P(other.A[k])
            if k == 0:
                A, B = A * ca, B * cb
            if n == 1:
                out.append(A + B)
            elif k == 0:
                out.append(be.concat([A, B], axis=2))
            elif k == n - 1:
                out.append(be.concat([A, B], axis=0))
            else:
                la, d, ra = A.shape
                lb, _, rb = B.shape
                T = be.zeros((la + lb, d, ra + rb), complex if cplx else float)
                T[:la, :, :ra] = A
                T[la:, :, ra:] = B
                out.append(T)
        m = type(self)(out, be, 0)
        m.canonicalize()
        return m

    def real_part(self):
        """MPS of the real part of the coefficients, in real arithmetic (bond dimension doubles before compression).
        A complex bond vector v_r + i v_i is carried as the real vector (v_r, v_i)."""
        be = self.be
        if not self.is_complex():
            return self.copy()
        n = len(self.A)
        out = []
        for k, A in enumerate(self.A):
            Ar, Ai = be.real_part(A), be.real_part(A * (-1j))      # A = Ar + i Ai
            top = be.concat([Ar, Ai], axis=2)                     # input v_r
            bot = be.concat([-Ai, Ar], axis=2)                    # input v_i
            T = be.concat([top, bot], axis=0)                     # (2l, d, 2r)
            if k == 0:
                T = T[:A.shape[0]]
            if k == n - 1:
                T = T[:, :, :A.shape[2]]
            out.append(T)
        m = type(self)(out, be, 0)
        m.canonicalize()
        return m

    def overlap(self, other):
        """<self|other> = sum conj(self) other"""
        be = self.be
        C = lambda a: be.to_complex(a) if not be.is_complex(a) else a
        E = be.asarray([[1.0]])
        for a, b in zip(self.A, other.A):
            E = be.tensordot(be.tensordot(E, C(a).conj(), (0, 0)), C(b), ([0, 1], [0, 1]))
        return complex(be.to_numpy(E)[0, 0])

    # ------------------------------------------------------------------ checkpoint helpers
    def to_numpy(self):
        return dict(A=[self.be.to_numpy(a) for a in self.A], center=self.center)

    @classmethod
    def from_numpy(cls, d, be):
        import numpy as np
        return cls([be.asarray(a, complex if np.iscomplexobj(a) else float) for a in d['A']], be, d['center'])
