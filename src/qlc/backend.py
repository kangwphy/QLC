"""Array backends: NumPy (default) or PyTorch on CPU / GPU.

All heavy linear algebra in qlc goes through a backend object, so the same code runs on CPU and GPU::

    be = get_backend()                          # NumPy
    be = get_backend('torch', device='cuda')    # NVIDIA GPU; ROCm / Hygon-DCU builds of torch also use 'cuda'

Everything is double precision (complex128 / float64).  Apple 'mps' devices have no complex128 and are refused.
The backend only wraps the few operations whose spelling differs between NumPy and torch; ordinary arithmetic,
slicing, ``reshape``, ``@`` and ``.conj()`` are used directly on the arrays.
"""
import numpy as np


class NumpyBackend:
    name = 'numpy'
    device = 'cpu'

    # ------------------------------------------------------------------ creation / conversion
    def asarray(self, x, dtype=complex):
        return np.asarray(x, dtype=dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def zeros(self, shape, dtype=complex):
        return np.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=complex):
        return np.eye(n, dtype=dtype)

    def arange(self, n):
        return np.arange(n)

    # ------------------------------------------------------------------ shape
    def permute(self, a, axes):
        return np.transpose(a, axes)

    def concat(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def copy(self, a):
        return a.copy()

    # ------------------------------------------------------------------ contraction
    def tensordot(self, a, b, axes):
        return np.tensordot(a, b, axes=axes)

    def einsum(self, spec, *ops):
        return np.einsum(spec, *ops, optimize=True)

    # ------------------------------------------------------------------ linear algebra
    def svd(self, m):
        """thin SVD m = U diag(s) Vh; falls back to LAPACK gesvd if gesdd does not converge"""
        try:
            return np.linalg.svd(m, full_matrices=False)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd
            return svd(m, full_matrices=False, lapack_driver='gesvd')

    def host(self, s):
        """host (NumPy) copy of a small vector such as singular values"""
        return s

    def qr(self, m):
        return np.linalg.qr(m)

    def eigh(self, h):
        """eigh (LAPACK heevd); h may be overwritten.  Needs about 2 D^2 (complex) beyond h itself."""
        from scipy.linalg import eigh
        return eigh(h, overwrite_a=True, check_finite=False, driver='evd')

    def norm(self, a):
        return float(np.linalg.norm(a))

    # ------------------------------------------------------------------ elementwise
    def exp(self, a):
        return np.exp(a)

    def abs2(self, a):
        return a.real ** 2 + a.imag ** 2

    def is_complex(self, a):
        return np.iscomplexobj(a)

    def real_part(self, a):
        return np.ascontiguousarray(a.real)

    def to_complex(self, a):
        return a.astype(complex)

    def add_at(self, mat, rows, cols, vals):
        """mat[rows, cols] += vals for index pairs without repetition"""
        mat[rows, cols] += vals

    def take_rows(self, a, idx):
        return a[idx]

    def sync(self):
        pass


class TorchBackend(NumpyBackend):
    name = 'torch'

    def __init__(self, device=None):
        import torch
        self.torch = torch
        if device in (None, 'gpu', 'auto'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'mps':
            raise ValueError("Apple 'mps' has no complex128; use device='cpu' or a CUDA/ROCm GPU")
        self.svd_fallbacks = 0
        self._dtypes = {complex: torch.complex128, float: torch.float64, int: torch.int64, bool: torch.bool,
                        np.complex128: torch.complex128, np.float64: torch.float64, np.int64: torch.int64}

    def _dt(self, dtype):
        return self._dtypes[dtype]

    def asarray(self, x, dtype=complex):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=self._dt(dtype))
        return t.as_tensor(np.asarray(x, dtype=dtype), device=self.device)

    def to_numpy(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.detach().resolve_conj().cpu().numpy()
        return np.asarray(x)

    def zeros(self, shape, dtype=complex):
        return self.torch.zeros(shape, dtype=self._dt(dtype), device=self.device)

    def eye(self, n, dtype=complex):
        return self.torch.eye(n, dtype=self._dt(dtype), device=self.device)

    def arange(self, n):
        return self.torch.arange(n, device=self.device)

    def permute(self, a, axes):
        return a.permute(*axes)

    def concat(self, arrays, axis=0):
        return self.torch.cat(arrays, dim=axis)

    def copy(self, a):
        return a.clone()

    def tensordot(self, a, b, axes):
        if not isinstance(axes, int):
            axes = tuple([x] if isinstance(x, int) else list(x) for x in axes)
        return self.torch.tensordot(a, b, dims=axes)

    def einsum(self, spec, *ops):
        return self.torch.einsum(spec, *ops)

    def svd(self, m):
        """torch SVD, validated: some builds (e.g. torch on macOS / Accelerate) return wrong factors for
        rank-deficient matrices without raising.  Fallbacks: driver 'gesvd' (CUDA), then SciPy on the CPU;
        the dtype of the input is kept.  The checks need one device-to-host transfer (shared with `host(s)`)."""
        t = self.torch
        try:
            u, s, v = t.linalg.svd(m, full_matrices=False)
            if self._svd_ok(m, u, s, v):
                return u, s, v
        except Exception:
            pass
        self.svd_fallbacks += 1
        if m.is_cuda:
            try:
                u, s, v = t.linalg.svd(m, full_matrices=False, driver='gesvd')
                if self._svd_ok(m, u, s, v):
                    return u, s, v
            except Exception:
                pass
        u, s, v = _NP.svd(self.to_numpy(m))
        dt = complex if np.iscomplexobj(u) else float
        return self.asarray(u, dt), self.asarray(s, float), self.asarray(v, dt)

    def _svd_ok(self, m, u, s, v, rtol=1e-9, tol=1e-8):
        """O(mn) checks on the device: sum s^2 = |m|_F^2, m x = U s Vh x, U and Vh orthonormal on probe vectors;
        all scalars and s are moved to the host in one transfer"""
        t = self.torch
        x, nx = _probe(self, m.shape[1], m, 0)
        y, ny = _probe(self, u.shape[1], u, 1)
        vals = t.stack([t.linalg.norm(m) ** 2, (s ** 2).sum(), t.linalg.norm(m @ x - u @ (s * (v @ x))),
                        t.linalg.norm(u.conj().T @ (u @ y) - y), t.linalg.norm(v @ (v.conj().T @ y) - y)])
        h = t.cat([s.real if s.is_complex() else s, vals]).cpu().numpy()
        s_h, (fro2, s2, r, ou, ov) = h[:-5], h[-5:]
        self._s_host = (s, s_h)
        if not np.all(np.isfinite(h)):
            return False
        if fro2 == 0:
            return True
        return (abs(s2 - fro2) <= rtol * fro2 and r <= tol * np.sqrt(fro2) * nx and ou <= tol * ny and
                ov <= tol * ny)

    def host(self, s):
        c = getattr(self, '_s_host', None)
        if c is not None and c[0] is s:
            return c[1]
        return self.to_numpy(s)

    def qr(self, m):
        return self.torch.linalg.qr(m, mode='reduced')

    def eigh(self, h):
        w, v = self.torch.linalg.eigh(h)
        if not eigh_ok(self, h, w, v):
            self.svd_fallbacks += 1
            w, v = _NP.eigh(self.to_numpy(h).copy())
            w, v = self.asarray(w, float), self.asarray(v, complex if np.iscomplexobj(v) else float)
        return w, v

    def norm(self, a):
        return float(self.torch.linalg.norm(a))

    def exp(self, a):
        return self.torch.exp(a)

    def abs2(self, a):
        return a.real ** 2 + a.imag ** 2 if a.is_complex() else a ** 2

    def is_complex(self, a):
        return a.is_complex()

    def real_part(self, a):
        return a.real.contiguous() if a.is_complex() else a

    def to_complex(self, a):
        return a.to(self.torch.complex128)

    def add_at(self, mat, rows, cols, vals):
        mat.index_put_((rows, cols), vals, accumulate=True)

    def take_rows(self, a, idx):
        return a.index_select(0, idx)

    def sync(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize()


_NP = NumpyBackend()
_PROBES = {}


def _probe(be, n, like, key):
    """fixed pseudo-random probe vector of length n (dtype/device of `like`) and its norm"""
    k = (be.name, str(be.device), n, be.is_complex(like), key)
    if k not in _PROBES:
        rng = np.random.default_rng(12345 + n + 7 * key)
        x = rng.normal(size=n) + (1j * rng.normal(size=n) if be.is_complex(like) else 0)
        _PROBES[k] = (be.asarray(x, complex if be.is_complex(like) else float), float(np.linalg.norm(x)))
    return _PROBES[k]


def eigh_ok(be, h, w, v):
    """H V x = V (w x) and V orthonormal on a probe vector"""
    w_h = np.asarray(be.to_numpy(w), dtype=float)
    if not np.all(np.isfinite(w_h)):
        return False
    x, nx = _probe(be, h.shape[0], v, 2)
    nh = be.norm(h)
    if be.norm(h @ (v @ x) - v @ (w * x)) > 1e-9 * max(nh, 1e-300) * nx:
        return False
    return be.norm(v.conj().T @ (v @ x) - x) <= 1e-9 * nx


def get_backend(name='numpy', device=None):
    """name: 'numpy' | 'torch' (CPU or GPU) | 'gpu' (= torch on cuda)"""
    if name in (None, 'numpy', 'np'):
        return NumpyBackend()
    if name == 'torch':
        return TorchBackend(device)
    if name in ('gpu', 'cuda'):
        return TorchBackend(device or 'cuda')
    raise ValueError(f'unknown backend {name!r}')


def set_threads(n):
    """limit BLAS/OpenMP threads; call before numpy does any heavy work (scripts do this before importing numpy)"""
    import os
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = str(n)
    try:
        import torch
        torch.set_num_threads(int(n))
    except Exception:
        pass
