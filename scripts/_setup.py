"""Shared start-up for the scripts: thread limits (before NumPy is imported) and the src/ path.

BLAS threading hurts the tensor-network codes (many small SVDs: one thread per process is several times faster
than ten), so run_mpo.py / run_mps.py default to --threads 1; ED (large dense eigh) uses the machine default.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def setup(default_threads=None):
    n = default_threads
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == '--threads' and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith('--threads='):
            n = a.split('=', 1)[1]
    if n is not None and str(n) != '0':
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                  'NUMEXPR_NUM_THREADS'):
            os.environ[k] = str(n)
    src = os.path.join(ROOT, 'src')
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    return n


def add_backend_args(p, default_threads):
    p.add_argument('--backend', default='numpy', choices=['numpy', 'torch', 'gpu'],
                   help="numpy (CPU) | torch (CPU or GPU, see --device) | gpu (= torch on cuda)")
    p.add_argument('--device', default=None, help="torch device: cpu, cuda, cuda:1, ... (default: cuda if available)")
    p.add_argument('--threads', default=default_threads,
                   help=f'BLAS/OpenMP threads (default {default_threads}; 0 = library default)')


def make_backend(a):
    from qlc.backend import get_backend
    be = get_backend(a.backend, a.device)
    if be.name == 'torch' and a.threads not in (None, '0', 0):
        import torch
        torch.set_num_threads(int(a.threads))
    return be


def log(msg):
    print(msg, flush=True)
