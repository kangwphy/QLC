import os
import sys

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
sys.path.insert(0, HERE)

import pytest  # noqa: E402

REF_DIR = os.path.join(os.path.dirname(HERE), 'data', 'reference')


def backends():
    out = [pytest.param('numpy', id='numpy')]
    try:
        import torch  # noqa: F401
        out.append(pytest.param('torch', id='torch-cpu'))
    except ImportError:
        out.append(pytest.param('torch', id='torch-cpu', marks=pytest.mark.skip(reason='torch not installed')))
    return out
