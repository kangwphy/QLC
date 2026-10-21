"""qlc: energy transport and Drude weight of the free-fermions-in-disguise chain H = sum_j Z_j Z_{j+1} X_{j+2}.

    qlc.model      chain, local terms, current patterns J, R, Q1, J3, boundary conditions
    qlc.ed         exact diagonalisation (reduced representation; ring and open chain)
    qlc.tn         h-word MPO Heisenberg evolution (direct / split routes) and spin-basis purification MPS
    qlc.analysis   Drude-weight estimators from C(t), Richardson and finite-size extrapolation
    qlc.theory     closed forms: Mazur bound, D_GHD(beta), Onsager coefficient, mode functions
    qlc.backend    NumPy / torch (CPU or GPU) array backend
"""
__version__ = '0.1.0'

from .model import Chain, PATTERNS, pattern
from .backend import get_backend
