import math

import numpy as np
import pytest

from qlc import theory, analysis


def test_closed_forms():
    assert abs(theory.M_STAR - 18.6027078413) < 1e-9
    assert abs(theory.drude_ghd(1e-9) - theory.M_STAR) < 1e-8                   # integral = closed form
    assert abs(theory.onsager_ghd(1e-9) - theory.L_ONSAGER_0) < 1e-8
    assert abs(theory.L_ONSAGER_0 - 0.24356750617731) < 1e-12
    assert abs(theory.K_FINITE_SIZE - 1.58916863499818) < 1e-12
    assert abs(theory.K_FINITE_SIZE - (76 - 4 * theory.M_STAR)) < 1e-12
    # <H, H> = int eps^2 dtheta / pi = 1
    from scipy.integrate import quad
    assert abs(quad(lambda th: theory.eps2(th) / math.pi, 0, math.pi / 3)[0] - 1) < 1e-10
    th = np.linspace(0.05, 1.0, 50)
    assert abs(theory.velocity(th).max() - 7.24) < 0.01


@pytest.mark.parametrize('beta,D', [(0.5, 9.32235), (1, 2.881327), (2, 0.442764), (4, 0.0507941), (8, 0.00541503)])
def test_drude_ghd_table(beta, D):
    assert abs(theory.drude_ghd(beta) / D - 1) < 5e-6


@pytest.mark.parametrize('beta,L', [(1, 0.090626), (2, 0.0331)])
def test_onsager_table(beta, L):
    assert abs(theory.onsager_ghd(beta) / L - 1) < 2e-3


@pytest.mark.parametrize('n,M,tol', [(4, 18.18747, 1e-5), (8, 18.59992, 1e-5), (12, 18.6026931, 1e-7),
                                     (16, 18.602707768, 1e-9), (24, 18.6027078406, 1e-10)])
def test_mazur_family(n, M, tol):
    """note, Sec. 4.4 (exact arithmetic: the kernel is too ill-conditioned for float64 beyond ~20 charges)"""
    v = theory.mazur_family(n)
    assert abs(v - M) < tol and v < theory.M_STAR
    with pytest.raises(ValueError):
        theory.mazur_kernel(0.5, -0.3)


def test_mazur_kernel_mode_representation():
    """G(p, q) = int_0^{pi/2} phi_p phi_q dtheta/pi with phi_p = 2p Im 1/(2 cos th e^{3i th} - p)  (Prop. 4.2)"""
    from scipy.integrate import quad
    phi = lambda p, th: 2 * p * np.imag(1 / (2 * np.cos(th) * np.exp(3j * th) - p))
    for p, q in ((-0.3, -0.7), (-1.5, -2.5), (-0.4, -3.0)):
        val = quad(lambda th: phi(p, th) * phi(q, th) / math.pi, 0, math.pi / 2, limit=200)[0]
        assert abs(val - theory.mazur_kernel(p, q)) < 1e-8


def test_estimators_synthetic():
    D, A = 18.6, 0.8
    t = np.arange(0, 8.0001, 0.01)
    C = D + A * np.cos(theory.OMEGA_TAIL * t + 0.3) / np.maximum(t, 0.5) ** 3
    e = analysis.drude_estimates(t, C, t_min=2.0, reference=D)
    assert abs(e['tail_fit']['D'] - D) < 1e-10
    assert np.isfinite(analysis.tail_fit(t, C, t_min=0.0)['D'])       # t = 0 is excluded
    assert abs(e['plateau']['D'] - D) < 2e-3
    assert e['plateau']['periods'] >= 4


def test_richardson_removes_dt2():
    t1 = np.arange(0, 2.0001, 0.1)
    t2 = np.arange(0, 2.0001, 0.05)
    f = lambda t, dt: np.sin(t) + 3 * dt ** 2 * t
    t, C = analysis.richardson(t1, f(t1, 0.1), t2, f(t2, 0.05))
    assert np.allclose(C, np.sin(t), atol=1e-12) and len(t) == len(t1)


def test_size_extrapolation():
    Ns = np.array([12, 15, 18, 21, 24, 27, 30])
    D = theory.M_STAR + 1.6 / Ns + 3.0 / Ns ** 2
    fit = analysis.extrapolate_size(Ns, D, deg=2, by_residue=0, last=5)
    assert abs(fit[0]['D_inf'] - theory.M_STAR) < 1e-9
