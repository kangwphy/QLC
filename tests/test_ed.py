import numpy as np
import pytest

from conftest import backends
from qlc.backend import get_backend
from qlc.ed import run_ed, ReducedStructure, representations, describe
from qlc.model import Chain, dense_hamiltonian

TS = np.linspace(0, 2, 6)


@pytest.mark.parametrize('L,bc', [(6, 'pbc'), (7, 'pbc'), (9, 'pbc'), (10, 'pbc'), (6, 'obc'), (9, 'obc'),
                                  (10, 'obc')])
def test_reduced_spectrum_matches_dense(L, bc):
    ch = Chain(L, bc)
    rs = ReducedStructure(ch)
    be = get_backend()
    E = []
    for rep in rs.representations(use_orbits=False):
        H = be.to_numpy(rep.dense(rep.op_terms('H'), be))
        assert np.allclose(H, H.conj().T)
        E += list(np.linalg.eigvalsh(H)) * (1 << rs.multiplicity_log2)
    assert len(E) == 1 << L
    assert np.allclose(np.sort(E), np.linalg.eigvalsh(dense_hamiltonian(ch)), atol=1e-10)


@pytest.mark.parametrize('L,bc,probe', [(8, 'pbc', 'total'), (9, 'pbc', 'total'), (8, 'pbc', 'local'),
                                        (9, 'obc', 'total'), (10, 'obc', 'local')])
def test_reduced_equals_full(L, bc, probe):
    kw = dict(betas=[0.0, 0.7], ts=TS, probe=probe, log=None)
    r1 = run_ed(L, bc, method='reduced', **kw)
    r2 = run_ed(L, bc, method='full', **kw)
    assert np.allclose(r1['D'], r2['D'], atol=1e-10)
    assert np.allclose(r1['C0'], r2['C0'], atol=1e-10)
    for a, b in zip(r1['C'], r2['C']):
        assert np.allclose(a, b, atol=1e-10)


def test_ring_local_probe_equals_total():
    r1 = run_ed(12, 'pbc', betas=[0, 1], ts=TS, log=None)
    r2 = run_ed(12, 'pbc', betas=[0, 1], ts=TS, probe='local', log=None)
    for a, b in zip(r1['C'], r2['C']):
        assert np.allclose(a, b, atol=1e-11)


def test_open_chain_drude_vanishes():
    r = run_ed(12, 'obc', betas=[0, 1], log=None)
    assert np.allclose(r['D'], 0, atol=1e-10)


def test_static_values():
    r = run_ed(11, 'pbc', betas=[0.0], ts=TS, log=None)
    assert abs(r['C0'][0] - 20) < 1e-12
    assert np.abs(r['C'][0].imag).max() < 1e-10           # real at beta = 0 only; at beta > 0 C(t)* = C(-t)


@pytest.mark.parametrize('L,bc,probe', [(8, 'pbc', 'total'), (10, 'obc', 'local'), (9, 'obc', 'total')])
def test_independent_dense_reference(L, bc, probe):
    """C(t) = Tr(rho A(t) B)/(norm Tr rho), A(t) = e^{iHt} A e^{-iHt}, against expm, incl. Im C at beta > 0"""
    from scipy.linalg import expm
    from qlc.model import dense_operator
    beta = 0.7
    ch = Chain(L, bc)
    H = dense_hamiltonian(ch)
    A = dense_operator(ch, ch.placements('J'))
    if probe == 'total':
        B, norm = A, L
    else:
        cs = ch.centre_terms(True, 'J')
        B, norm = sum(dense_operator(ch, ch.local_placements('J', c)) for c in cs) / len(cs), 1.0
    rho = expm(-beta * H)
    ref = []
    for t in TS:
        U = expm(1j * H * t)
        ref.append(np.trace(rho @ U @ A @ U.conj().T @ B) / (norm * np.trace(rho)))
    r = run_ed(L, bc, betas=[beta], ts=TS, probe=probe, log=None)
    assert np.allclose(r['C'][0], ref, atol=1e-10)
    assert np.abs(np.imag(ref)).max() > 1e-3                 # the imaginary part is really tested


def test_empty_operator_blocks():
    """on the L = 6 ring Q1 and R vanish in some sector blocks"""
    for op in ('Q1', 'R'):
        a = run_ed(6, 'pbc', op=op, betas=[0, 1], ts=TS, log=None)
        b = run_ed(6, 'pbc', op=op, betas=[0, 1], ts=TS, method='full', log=None)
        assert np.allclose(a['D'], b['D']) and np.allclose(a['C'][1], b['C'][1])


def test_custom_pattern_and_validation():
    from qlc.model import PATTERNS
    a = run_ed(8, 'pbc', op=dict(PATTERNS['J']), betas=[0.5], log=None)
    b = run_ed(8, 'pbc', op='J', betas=[0.5], log=None)
    assert np.allclose(a['D'], b['D'])
    with pytest.raises(ValueError):
        run_ed(8, 'pbc', op={(0, 1): 1.0}, log=None)          # h_0 h_1 is not Hermitian
    with pytest.raises(ValueError):
        run_ed(6, 'obc', probe='local', log=None)            # the probe density does not fit
    with pytest.raises(ValueError):
        run_ed(8, 'pbc', chunk=0, log=None)


@pytest.mark.parametrize('N,D', [(14, 18.7897), (18, 18.7297), (22, 18.7029)])
def test_note_table(N, D):
    """D_N(infinity) of the note, App. E / Sec. 4.4"""
    r = run_ed(N, 'pbc', betas=[0.0], log=None)
    assert abs(r['D'][0].real - D) < 6e-5


def test_sizes_table():
    assert describe(24) == dict(L=24, bc='pbc', k=10, c=4, dim=1024, sectors=16, orbits=6)
    assert describe(30)['dim'] == 8192
    assert describe(12, 'obc')['k'] == 4


def test_sector_cache(tmp_path):
    """the cache is beta-independent: a run at other temperatures reuses every block"""
    kw = dict(ts=TS, outdir=str(tmp_path), log=None)
    r1 = run_ed(10, 'pbc', betas=[0.0, 1.0], **kw)
    msgs = []
    r2 = run_ed(10, 'pbc', betas=[1.0, 2.0], **dict(kw, log=msgs.append))
    assert sum('loaded' in m for m in msgs) == len(representations(Chain(10, 'pbc')))
    assert np.allclose(r1['C'][1], r2['C'][0]) and np.allclose(r1['D'][1], r2['D'][0])
    r3 = run_ed(10, 'pbc', betas=[2.0], ts=TS, log=None)
    assert np.allclose(r3['C'][0], r2['C'][1])


@pytest.mark.parametrize('name', backends())
def test_backends_agree(name):
    be = get_backend(name, 'cpu' if name == 'torch' else None)
    r1 = run_ed(10, 'pbc', betas=[0, 1], ts=TS, log=None, chunk=7)
    r2 = run_ed(10, 'pbc', betas=[0, 1], ts=TS, log=None, be=be, chunk=5)
    assert np.allclose(r1['D'], r2['D'], atol=1e-10)
    for a, b in zip(r1['C'], r2['C']):
        assert np.allclose(a, b, atol=1e-10)
