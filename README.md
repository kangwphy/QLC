# QLC — energy transport and Drude weight of the free-fermions-in-disguise chain

Numerics for the periodic FFD chain (Fendley's "free fermions in disguise")

$$H=\sum_j h_j,\qquad h_j = Z_j Z_{j+1} X_{j+2},\qquad J = 2i\sum_j\left(h_j h_{j+1} + 2 h_j h_{j+2}\right),$$

and its Drude weight $D(T)=\lim_{\tau\to\infty}\frac1\tau\int_0^\tau dt\,\lim_{N\to\infty}\langle J(t)J(0)\rangle_\beta/N$.
At infinite temperature the quasi-local charges give the Mazur bound
$D(\infty)\ge M^\star = 32-243\sqrt3/(10\pi)=18.6027$, which the numerics here are consistent with saturating.

Three independent engines, all with NumPy (CPU) or PyTorch (CPU / GPU) back ends:

| engine | what | boundary / size |
|---|---|---|
| `qlc.ed` | exact diagonalisation in the reduced (irreducible-block) representation of the h-algebra: finite-size Drude weight $D_N(\beta)$ and $C_N(t)$ | ring (`pbc`, block $2^k$: $N=24\to1024$, $N=30\to8192$, $N=33\to32768$) and open chain (`obc`) |
| `qlc.tn.heisenberg` | h-word MPO Heisenberg evolution with time folding, **direct route** (evolve $j$) and **split route** (evolve $R$, $J=3Q_1+R$), $\beta=0$ and $\beta>0$ | infinite chain from an open window of $n$ spins ($n=64$ for $t\le6$; $n\ge32$ at $\beta\sim1$) |
| `qlc.tn.purification` | spin-basis purification MPS + TEBD (independent check), same two routes | open chain of $L$ spins |

The Drude weight is measured directly as the long-time value of $C_J(t)$ (`qlc.analysis`, `scripts/drude.py`) and compared with the closed forms in `qlc.theory` ($M^\star$, $D_{\rm GHD}(\beta)$, Mazur kernels, Onsager coefficient).

## Layout

```
src/qlc/
  model.py          chain, h_j, current patterns J R Q1 J3 H, boundary conditions, dense matrices
  backend.py        NumPy / torch (CPU, CUDA, ROCm) array backend
  ed/               algebra.py (reduced representation, sectors, translation orbits), spectrum.py, run.py
  tn/               mps.py (canonical MPS, k-site gates), hword.py (local 5-bit maps), wordmpo.py (word MPO,
                    shifted overlaps), heisenberg.py (direct / split driver), purification.py
  analysis.py       Drude estimators (plateau, tail fit, running mean), Richardson, 1/N extrapolation
  theory.py         closed forms of the note
scripts/            run_ed.py  run_mpo.py  run_mps.py  drude.py  make_jobs.py
slurm/              array_cpu.sbatch  gpu.sbatch  env.sh
tests/              pytest suite (dense same-circuit checks, note values, reference data, backends, scripts)
data/reference/     small reference series: C_J(t) at beta = 0 (N = 96, chi = 512) and beta = 1 (h-word TN)
docs/               physics.md  methods.md  cluster_gpu.md  README_zh.md (中文说明)
```

## Install

```bash
git clone git@github.com:kangwphy/QLC.git && cd QLC
pip install -e .            # numpy, scipy
pip install -e '.[test]'    # + pytest
pip install torch           # optional: GPU back end (CUDA, or the ROCm / DTK build on AMD / Hygon DCU)
```
The scripts also run without installation (they add `src/` to the path).

## Quick start

```bash
# ED: finite rings, Drude weight at several temperatures (N = 14: D_N = 18.78967 at beta = 0)
python scripts/run_ed.py --L 12 14 16 18 20 --bc pbc --beta 0 0.5 1 2
python scripts/run_ed.py --L 24 --beta 0 1 --tmax 20 --dt 0.05             # + C_N(t)
python scripts/run_ed.py --L 20 --bc obc --probe local --beta 0 --tmax 3   # open chain, bulk-like C(t)

# MPO, infinite chain at beta = 0: split and direct routes
python scripts/run_mpo.py --route split  --n 64 --chi 256 --dt 0.05 --tmax 6
python scripts/run_mpo.py --route direct --n 64 --chi 256 --dt 0.05 --tmax 6
# finite temperature
python scripts/run_mpo.py --route split --n 64 --beta 1 --chi 256 --chi-rho 128 --tmax 4

# purification MPS (independent method)
python scripts/run_mps.py --route split --L 64 --beta 1 --chi 256 --chi-psi 192 --dt 0.025 --tmax 4 \
    --psi-file results/mps/psi_b1_L64.pkl

# Drude weight from the outputs (+ Richardson extrapolation of two dt)
python scripts/drude.py results/mpo/*.json results/ed/pbc_L*_J_total_*.json
python scripts/drude.py --richardson results/mpo/run_dt0.05.json results/mpo/run_dt0.025.json
```

TN runs (`run_mpo.py`, `run_mps.py`) write JSON after every sample; their default output names contain the key
parameters plus a short hash of all of them except `tmax`, so job-array scans never collide and resubmitting the
same command (also with a larger `--tmax`) resumes from the checkpoint (`<out>.ckpt.pkl`).  ED output names are
built from the parameters (L, bc, op, probe, t grid, betas); every sector is cached independently of $\beta$
(`results/ed/sectors/`), so further temperatures reuse the diagonalisations (a new t grid recomputes).  Library use:

```python
from qlc.ed import run_ed
from qlc.tn import run_hword, run_purification
from qlc import analysis, theory
r = run_hword(route='split', n=48, chi=128, dt=0.05, tmax=3)
t, C = [s['t'] for s in r['samples']], [s['C_J'] for s in r['samples']]
print(analysis.drude_estimates(t, C, reference=theory.M_STAR))
```

## Direct and split routes

`direct` evolves the current density $j_0$ and gives $C_J(t)$.  `split` uses $J=3Q_1+R$ with
$Q_1=i\sum_j(h_jh_{j+1}+h_jh_{j+2})$ conserved and $\langle Q_1,R\rangle=0$: it evolves only
$r_0=i(h_0h_2-h_0h_1)$ (norm 2 instead of 20) and reconstructs

$$C_J(t)=C_R(t)-C_R(0)+C_J(0)\qquad(\beta=0:\ C_J=18+C_R).$$

At the same bond dimension the split route stays accurate much longer: $n=40$, $\chi=64$, $dt=0.05$, $\beta=0$
gives $2\times10^{-4}$ (split) against $1\times10^{-2}$ (direct) from the $\chi=512$ reference at $t=3$, while for
$t\lesssim2$ both are limited by the same Trotter error.  On an open chain (purification, OBC ED) the split route is
also insensitive to the boundary-induced decay of the $Q_1$ part.  The two routes
differ by an $O(dt^2)$ Trotter term (the circuit conserves $Q_1$ only approximately).

## Tests

```bash
pytest -m "not slow"     # ~1-2 min: dense same-circuit checks of every engine, note values, backends, scripts
pytest -m slow           # bulk reference data at beta = 0 and beta = 1 (a few minutes)
```

## Compute nodes and GPU

See [docs/cluster_gpu.md](docs/cluster_gpu.md).  In short: TN runs are fastest with **one BLAS thread per
process** (the scripts default to `--threads 1`; ten OpenBLAS threads were 9x *slower* in a test) and are best
run as SLURM job arrays (`mkdir -p jobs logs results`, `scripts/make_jobs.py`, `slurm/array_cpu.sbatch`); ED of
large rings and TN runs with $\chi\gtrsim 512$ can use the GPU (`--backend gpu`, `slurm/gpu.sbatch`).  ED needs
about $4D^2\times16$ bytes per block ($D=2^k$: 4 GiB at $N=30$, 64 GiB at $N=31$–33).  GPU runs are validated in
double precision on the CPU code path only; GPU timings have not been measured.

## Provenance

Cleaned-up rewrite of the programs of the 2026-10-20 package (`FFD_periodic_chain_package_20261020`):
`q2_finiteT/finiteT_numerics/{ringred,secorbits,ed_drude}.py` and `ffd_purif/ring_ed/ring_ed_ct.py` → `qlc.ed`;
`remainder/tn/{thermal_word,run_thermal,run_pair}.py` → `qlc.tn.heisenberg`; `ffd_purif/purif_tebd.py` →
`qlc.tn.purification`.  The new code reproduces the old results (tests compare with the note's $D_N$ table and
the old TN reference series).  Physics background: `docs/physics.md` and the working note
*Notes on quasi-local charges and ballistic energy transport in free fermions in disguise at infinite temperature*.
