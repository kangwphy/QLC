# Running on compute nodes; GPU

## Environment

```bash
git clone git@github.com:kangwphy/QLC.git && cd QLC
conda create -n qlc python=3.11 numpy scipy pytest -y && conda activate qlc
pip install -e .
pip install torch            # GPU nodes only (CUDA build, or the ROCm / DTK build on AMD / Hygon DCU)
pytest -m "not slow"         # ~1-2 min
```
Edit `slurm/env.sh` (modules, conda env).  All batch scripts are run from the repository root.

## Threads

* **TN (`run_mpo.py`, `run_mps.py`)**: one BLAS thread per process.  The work is many medium SVDs
  ($2\chi\times16\chi$ for the h-word gates); in tests, ten OpenBLAS threads were 9x *slower* than one
  (37 s vs 4.3 s for the same run).  The scripts default to `--threads 1`; run many independent parameter
  points as a job array instead.
* **ED (`run_ed.py`)**: large dense `eigh` scales well with threads; give big rings a whole node
  (`--cpus-per-task=32 ... --threads 32`) or a GPU.

## Job arrays

```bash
mkdir -p jobs logs results         # logs/ must exist before sbatch
python3 scripts/make_jobs.py run_mpo.py --route direct split --n 64 --chi 256 512 --dt 0.05 0.025 > jobs/mpo.txt
sbatch --array=1-$(wc -l < jobs/mpo.txt) slurm/array_cpu.sbatch jobs/mpo.txt
python3 scripts/make_jobs.py run_ed.py --L 12 14 16 18 20 22 24 26 --bc pbc --beta 0,0.5,1,2 > jobs/ed.txt
sbatch --array=1-$(wc -l < jobs/ed.txt) slurm/array_cpu.sbatch jobs/ed.txt
```
Each line gets its own output file (default names contain a hash of all parameters except `tmax`).  TN tasks
checkpoint every `--ckpt-every` samples and resume when the same line is resubmitted (also with a larger
`--tmax`).  ED tasks cache each sector independently of $\beta$; concurrent tasks use per-process temporary files.
Commas in `make_jobs.py` values make one multi-valued option (only for `run_ed.py --L/--beta`); spaces make a
cartesian product.  Collect with `python3 scripts/drude.py results/...`.

ED memory per block is about $4D^2\times16$ bytes: `--mem=8G` is enough up to $N=28$, $N=29$–30 need ~24G,
$N=31$–33 ~80G (or an 80 GB GPU).

## GPU (`--backend gpu`, `slurm/gpu.sbatch`)

The backend is PyTorch in double precision (`complex128`; Apple `mps` is refused).  The same code path is
tested on the CPU (`pytest` runs the torch backend when torch is installed); GPU timings below are estimates,
not measurements.  Every torch SVD and eigh is validated by cheap $O(mn)$ checks (singular-value sum, probe
reconstruction and orthogonality): torch on macOS (Accelerate) returns wrong SVDs of some rank-deficient matrices
without raising.  On failure the code retries with `driver='gesvd'` (CUDA) and then SciPy on the CPU.

| part | dominant kernel | GPU benefit |
|---|---|---|
| ED, block dim $D=2^k$ | dense Hermitian `eigh` $O(D^3)$, then $V^\dagger AV$ in chunks | large for $D\ge4096$ ($N\ge25$): one $D=4096$ block took ~75 s on 4 CPU threads here; $D=8192$ takes minutes on a multi-core CPU and is expected to take seconds to tens of seconds on an A100/H100.  Memory: $H$, $V$ and the solver workspace, at least $3D^2\times16$ B on a GPU, so $N=31$–33 ($D=32768$) needs an 80 GB card or a big-memory CPU node. |
| h-word MPO | 5-site gate: 4 SVDs of $(2\chi)\times(16\chi)\dots$; shifted overlaps $O(n\chi^3)$ per shift | measured on one Apple-M4 core ($n=48$, $\beta=0$): one Trotter step 1.1 s ($\chi=64$), 5 s (128), 40 s (256), cost $\propto\chi^3$; worthwhile on a GPU for $\chi\gtrsim512$; for $\chi\le256$ one CPU core per run in a job array is usually the better use of resources (the gate loop is sequential and launch-bound on a GPU). |
| purification | 3-site gates (local dim 4): SVDs of $(4\chi)\times(16\chi)$ | worthwhile for $\chi\gtrsim256$. |

Possible further GPU work (not done): batch the commuting gates of one sublattice layer into batched SVDs;
mixed-precision (complex64) gate application with float64 accumulation; real symmetric ED blocks.

Examples:

```bash
sbatch slurm/gpu.sbatch python3 scripts/run_ed.py --L 30 --beta 0 0.5 1 2 --backend gpu
sbatch slurm/gpu.sbatch python3 scripts/run_mpo.py --route split --n 96 --chi 768 --tmax 6 --backend gpu
```

On NVIDIA nodes torch uses CUDA; on AMD / Hygon DCU nodes the ROCm (DTK) build of torch also exposes the device as
`cuda`, so the same flags work.  `python3 -c "import torch; print(torch.cuda.is_available())"` must print `True`.

## Typical sizes (from the note)

* $\beta=0$: $n=64$–96, $\chi=512$, $dt=0.05$ and $0.025$ (Richardson), $t\le6$; split route.
* $\beta=1$: $n=64$, $\chi=256$, $\chi_\rho=128$, $\tau_{\rm im}=0.0125$, $t\le4$–6.
* Purification: $L=64$, $\chi=256$, $\chi_\psi=192$, $dt=0.025$, $d\tau=0.0125$; cache the thermal state.
* ED: rings $N=12$–30 (all residues mod 3) for $D_N(\beta)$ and the $1/N$ extrapolation.
