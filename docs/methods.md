# Methods

## 1. Exact diagonalisation (`qlc.ed`)

**Reduced representation.**  The $h_j$ are independent Pauli strings that square to one and pairwise commute or
anticommute.  Their words form a twisted group algebra of $\mathrm{GF}(2)^{M}$ ($M$ = number of terms).  A
symplectic Gram–Schmidt of the commutation form gives $M=2k+c$: $k$ symplectic pairs $(a_i,b_i)$ and a radical
of dimension $c$ (central words).  The map $P(a_i)\to X_i$, $P(b_i)\to Z_i$, $P(z_l)\to s_l=\pm1$ is an algebra
isomorphism onto $k$ logical qubits in each sector $s\in\{\pm1\}^c$, and every sector occurs in the spin space
with the same multiplicity $2^{L-k-c}$, so $\mathrm{Tr}_{\rm spin}A = 2^{L-k-c}\sum_s\mathrm{Tr}_sA$.  Every word
acts as a signed XOR permutation $|n\rangle\to\phi(-1)^{|b\wedge n|}|n\oplus a\rangle$.  On the ring, sectors
related by translation have identical spectra and current matrix elements, so only one sector per orbit is
diagonalised (weight = orbit size); a *local* probe breaks translation and uses all sectors.

| ring $N$ | 18 | 22 | 24 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block dim $2^k$ | 128 | 1024 | 1024 | 4096 | 4096 | 8192 | 16384 | 8192 | 32768 | 32768 | 32768 | 65536 | 65536 |
| sector orbits | 6 | 3 | 6 | 3 | 4 | 3 | 2 | 6 | 2 | 3 | 4 | 3 | 6 |

`describe(L, bc)` prints these numbers.  The open chain has the same structure (no orbits).

**Spectral sums** (`ed/spectrum.py`).  Per block: dense $H$, `eigh`, then the eigenvector matrix is processed in
column chunks: $A V_{:,c}$ is applied as a sum of gathered rows (no dense $J$), $(V^\dagger AV)_{:,c}$ and
$W_{mn}=A_{mn}B_{nm}$ give the degenerate-block sums (Drude weight), the diagonal (static value) and
$\sum_n W_{mn}e^{-iE_nt}$ (correlation).  Memory: one $D\times D$ matrix plus $O(D\times\text{chunk})$ and
$O(D\times n_t)$.  Degenerate blocks: eigenvalue gaps below `tol` $\cdot\max(1,\max|E|)$ (default $10^{-9}$).
Every sector is saved as an `.npz` whose content does not depend on $\beta$ (energies, block sums, diagonal and,
for $C(t)$, $X_m(t)=e^{iE_mt}\sum_nW_{mn}e^{-iE_nt}$, a $D\times n_t$ array); any $\beta$ is evaluated afterwards,
so further temperatures reuse the diagonalisations.  Thermal weights use a common reference energy (no overflow
at large $\beta$).  **Memory** per block: about $4D^2\times16$ bytes on the CPU (dense $H$, eigenvectors, LAPACK
workspace, measured about $3D^2\times16$ plus $\sim5D\times\text{chunk}\times16$ bytes of chunk buffers), i.e.
~1 GiB at $D=4096$, ~4 GiB at 8192, ~16 GiB at 16384, ~64 GiB at 32768 (upper estimates); on a GPU at least $H$, $V$ and the cuSOLVER workspace (an 80 GB card for $D=32768$).  The operator must be
Hermitian (checked).  A local probe on an open chain is centred and must fit completely (checked).

## 2. h-word MPO Heisenberg evolution (`qlc.tn.heisenberg`)

The operator is an MPS over the occupation bits of the words (local dimension 2).  A window of $n$ spins has
$m=n-2$ words plus two vacuum bits on each side.  Each Trotter step is the palindromic product of the 5-site
rotation maps of all words (second order), applied with an explicit orthogonality centre and SVD truncation
(`chi`, relative cutoff `tol`).  Time folding evolves only to $\tau=t/2$; translation gives all shifts $x$ from one
MPO; the sum runs over $|x|\le x_0+v t$ ($v=7.3$).

* $\beta=0$: $M=o_0$; real arithmetic (Hermitian operator, real gates), $Z_x=1$.
* $\beta>0$: $\rho^{1/2}=e^{-\beta H/2}$ from imaginary-time sweeps (bond `chi_rho`; the step is adjusted to
  $\beta/(2n)\le$ `tau_im` so that $\beta$ is reached exactly), converted to real form; $M=o_0\rho^{1/2}$ (complex).
  Only $K$-odd operators (currents) are accepted, since a $K$-even one has $\langle o\rangle_\beta\ne0$.  $\rho^{1/2}$ is evolved with the same gates and $Z_x$ divides out the
  window-edge mismatch of the shifted thermal factor.  **Window size**: at $\beta=1$ the static values need
  $n\ge32$ ($n=24$ is off by 3%, $n=32$ by $<10^{-4}$).
* Routes: `direct` ($o=j$) and `split` ($o=r$; $C_J=C_R-C_R(0)+C_J(0)$, $C_J(0)$ exact at $\beta=0$ and from the
  same thermal factor at $\beta>0$, or given with `--cj0`).

**Validity**: the folded operator spreads by $v\,t/2$ to each side, so the light-cone front reaches the window edge
at $t_{\rm edge}=(n-2-\text{span})/v$.  Window errors of $\sim10^{-4}$ in $C_J$ already appear around
$t_{\rm edge}/2$ to $2t_{\rm edge}/3$ (split, $\chi=64$: $n=32$ is $1.4\times10^{-3}$ off at $t=3$), so watch
`edge_weight`; a warning is printed when it exceeds $10^{-4}$.  The note used $n=96$ for $t\le3.2$ and $n=64$ up to
$t=6$.
Diagnostics per sample: `discarded` (truncation of the evolved operator; `discarded_init` for its construction),
`discarded_rho` / `discarded_rho_init` (thermal factor), `edge_weight` (occupation weight of the outermost words,
i.e. how much of the operator reached the window boundary; at $\beta>0$ relative to that of $\rho^{1/2}$ itself),
`capped` (shift sum limited by the window).
Errors to control: truncation (`chi`, `chi_rho`), Trotter ($O(dt^2)$, Richardson with
`scripts/drude.py --richardson`), window ($n$), and at $\beta>0$ `tau_im`.

**Checks** (`tests/test_hword.py`): every local map and overlap against dense matrices; the full run
(both routes, $\beta=0$ and $0.6$) against a dense emulation of the same Trotter circuit to $10^{-10}$; bulk
references at $\beta=0$ (N = 96, $\chi=512$) and $\beta=1$ (old production run, $n=168$).

## 3. Purification MPS (`qlc.tn.purification`)

Physical ⊗ ancilla per site (dimension 4); $|\psi\rangle=e^{-\beta H_P/2}|\Phi\rangle$ by imaginary-time TEBD;
real time with $\mathcal H=H_P-H_A$ so that $C(t)=\langle A(\tau)|B(\tau)\rangle$, $|B\rangle=e^{-i\mathcal H\tau}o_c|\psi\rangle$,
$|A\rangle=e^{+i\mathcal H\tau}O|\psi\rangle$.  3-sublattice second-order Trotter circuit; $o_c$ averaged over three
neighbouring sublattices (`sym`, removes most of the method error of the 3-periodic circuit; use $dt\le0.025$
at $\beta\sim1$).  No translation or conjugation tricks, so it is an independent check of the h-word code.  On a
finite open chain the direct route sees the boundary-induced decay of the $Q_1$ part (it equals OBC ED with a
local probe), the split route stays close to the bulk.

## 4. Drude weight from $C_J(t)$ (`qlc.analysis`)

`plateau` (mean over the longest whole number of tail periods $2\pi/3\sqrt3$ in $[t_{\min},t_{\max}]$),
`tail_fit` ($D+(a\cos\omega t+b\sin\omega t)/t^3$), `running_mean`; `drude_estimates` returns all with their
spread and the deviation from a reference ($M^\star$ or $D_{\rm GHD}(\beta)$).  `richardson` removes the $dt^2$
Trotter error; `extrapolate_size` fits $D_N$ in $1/N$ per residue class mod 3.
