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
Every sector is saved as an `.npz` (restart).  Thermal weights use a per-sector reference energy (no overflow
at large $\beta$).

## 2. h-word MPO Heisenberg evolution (`qlc.tn.heisenberg`)

The operator is an MPS over the occupation bits of the words (local dimension 2).  A window of $n$ spins has
$m=n-2$ words plus two vacuum bits on each side.  Each Trotter step is the palindromic product of the 5-site
rotation maps of all words (second order), applied with an explicit orthogonality centre and SVD truncation
(`chi`, relative cutoff `tol`).  Time folding evolves only to $\tau=t/2$; translation gives all shifts $x$ from one
MPO; the sum runs over $|x|\le x_0+v t$ ($v=7.3$).

* $\beta=0$: $M=o_0$; real arithmetic (Hermitian operator, real gates), $Z_x=1$.
* $\beta>0$: $\rho^{1/2}=e^{-\beta H/2}$ from imaginary-time sweeps (step `tau_im`, bond `chi_rho`), converted to
  real form; $M=o_0\rho^{1/2}$ (complex).  $\rho^{1/2}$ is evolved with the same gates and $Z_x$ divides out the
  window-edge mismatch of the shifted thermal factor.  **Window size**: at $\beta=1$ the static values need
  $n\ge32$ ($n=24$ is off by 3%, $n=32$ by $<10^{-4}$).
* Routes: `direct` ($o=j$) and `split` ($o=r$; $C_J=C_R-C_R(0)+C_J(0)$, $C_J(0)$ exact at $\beta=0$ and from the
  same thermal factor at $\beta>0$, or given with `--cj0`).

**Validity**: the folded operator spreads by $v\,t/2$ to each side; keep the light cone inside the window,
$n\gtrsim v\,t_{\max}+$ operator span $+$ margin ($n=64$ for $t\le6$ as in the note).  Errors to control:
truncation (`chi`, see `discarded`), Trotter ($O(dt^2)$, Richardson with `scripts/drude.py --richardson`),
window (`edge` = last profile entry), and at $\beta>0$ `tau_im`, `chi_rho`.

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
