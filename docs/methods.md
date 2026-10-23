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

## 4. Static Mazur bound from the charge MPS (`qlc.tn.charges`, `qlc.tn.mazur`)

No time evolution: $D(\beta)\ge M(\beta)=b^\top G^+b$ with $G_{ab}=\langle Q_a,Q_b\rangle_\beta$, $b_a=\langle J,Q_a\rangle_\beta$
(per site), for the even parts $Q^{\lambda,+}_p$, $Q^{\gamma,+}_p$, the local $Q_1$, and the odd parts $Q^{\gamma,-}_p$
that couple to the even ones at $\beta>0$ (all Hermitian and $T$-odd, so $\langle Q\rangle_\beta=0$).

* **Charges as MPS.**  In the basis of ordered products $h_S$ a Type I word is a sequence of blocks separated by single
  empty sites with a product of block weights ($\lambda$: $w(1)=i\sqrt{-1-p}/p$, $w(2)=1/p$; $\gamma$:
  $w(b)=(-1)^bp(1+p)^{b/2-1}$, $b\ge2$; overall factor $i$); a 5-state automaton generates all of them, so a charge is
  an exact MPS of bond 5 (window sum or bulk density with first word at $y_0$; a parity part has two copies, bond 10,
  compressed without loss to 8 for the sum and 7 for the density); even/odd parts are
  $(Q[w]\pm Q[(-1)^bw])/2$.  Checked against word enumeration and the coefficients of the note (App. B).
* **Thermal factor.**  In the $h_S$ basis left multiplication is a real sign, $h_ch_S=(-1)^{n_{c-1}+n_{c-2}}h_{S\oplus c}$,
  so $\rho=e^{-\beta H}$ has real coefficients; it is built by imaginary-time steps (size $\le\tau$, bond $\chi$), and one
  trajectory gives every requested $\beta$.  The second-order sweep biases $M$ upwards by $O(\tau^2)$ ($+0.25\%$ at
  $\beta=1$, $\tau=0.05$), so the default is the 4th-order Suzuki composition $S_2(s\tau)S_2((1-2s)\tau)S_2(s\tau)$,
  $s=1/(2-2^{1/3})$: at $\beta=1$ the error of $M$ is $4\times10^{-5}$ ($\tau=0.05$) and $2\times10^{-6}$ ($\tau=0.025$).  Bond of $e^{-\beta H}$ at cutoff $10^{-10}$: 27, 49, 102, 167, 243 at
  $\beta=0.25,0.5,1,1.5,2$ (so for $\beta\le1$ the bond is set by the cutoff, not by $\chi$).
* **Correlations.**  $G_{ab}=\omega(\rho\,Q_a\,q_{b,y_0})/\omega(\rho)$ with $Q_a$ the window sum and $q_{b,y_0}$ the bulk
  density; $\omega(\rho AB)=\sum_U\rho_U(AB)_U(-1)^{a(U)}$ with $(AB)_U=\sum_{S\oplus T=U}a_Sb_T\,\mathrm{sign}(S,T)$,
  $\mathrm{sign}=(-1)^{\#\{s\in S,t\in T:\,s-t\in\{1,2\}\}}$.  One environment sweep with the last two bits of $T$ and
  of $U$ (16 states) gives the whole matrix $\omega(\rho A_aB_b)$ at once (`omega_matrix`), cost $O(m\chi^2K^2D^2)$.
* **Window.**  The charge words are cut at the window: for the default set $m-y_0\approx200$ words keeps every tail
  below $\sim10^{-6}$ ($\gamma$ at $p=-0.8$ and $\lambda$ at $p=-1.25$ decay slowest); `beta0_check` compares the window
  kernels and the window bound with the closed forms at $\beta=0$.  The left buffer $y_0$ must exceed the thermal
  boundary layer of the open window (default $\max(24,16\beta_{\max})$; at $\beta=1$ the results are the same for
  $y_0=32$, 64, 128).  At $\beta>0$, however, the entries of $G$ that involve the $\gamma$ charges converge slowly in
  the distance $m-y_0$ to the right end of the window, although their word tails are converged at $\beta=0$: at
  $\beta=1$ the $\gamma^+_{-0.2}$ diagonal is off by $-1.15\times10^{-2}$, $-2.6\times10^{-3}$, $-6\times10^{-4}$ at
  $m-y_0=100$, 200, 300 and saturates at $-1.6\times10^{-4}$ from 400 on (against the iMPS of Sec. 5 and the TBA Gram
  matrix).  `G_asym` and `b_twoway` cannot see this (the cut biases $G_{ab}$ and $G_{ba}$ alike).  A too small window
  makes $M$ overshoot.
* **Cutoff.**  $G$ is very ill-conditioned (condition number $\sim10^8$ for 17 charges at $\beta=0$, smallest relative
  eigenvalues $\sim10^{-11}$ at $\beta=2$).  The eigenvalue cutoff is raised to the estimated noise of $G$ (window tails,
  `G_asym`); dropping directions only lowers the bound (projection on a sub-span).  $M$ for a scan of cutoffs is
  stored with every point.
* **Checks** (`tests/test_mazur.py`): automata vs enumeration, Hermiticity, odd-kernel closed form
  $pq\sqrt{(1+p)(1+q)}/\Delta(p,q)$, contractions vs dense matrices of the same imaginary-time circuit ($10^{-12}$),
  $\beta=0$ bound vs the exact rational `theory.mazur_bound`, numpy vs torch.

**Results** (`scripts/run_mazur.py --beta 0 0.25 0.5 1 1.5 2 --chi 256 --tau 0.05 --backend gpu`, default 17 charges:
$Q_1$, $Q^{\lambda,+}_p$ at $p=-1.25,-1.6,-2.2,-3.5,-6,-12$, $Q^{\gamma,\pm}_p$ at $p=-0.2,-0.35,-0.5,-0.65,-0.8$;
$m=232$, $y_0=32$; 4th-order imaginary time; one H100 NVL):

| $\beta$ | 0 | 0.25 | 0.5 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|---|
| $D_{\rm GHD}(\beta)$ | 18.60271 | 15.19322 | 9.32235 | 2.88133 | 1.02324 | 0.44276 |
| $M_{\rm full}/D_{\rm GHD}$ | 1.000000 | 0.999993 | 0.999976 | 0.999992$^\dagger$ | 0.999961 | 0.99991–0.99998 |
| $M_{\rm even}/D_{\rm GHD}$ | 1.000000 | 0.998259 | 0.993970 | 0.979865$^{\dagger\ddagger}$ | 0.958709$^\ddagger$ | 0.935415$^{\dagger\ddagger}$ |

$^\dagger$ $\tau=0.025$.  $^\ddagger$ Superseded: biased upwards by the window error of the $\gamma$ entries (see above);
the same 12 even charges give 0.979694, 0.956706, 0.936007 on the infinite chain (iMPS, Sec. 5; TBA 0.936005 at
$\beta=2$).  Convergence: at $\beta=2$
$\chi=192$ and $256$ agree to $3\times10^{-7}$ (at $\beta\le1$ the bond is below 192 in both runs);
$\tau=0.05\to0.025$ changes $M/D_{\rm GHD}$ by $4\times10^{-5}$ at $\beta=1$ and $4\times10^{-6}$ at $\beta=2$.  With the odd
parts $Q^{\gamma,-}$ the bound reaches $D_{\rm GHD}(\beta)$ to $\le10^{-4}$ for $\beta\le2$ and lies within $4\times10^{-5}$ of
the infinite-chain value of the same family at the same regularization (unit diagonal, $10^{-10}$; $5.6\times10^{-5}$ for
the production values above; at $\beta=0.5$ the deficit is $2.4\times10^{-5}$; at $\beta=1$ it is
$8\times10^{-6}$ only with $\tau=0.025$).  At $\beta=0$, $M=18.60270017$ against $M^\star=18.60270784$ (the family value
with $Q_1$ is $18.60270024$).  The range at $\beta=2$ is the dependence on the eigenvalue cutoff ($2.5\times10^{-6}$ to
$10^{-8}$).

Timings ($m=232$, 17 charges; one imaginary-time sweep / one evaluation of $G$ and $b$):

| $\chi$ | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| H100 NVL (torch) | 12 s / 1.2 s | 31 s / 1.9 s | 99 s / 3.6 s | 337 s / 7.7 s |
| Xeon Gold 6542Y, 16 threads (MKL) | 7.5 s / 144 s | 30 s / 259 s | — | — |

The evaluation of $G$ and $b$ is ~100x faster on the GPU; the imaginary-time sweeps (sequential SVDs) gain only for
$\chi\gtrsim256$.  The full scan above took 56 min on the GPU (almost all in the sweeps; order 4 needs three sweeps per
step) and 1.1 h on 16 CPU cores at $\chi=192$.

## 5. Static Mazur bound on the infinite chain: purification iMPS (`qlc.tn.imps`, `qlc.tn.pauli_mpo`, `qlc.tn.imazur`)

The method of the note's Fig. 2(a) (`lax_drude.tex`, Methods (a)-(d), whose code was not archived), reimplemented.  It
has no window and no truncation of the charge words, and shares with Sec. 4 only the charge rules (with the ordering
sign `_sort_word`), the default family, `mazur_value` with its cutoff list, and the truncation rule `keep_rank`.

* **Thermal state.**  Purification $|\psi\rangle=e^{-\beta H/2}|\Phi\rangle$ ($|\Phi\rangle$ = Bell pairs spin-ancilla, local
  dimension 4) as an infinite MPS with a three-site unit cell.  The terms $h_j$, $j=r$ mod 3, commute and act on disjoint
  triples: layers $L_r(t)$, $S_2(t)=L_0(t/2)L_1(t/2)L_2(t)L_1(t/2)L_0(t/2)$, and the five-stage
  $S_4(t)=S_2(pt)^2S_2((1-4p)t)S_2(pt)^2$, $p=1/(4-4^{1/3})$ (21 layers after merging); the step `--dtau` is in units of
  $\beta/2$.  Each layer is a Hastings-type three-site update of the right-canonical tensors (no inverse Schmidt values);
  after every $S_4$ step (`--canon-every`) the exact canonical form is restored from the dominant fixed points of the cell
  transfer matrix.  Checks: the error of $E/N$ falls by 16 per halving of `dtau` ($5\times10^{-7}$ at `dtau` 0.05, $\beta=0.4$);
  $E/N$ and $\langle H,H\rangle$ agree with the free-fermion values (`theory.thermal_energy`) to $\le3\times10^{-7}$ and
  $4\times10^{-7}$ at $\beta\le2$, $\chi=32$.
* **Charges as Pauli MPOs.**  An ordered product $h_S$ is the Pauli string $\prod_kX^{n_{k-2}}Z^{n_{k-1}\oplus n_k}$ (no
  phase), so the automaton of a charge's word rule is an MPO whose bond state is the automaton state: bond 7 for a parity
  part (IDLE plus three states per sign $\eta$), tries for the local patterns ($Q_2$: 19).  IDLE form: $W_0$ (no word
  completed), $W_1$ (a word completed); $\prod_x(W_0+\mu W_1)=\sum_k\mu^k\{k$ separated words$\}$.
* **Correlations.**  $\langle A,B\rangle_\beta=\frac13\partial_{\mu_a}\partial_{\mu_b}\log\Lambda|_0$, $\Lambda$ the dominant
  eigenvalue of the two-layer cell transfer matrix.  $E_0$ is block triangular in the MPO states; with $r=\mathbb 1$ on
  IDLE$\times$IDLE and the left fixed point $l$, $\partial_a\partial_b\Lambda=\langle l|E_{ab}|r\rangle+\langle l|E_a\mathcal GE_b|r\rangle
  +\langle l|E_b\mathcal GE_a|r\rangle$, $\mathcal G=\mathcal Q(1-E_0)^{-1}\mathcal Q$.  GMRES solves: per operator the
  (IDLE, word) block of $l$ and $\mathcal GE_b|r\rangle$, per pair the (word, word) block of $l$; the derivatives in one
  pass with flagged MPOs $[[W_0,W_1],[0,W_0]]$.  All sums over word lengths and separations are exact, so charges with
  $p$ next to $-1$ (slow tails) are allowed.  One application of the two-layer cell map costs $\propto D_aD_b\chi^3$ (MPO
  bonds $D_a$, $D_b$).  GMRES runs on the host, also with `--backend gpu`, and needs $61(D_a-1)(D_b-1)\chi^2$ complex
  numbers for a pair: about 12 GB for the $Q_2$ pair at $\chi=192$ and 21 GB at $\chi=256$.
* **Mazur value.**  As in the reference: $G$ normalized to unit diagonal, eigenvalues below `--rcond` ($10^{-10}$) times
  the largest dropped; a scan of cutoffs is stored.
* **Checks** (`tests/test_imps.py`): MPOs against dense operators (all patterns and parity parts); $\beta=0$ kernels, odd
  kernel and overlaps for $p=-0.97$, $-1.03$ to $10^{-10}$ and the exact rational family bound; pairs against the dense
  cell transfer matrix (Cauchy-formula derivatives of $\log\Lambda$, $10^{-11}$, including $\langle H\rangle\ne0$);
  adjointness of the maps; the canonical form (isometry, Schmidt values on all bonds); 4th-order convergence; $\beta=1$
  values against the TBA Gram matrix and the bare overlaps ($<10^{-5}$); numpy vs torch.

**Cross-check with the window code (Sec. 4).**  At $\beta=1$ ($\chi=32$) the iMPS reproduces the two-layer TBA Gram
matrix of the 2026-10 package to $3\times10^{-6}$ ($\langle Q_1,Q_1\rangle$, $\langle Q_2,Q_2\rangle$, the
$\gamma^+_{-0.2}$ diagonal: 0.13651110 against 0.13651106), where the window code is off by $-1.6\times10^{-4}$ even
with $m-y_0=600$.

**Results** (`scripts/run_imazur.py`, `--dtau 0.05`, 12 CPU threads unless stated; unit-diagonal $G$, cutoff
$10^{-10}$, all directions kept).  Fig. 2(a) family (`--family note`, 13 even + 6 odd charges) against the published
values (`lax_drude.tex` Table `tab:ft-odd`, step 0.1, 0.05 for $\beta\le0.5$) and the two-layer TBA Gram matrix with bare overlaps:

| $\beta$ | $\chi$, `dtau` | $M_{\rm even}/D_{\rm GHD}$ iMPS / TBA / note | $M_{\rm full}/D_{\rm GHD}$ iMPS / TBA / note |
|---|---|---|---|
| 0.1 | 32, 0.05 | 0.999704 / 0.999704 / 0.999704 | 0.9999985 / 0.9999986 / 0.999999 |
| 0.5 | 32, 0.05 | 0.993990 / 0.993990 / 0.993990 | 0.9999940 / 0.9999946 / 0.999994 |
| 1 | 64, 0.05 | 0.979671 / 0.979672 / 0.979658 | 0.9999808 / 0.9999816 / 0.999968 |
| 2 | 192, 0.025 | 0.934997 / 0.934997 / 0.934992 | 0.9999205 / 0.9999204 / 0.999936 |
| 3 | 192, 0.025 | 0.956261 / 0.956262 / 0.956325 | 0.9998113 / 0.9998110 / 0.999922 |
| 4 | 256, 0.025 | 0.958917 / 0.958919 / 0.960019 | 0.9998388 / 0.9998383 / 1.00099 |
| 6 | 256, 0.025 | 0.950077 / 0.950080 / 0.948068$^*$ | 0.9999002 / 0.9999024 / 1.00034$^*$ |
| 8 | 256, 0.025 | 0.938820 / 0.938835 / 0.937496$^*$ | 0.9999076 / 0.9999290 / 1.00157$^*$ |

$^*$ The note drops $Q_2$ at $\beta=6,8$ (TBA without $Q_2$: 0.948792 / 0.9997447 and 0.937406 / 0.9999274).  The
`dtau 0.025` rows are from runs on one H100 (`--backend gpu`).

At the published settings (`--dtau 0.1`; $\chi=64$ for $\beta\le4$ and 128 at $\beta=6,8$, without $Q_2$ there) the
iMPS reproduces the published values: 0.979658 / 0.9999680 ($\beta=1$) and 0.934991 / 0.9999360 ($\beta=2$) to
$10^{-6}$, and $\beta=3$, 4, 6, 8 to $10^{-5}$–$2\times10^{-4}$ (at $\beta=8$: 0.937496 / 1.00176 against
0.937496 / 1.00157).  The offsets of the published values from the TBA are therefore Trotter errors of `dtau 0.1` and
truncation at $\chi=64$ ($\beta=4$, and $\beta=3$ in the even column) and $\chi=128$ ($\beta=8$); differences of up to
$2\times10^{-4}$ at the published settings remain unexplained, and at $\beta=0.3,0.4$ the published values lie $2$–$7\times10^{-6}$
below both the iMPS and the TBA (cause not identified).  At low temperature use `--dtau 0.025` and a $\chi$ sequence:
`dtau 0.05` leaves Trotter errors of $6\times10^{-6}$ ($\beta=3$) to $4\times10^{-5}$ ($\beta=8$); at `dtau 0.025` the
values converge in $\chi$ at $\beta=4$ and approach the TBA at $\beta=6$ (0.9998681, 0.9998897, 0.9999002 for $\chi=128$,
192, 256 against 0.9999024, still changing by $10^{-5}$), while $\chi=128$ still overshoots at $\beta=8$ (1.00044).  $G$ is ill-conditioned there (smallest relative
eigenvalue $1.7\times10^{-8}$ at $\beta=6$, $4\times10^{-9}$ at $\beta=8$).  $E/N$ and $\langle H,H\rangle$ stay within
$3\times10^{-7}$ and $4\times10^{-7}$ of the exact values in all `--dtau 0.05` runs, so they do not detect these errors
of $M$.  At $\beta=3$ the family is $1.9\times10^{-4}$ below $D_{\rm GHD}$ (iMPS and TBA), not within $10^{-4}$ as stated
in the note.

Default family (`--family qlc`, $\chi=64$) against the TBA and the window code (Sec. 4, unit-diagonal $G$):

| $\beta$ | 0.25 | 0.5 | 1 | 1.5 | 2 |
|---|---|---|---|---|---|
| $M_{\rm full}/D_{\rm GHD}$ iMPS | 0.9999993 | 0.9999984 | 0.9999943 | 0.9999853 | 0.9999672 |
| $M_{\rm full}/D_{\rm GHD}$ TBA | 0.9999994 | 0.9999986 | 0.9999949 | 0.9999854 | 0.9999665 |
| $M_{\rm even}/D_{\rm GHD}$ iMPS | 0.998266 | 0.993992 | 0.979694 | 0.956706 | 0.936007 |
| $M_{\rm even}/D_{\rm GHD}$ TBA | 0.998266 | 0.993993 | 0.979694 | 0.956706 | 0.936005 |
| $M_{\rm even}/D_{\rm GHD}$ window | 0.998259 | 0.993970 | 0.979827 | 0.958936 | 0.941655 |

Entry by entry the iMPS Gram matrix agrees with the TBA to $\le1.5\times10^{-6}$ (relative to
$\sqrt{G_{aa}G_{bb}}$) for $\beta\le2$, where the window code deviates by up to $7.5\times10^{-3}$.

Timings (`scripts/benchmark.py --part imps`, bond saturated at $\chi$; one $S_4$ step = 21 layers + canonical form /
one transfer-matrix pair $\langle Q_a,Q_b\rangle$):

| $\chi$ | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| Xeon Gold 6542Y, 1 thread | 0.15 s / 0.44 s | 0.80 s / 3.0 s | 7.1 s / 14.3 s | — |
| Xeon Gold 6542Y, 12 threads | 0.14 s / 0.49 s | 0.58 s / 1.9 s | 2.5 s / 11.1 s | 14.4 s / 44 s |
| H100 NVL (torch) | 0.42 s / 0.05 s | 1.0 s / 0.11 s | 3.2 s / 0.47 s | 11.1 s / 3.7 s |

A $\beta$ point of the 19-charge family needs 211 pairs: 6–8 min at $\chi=64$ and about 40 min at $\chi=128$ on 12
CPU threads; on one H100 the measurement takes 2–3.5 min at $\chi=128$ and 14–20 min at $\chi=256$.  The script defaults to one thread; `--threads` must not exceed the allocated cores (oversubscribed BLAS threads made
the code about 100 times slower in our runs).

## 6. Drude weight from $C_J(t)$ (`qlc.analysis`)

`plateau` (mean over the longest whole number of tail periods $2\pi/3\sqrt3$ in $[t_{\min},t_{\max}]$),
`tail_fit` ($D+(a\cos\omega t+b\sin\omega t)/t^3$), `running_mean`; `drude_estimates` returns all with their
spread and the deviation from a reference ($M^\star$ or $D_{\rm GHD}(\beta)$).  `richardson` removes the $dt^2$
Trotter error; `extrapolate_size` fits $D_N$ in $1/N$ per residue class mod 3.
