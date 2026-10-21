# Physics, conventions and reference values

## Model

Spins $0,\dots,L-1$, local terms $h_j = Z_jZ_{j+1}X_{j+2}$ labelled by their first spin.

* ring (`bc='pbc'`): $j=0,\dots,L-1$, spins mod $L$ ($L\ge5$; the anticommutation graph is the square of the
  cycle for $L\ge 6$).
* open chain (`bc='obc'`): $j=0,\dots,L-3$.

$h_j^2=1$; $h_i,h_j$ anticommute iff $|i-j|\in\{1,2\}$ (mod $L$ on the ring).  Energy current

$$J = i[H,P] = 2i\sum_j (h_jh_{j+1}+2h_jh_{j+2}),\qquad P=\sum_j j\,h_j\ \text{(open chain)},$$

density $j_x = 2Z_xY_{x+2}X_{x+3}+4Z_xZ_{x+1}Y_{x+2}Z_{x+3}X_{x+4}$.  Split:

$$J = 3Q_1+R,\quad Q_1=i\sum_j(h_jh_{j+1}+h_jh_{j+2})\ \text{(conserved on the ring)},\quad R=i\sum_j(h_jh_{j+2}-h_jh_{j+1}).$$

At $\beta=0$: $\langle J,J\rangle=20$, $\langle Q_1,Q_1\rangle=\langle R,R\rangle=2$, $\langle Q_1,R\rangle=0$ (per site,
inner product $\mathrm{tr}(A^\dagger B)/(N2^N)$).  Operators are given as *patterns* (`qlc.model.PATTERNS`):
`{offsets: coefficient}` for $\sum_j\sum_P c_P\,h_{j+P_1}h_{j+P_2}\cdots$ (ordered product); available: `H`, `J`,
`R`, `Q1`, `J3` (current of the third local charge).

Symmetries used: time reversal $K$ (complex conjugation in the $\sigma^z$ basis): $KHK=H$, $KJK=-J$; translation
on the ring.

## Drude weight

$$D(T) = \lim_{\tau\to\infty}\frac1\tau\int_0^\tau dt\,\lim_{N\to\infty}\frac{\langle J(t)J(0)\rangle_{\beta,N}}{N}
\qquad(\text{no factor }\beta).$$

* Finite ring (ED): $D_N(\beta)=\frac{1}{NZ}\sum_{E_m=E_n}e^{-\beta E_m}|J_{mn}|^2$ (infinite-time average).
* Open chain: $J=i[H,P]$ has no matrix elements inside degenerate blocks, so $D_N^{\rm obc}=0$ exactly; bulk
  information is in $C(t)$ before the light cone reaches the ends.
* Infinite chain (MPO / MPS): $C_J(t)=\sum_x\langle j_x(t)j_0\rangle_\beta$ (real part); $D$ is its long-time
  value.  $C_J(t)-D$ oscillates at the band-top frequency $3\sqrt3$ with an amplitude $\sim t^{-3}$, so a mean
  over full periods (`analysis.plateau`) or a tail fit converges fast; the running mean
  $\frac1\tau\int_0^\tau C_J$ carries a $\mathfrak L/(2\tau)$ bias from the diffusive part.

## Reference values

| quantity | value | where |
|---|---|---|
| $M^\star = D_{\rm GHD}(0)$ | $32-243\sqrt3/(10\pi)=18.6027078413$ | note Eq. (40), `theory.M_STAR` |
| $D_{\rm GHD}(\beta)$ | $\int_0^{\pi/3}\frac{d\theta}{\pi}\,\mathrm{sech}^2(\beta\varepsilon)\,(d\varepsilon^2/d\theta)^2$; 9.32235, 2.881327, 0.442764 at $\beta=0.5,1,2$ | Eq. (51), `theory.drude_ghd` |
| $\varepsilon^2(\theta)$ | $(1+2\cos2\theta)^3/(4\cos^2\theta)$ | Eq. (39) |
| $D_N(\infty)$ | 18.7897 (14), 18.7297 (18), 18.7029 (22), 18.6845 (26), 18.6708 (30) | note Sec. 4.4, `tests/test_ed.py` |
| $N(D_N-M^\star)\to K$ | $486\sqrt3/(5\pi)-52=1.58917$ | Eq. (76) |
| Onsager coefficient $\mathfrak L(0)$ | $(74\sqrt3-95\ln(2+\sqrt3))/(4\pi)=0.2435675$; $\mathfrak L(1)=0.090626$ | Eqs. (57), (76), `theory.onsager_ghd` |
| $\langle J,J\rangle_\beta/N$ | 20 ($\beta=0$), 4.290727 ($\beta=1$) | ED ring |
| $\langle R,R\rangle_\beta/N$ | 2 ($\beta=0$), 2.607040 ($\beta=1$) | ED ring |
| finite Mazur family | 18.18747 ($n=4$), 18.59992 ($n=8$) | Sec. 4.4, `theory.mazur_family` |
| max velocity | 7.24 at $\theta\approx0.702$ | Eq. (43) |

## h-word basis

$B_S=i^{a(S)}\prod_{j\in S}h_j$ (ordered), $a(S)$ = number of pairs in $S$ at distance 1 or 2.  $B_S$ is
Hermitian, $B_S^2=1$, $\omega(B_SB_T)=\delta_{ST}$; $KB_SK=(-1)^{a(S)}B_S$.  A Hermitian operator has real
coefficients.  $J$'s density is $2B_{\{0,1\}}+4B_{\{0,2\}}$, $R$'s is $-B_{\{0,1\}}+B_{\{0,2\}}$.  Left
multiplication, Heisenberg rotation $e^{i\delta h_c}\cdot e^{-i\delta h_c}$ and $e^{-\tau h_c}$ change only bit
$c$ and depend on bits $c-2..c+2$ (see `qlc/tn/hword.py`).

Folding (used by the MPO code): with $M(\tau)=o_0(\tau)\rho^{1/2}$, $\tau=t/2$,

$$C_x(t)=\frac{\omega(\rho\,o_x(t)\,o_0)}{\omega(\rho)}=\kappa\,\frac{\overline{\sum_S M_{S-x}M_S(-1)^{a(S)}}}{Z_x},\qquad
Z_x=\sum_S\rho^{1/2}_{S-x}\rho^{1/2}_S,$$

$\kappa=-1$ for $K$-odd operators (currents), $+1$ for $K$-even ones; $C_{-x}=C_x$ by time reversal.
