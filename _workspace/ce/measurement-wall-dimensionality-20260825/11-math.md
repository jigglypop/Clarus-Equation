# Independent mathematics audit: measurement-wall dimensionality

Status: COMPLETE

Scope: this lane checks R1--R6 in `00-contract.md` from the stated definitions. It does not identify a record label with an extra physical dimension, an objective-collapse mechanism, a gravitational source, or a cosmological component.

## 1. Instrument condition

Let the pointer alternatives be a complete orthogonal projector family $\{\Pi_r\}$, so $\Pi_r\Pi_s=\delta_{rs}\Pi_r$ and $\sum_r\Pi_r=I_A$. Then

$$
\mathcal I_r(\rho)=\operatorname{tr}_A[(I\otimes\Pi_r)U_M(\rho\otimes\sigma_A)U_M^\dagger(I\otimes\Pi_r)]
$$

is completely positive and trace non-increasing, while $\sum_r\mathcal I_r$ is trace preserving. A spectral decomposition of $\sigma_A$ and an apparatus basis give Kraus operators; total completeness follows from unitarity and $\sum_r\Pi_r^2=I_A$.

The same sandwich is not valid for arbitrary POVM effects $E_r$. Although $\sum_rE_r=I$, generally $\sum_rE_r^2\ne I$. The explicit counterexample $E_0=E_1=I/2$ gives $\sum_rE_r^2=I/2$, hence trace loss. A general POVM needs Kraus operators $M_{r\alpha}$ satisfying

$$
\sum_{r,\alpha}M_{r\alpha}^\dagger M_{r\alpha}=I.
$$

This is the required scope correction to the original `effect/projector` wording.

## 2. Four dimensions that must remain distinct

A finite discrete outcome set $\mathcal O$ has covering/topological dimension zero, and a selected singleton $\{r\}$ is a zero-dimensional atom of that outcome space. This establishes only **record dimension**.

It does not establish the dimension of spacetime support:

| operational object | typical spacetime-support dimension |
|---|---:|
| ideal localized interaction event | 0 |
| finite-duration point detector | 1 (worldline segment) |
| finite-duration detector surface | 3 (2-space surface $\times$ time) |
| finite-volume apparatus over a finite interval | 4 |

Thus “every measurement interaction is a spacetime point” has a direct counterexample: any finite-duration point detector has a worldline segment. A point interaction is only a scale limit. Likewise, a discrete record does not define a new physical coordinate. The outcome-space dimension, interaction-support dimension, protocol duration, and the codimension of an operational cut are different notions.

## 3. Hard and partial record walls

For a complete orthogonal projector family $\{P_r\}$, define

$$
\mathcal D_P(\rho)=\sum_rP_r\rho P_r.
$$

Its Kraus operators are $P_r$, so the map is CPTP, and

$$
\mathcal D_P^2(\rho)
=\sum_{r,s}P_rP_s\rho P_sP_r
=\sum_rP_r\rho P_r
=\mathcal D_P(\rho).
$$

For $0\le\eta\le1$,

$$
\Phi_\eta=(1-\eta)\operatorname{Id}+\eta\mathcal D_P
$$

is CPTP with Kraus list $\sqrt{1-\eta}\,I$ and $\sqrt\eta P_r$. In a rank-one pointer basis,

$$
(\Phi_\eta\rho)_{rr}=\rho_{rr},
\qquad
(\Phi_\eta\rho)_{rs}=(1-\eta)\rho_{rs}\quad(r\ne s).
$$

For higher-rank $P_r$, only inter-block coherence is scaled; coherence inside each block remains. Consequently $\eta=1$ is a hard wall only relative to the chosen record partition, while $0<\eta<1$ is a partial wall.

“Every measurement is a hard wall” is false. Weak or unsharp instruments leave coherence, and the one-outcome identity instrument $\mathcal I(\rho)=\rho$ is an immediate no-wall control. Moreover, the nonselective channel $\sum_r\mathcal I_r$ does not select a record. A selected history requires a conditional branch $\mathcal I_r(\rho)/\operatorname{tr}\mathcal I_r(\rho)$ together with an actual record variable.

## 4. Finite-duration wall formation

Set

$$
\Gamma(t)=\int_{t_0}^{t}\gamma(s)\,ds,
\qquad \gamma(t)\ge0.
$$

Because $\mathcal D_P$ is an idempotent superoperator,

$$
e^{\Gamma(\mathcal D_P-I)}
=\mathcal D_P+e^{-\Gamma}(I-\mathcal D_P)
=e^{-\Gamma}I+(1-e^{-\Gamma})\mathcal D_P.
$$

The exact solution of $\dot\rho_t=\gamma(t)(\mathcal D_P-I)\rho_t$ is therefore

$$
\rho_t=\Phi_{\eta(t)}(\rho_{t_0}),
\qquad
\eta(t)=1-e^{-\Gamma(t)}.
$$

At every finite integrated strength $\Gamma<\infty$, one has $0\le\eta<1$; the hard-wall value $\eta=1$ is reached only as $\Gamma\to\infty$. This is an exactly soluble dephasing witness, not a universal model of continuous measurement. General monitoring may carry information, cause jumps, rotate its measured axis, or show non-Markovian recoherence.

## 5. Qubit calculation

For

$$
\rho_0=\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix},
\qquad P_0=|0\rangle\langle0|,
\qquad P_1=|1\rangle\langle1|,
$$

one obtains

$$
\mathcal D_P(\rho_0)=\frac12I,
\qquad
\Phi_\eta(\rho_0)=
\begin{pmatrix}
1/2&(1-\eta)/2\\
(1-\eta)/2&1/2
\end{pmatrix}.
$$

At $\gamma_0=2$, $t-t_0=0.75$, the dimensionless integrated strength is $\Gamma=1.5$, so

$$
\eta=1-e^{-1.5}=0.776869839851570,
\qquad
\rho_{01}=\frac12e^{-1.5}=0.111565080074215.
$$

The executable check meets tolerance $10^{-12}$.

## 6. Zeno limit

For bounded self-adjoint $H$ (or suitable domain hypotheses), an initial state in $\operatorname{ran}P$, and ideal instantaneous selective projections at spacing $T/N$, the product limit is conditionally

$$
\lim_{N\to\infty}\left(Pe^{-iHT/(N\hbar)}P\right)^N
=Pe^{-iPHP T/\hbar}
$$

on the projected subspace, up to convention-dependent outer $P$. This is an ideal no-leakage boundary. It is not an arbitrary finite-strength instrument: finite $\Gamma$ leaves $\eta<1$, and nonselective monitoring is a different channel.

## 7. Time-dependent opportunity accounting

If a time-indexed instrument supplies outcome probabilities $p_r(t)$, an endpoint cost after outcome $o$ is

$$
C_I(o,t_1)=-\sum_{r\ne o}p_r(t_1)\ln p_r(t_1).
$$

Using the unknown future outcome inside the interval would be retrospective. A predictable ensemble-average cost is instead

$$
\overline C_I(t)
=\sum_o p_o(t)C_I(o;t)
=\sum_a p_a(t)[1-p_a(t)][-\ln p_a(t)].
$$

For the monotone Markovian witness, a wall-formation-weighted total can be defined by

$$
C_{\mathrm{wall}}
=\int_{t_0}^{t_1}\dot\eta(t)\,\overline C_I(t)\,dt.
$$

This construction is dimensionless and causal only after the time-indexed instrument has defined $p_a(t)$. It is an information/opportunity functional, not energy. The unweighted integral $\int C_I(t)dt$ has units of time, and $dC_I/dt$ has inverse-time units. Turning any of them into energy or stress requires an independently specified bridge such as $k_BT$, $\hbar/\tau$, or a covariant action density. “Energy without energy” can therefore be retained only as a metaphor for non-energetic bookkeeping.

## Findings

- P0: none for the narrowed operational model.
- P1 resolved by scope: use complete orthogonal pointer projectors in the displayed indirect instrument, or replace it with a genuine Kraus instrument for general POVMs.
- Direct counterexamples remove the universal claims that all measurements are spacetime-0D or hard walls.
- The surviving result is: a completed discrete record is a 0D outcome atom; its physical production is generally a finite-duration, finite-region process; a hard dephasing wall is an ideal limit relative to a chosen record algebra.

## Reproduction

```powershell
& '.codex\hooks\python.cmd' python '_workspace\ce\measurement-wall-dimensionality-20260825\artifacts\verify_measurement_wall.py'
```
