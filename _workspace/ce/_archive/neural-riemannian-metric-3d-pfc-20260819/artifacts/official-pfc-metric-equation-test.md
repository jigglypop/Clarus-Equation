# Official PFC metric-equation battery

Status: `REAL_DATA_EQUATION_TEST`

- Official author repository: `https://github.com/m-j-wojcik/pfc_learning.git`
- Frozen author commit: `48ada8054940f6a7ac26e8e83d150357a9f249d2`
- Official dataset DOI: https://doi.org/10.5061/dryad.c2fqz61kb
- Paper DOI: https://doi.org/10.1038/s41593-026-02333-w

## Equation and null

For each first/last learning stage, the official per-neuron selectivity vectors are treated as points in the released three-coordinate selectivity chart:

$$
C_k=\operatorname{Cov}(s_n\mid k),\qquad g_k=C_k^{-1}.
$$

The empirical null is exchangeability of first/last-stage selectivity rows while preserving the two observed sample sizes. The primary statistic is the affine-invariant distance

$$
D_{\mathrm{AI}}(g_1,g_L)=D_{\mathrm{AI}}(C_1,C_L)
=\left\|\log\left(C_1^{-1/2}C_LC_1^{-1/2}\right)\right\|_F.
$$

The same first/last covariances are also evaluated with the generalized log modes, exact scale/anisotropy decomposition, symmetric Gaussian KL (Jeffreys divergence), log-Euclidean distance, Bures/Wasserstein-2 distance, the SPD geodesic midpoint, and the covariant metric transformation law. AIRM and Jeffreys are GL(3)-congruence invariant. Log-Euclidean and Bures are fixed-chart sensitivity analyses.

Monte Carlo permutations: `20000` per row; seed base: `20260819`; p-values use the +1 correction.

## Results

| Dataset | N first/last | AIRM total | AIRM shape | Shape share | p(total) | p(shape) | Primary decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Exp1 main | 114/91 | 1.136529 | 1.083032 | 90.8% | 0.095745 | 0.052797 | DO NOT REJECT H0 |
| Exp1 fixation-bias control | 101/84 | 1.035930 | 1.004306 | 94.0% | 0.148993 | 0.083546 | DO NOT REJECT H0 |
| Exp2 (3-stage binning) | 140/103 | 1.758404 | 1.402773 | 63.6% | 0.000050 | 0.000150 | REJECT H0 |
| Exp2 (4-stage binning) | 110/87 | 1.966164 | 1.642727 | 69.8% | 0.000050 | 0.000050 | REJECT H0 |
| Exp2 (5-stage binning) | 110/98 | 1.912979 | 1.551033 | 65.7% | 0.000050 | 0.000150 | REJECT H0 |
| Exp2 (6-stage binning) | 110/98 | 1.912979 | 1.551033 | 65.7% | 0.000050 | 0.000050 | REJECT H0 |

The primary Exp1 initial-learning comparison does not reject equal first/last selectivity geometry. Its fixation-bias control agrees with that null result. The primary four-stage Exp2 rule-generalization comparison rejects exchangeability, and the same conclusion is stable under the official 3, 5, and 6-stage binnings.

## Full equation battery on the primary comparisons

The secondary p-values below reuse the same row-exchangeability sensitivity null. Jeffreys is a different weighting of the same generalized eigenvalues as AIRM, so it is not independent evidence. Log-Euclidean and Bures depend on the released coordinate scaling.

| Statistic | Transformation status | Exp1 value | Exp1 p | Exp2 value | Exp2 p |
|---|---|---:|---:|---:|---:|
| AIRM | GL(3) invariant | 1.136529 | 0.095745 | 1.966164 | 0.000050 |
| AIRM anisotropy | GL(3) invariant decomposition | 1.083032 | 0.052797 | 1.642727 | 0.000050 |
| Symmetric Gaussian KL | GL(3) invariant | 0.678011 | 0.101545 | 2.467075 | 0.000050 |
| Log-Euclidean | fixed-chart/O(3) | 1.124505 | 0.099695 | 1.959513 | 0.000050 |
| Bures/W2 | fixed Euclidean ground cost | 0.031780 | 0.136493 | 0.060965 | 0.000100 |

Generalized log-deformation modes `log(lambda_i)`:

- Exp1: `[-0.871004, -0.360645, 0.634811]`
- Exp2: `[-1.835143, -0.516774, 0.48062]`
- Signed log-volume changes: Exp1 `-0.298419`, Exp2 `-0.935648`

## Frozen relative-deformation equation

The coordinate-free relative object is the positive, $g_1$-self-adjoint endomorphism

$$
A_k=g_1^{-1}g_k,\qquad L_k=\log A_k,\qquad
A_k'=P A_k P^{-1}.
$$

For a declared task contrast $\delta$ that transforms with the chart, the directional stretch and the discovered calibration equation are

$$
R_k(\delta)=\frac{\delta^Tg_k\delta}{\delta^Tg_1\delta},\qquad
\rho_k(\delta)=\sqrt{R_k(\delta)},\qquad
\widehat A_{k,j}=\operatorname{logistic}(a_j+b\rho_k(e_j)).
$$

The released factors fix $e_j$ to the named factor chart. If the chart is recoded by $s'=Ps$, the same contrast must transform as $\delta'=P\delta$; then $R_k$ is invariant. Numerically one may use the symmetric whitened representation

$$
M_k=g_1^{-1/2}g_kg_1^{-1/2},\qquad
u_j=\frac{g_1^{1/2}e_j}{\sqrt{e_j^Tg_1e_j}},\qquad
R_k(e_j)=u_j^TM_ku_j.
$$

Using $e_j^TM_ke_j$ without the whitened normalized $u_j$ is not the same equation. The generalized eigenvalues of $A_k$ are chart-invariant; raw coordinates of $M_k$ and fixed numerical axes are not.

A canonical deformation transporter is

$$
T_k=A_k^{1/2}=\exp(\tfrac12 L_k),\qquad T_k^Tg_1T_k=g_k.
$$

It is the unique positive $g_1$-self-adjoint choice. A generic congruence factor is not unique, so no raw matrix $T$ is identified as a biological mechanism by this dataset.

### Exp2 relative-precision stability

- Relative precision stretches: `[0.6184, 1.67661, 6.266031]`
- Log stretches: `[-0.48062, 0.516774, 1.835143]`
- Dominant covariance-axis angle: `75.632250` degrees
- Dominant eigengaps, stage 1/stage 4: `0.722476` / `0.564303`
- Released-row bootstrap 95% angle interval: `[51.794, 89.1112]` degrees
- Released-row bootstrap 95% AIRM interval: `[1.3628, 2.7516]`
- Released-row bootstrap 95% log-stretch intervals: `[[-1.0976, 0.0745], [-0.015, 1.0287], [1.1772, 2.5848]]`
- Fixed-size released-row repartition p(angle): `0.001200`

These intervals and p-value condition on released pseudopopulation rows. They do not provide animal/session inference. The dominant axis is reasonably separated; the lower modes are less stable, so individual lower-axis rotations are not promoted.

## Tensor-law and SPD-geodesic checks on the official matrices

| Check | Exp1 residual/change | Exp2 residual/change |
|---|---:|---:|
| g'=P^-T g P^-1 residual | 1.290e-16 | 3.003e-16 |
| A'=P A P^-1 residual | 3.692e-16 | 2.173e-16 |
| T^T g1 T=gk residual | 1.861e-15 | 2.736e-15 |
| T is g1-self-adjoint residual | 1.106e-16 | 2.297e-16 |
| T'=P T P^-1 residual | 1.354e-15 | 7.320e-16 |
| directional stretch GL-invariance residual | 0.000e+00 | 8.882e-16 |
| R=u^T M u residual | 0.000e+00 | 3.553e-15 |
| relative precision log-spectrum residual | 1.665e-16 | 1.221e-15 |
| AIRM GL-invariance residual | 8.882e-16 | 6.661e-16 |
| Jeffreys GL-invariance residual | 0.000e+00 | 0.000e+00 |
| SPD midpoint half-distance residual | 6.661e-16 | 8.882e-16 |
| Log-Euclidean change under non-orthogonal P (expected) | 4.674e-03 | 1.557e-02 |
| Bures change under non-orthogonal P (expected) | 2.560e-03 | 1.528e-03 |

## Derived Fisher-pullback equation and cross-experiment test

Treat the released neuron-by-factor selectivity matrix `S` as the Jacobian of the mean population response in a linear Gaussian encoding model. Under the explicitly approximate homoscedastic residual model, the total population Fisher information and its one-parameter decoder calibration are

$$
J_F=S^TQ^{-1}S,\qquad Q=\sigma^2I,
\qquad \Phi^{-1}(A_{k,j})=\kappa\sqrt{e_j^TJ_{F,k}e_j}.
$$

The common unknown noise scale and fixed stimulus-contrast factor are absorbed into the single coefficient `kappa`. It was fitted once on all 12 Exp1 stage-axis values and frozen before predicting the 12 Exp2 stage-axis decoder accuracies. This is a homoscedastic Gaussian/Bayes calibration applied to the released cross-validated SVM readout, not an identity for SVM accuracy.

- Fitted `kappa`: `0.642870`
- Fisher tensor coordinate-law residual: `3.116e-16`
- No prediction p-value is assigned: ordered learning stages and distinct task axes are not exchangeable biological units.

| Frozen Exp1 model -> Exp2 | RMSE | MAE | Pearson r |
|---|---:|---:|---:|
| Fisher pullback (one parameter) | 0.048731 | 0.045321 | 0.965308 |
| inverse-covariance accessibility (one parameter) | 0.045172 | 0.040784 | 0.982969 |
| isotropic Fisher (one parameter) | 0.064648 | 0.059908 | 0.621773 |
| Exp1 global-mean baseline | 0.089479 | 0.081682 | nan |
| Exp1 stage-mean baseline | 0.087478 | 0.075678 | 0.381434 |

Exp2 observed vs Fisher-predicted within-decoder accuracy `[set, XOR2, context]`:

| Stage | Observed | Predicted |
|---:|---|---|
| 1 | `[0.708418, 0.605187, 0.59915]` | `[0.729044, 0.648969, 0.631112]` |
| 2 | `[0.57619, 0.523895, 0.569303]` | `[0.616508, 0.588782, 0.609716]` |
| 3 | `[0.543878, 0.5125, 0.659609]` | `[0.598918, 0.584567, 0.65243]` |
| 4 | `[0.533418, 0.530612, 0.581463]` | `[0.592382, 0.587579, 0.633107]` |

The Fisher calibration is compared directly with global/stage-mean, isotropic, and inverse-covariance controls below. Because residual Q is unavailable and the released rows are pseudopopulations, this comparison is descriptive cross-experiment transfer rather than a population-level significance test.

Exp1 first-to-last log changes in the coordinate-axis metric costs `(colour, shape, XOR)` are reported below. Positive means that coordinate became more expensive under the inverse-covariance candidate; negative means cheaper.

`[0.615599, 0.469348, -0.565714]`

## Finite equation-family tournament

The released data identify only stagewise three-factor selectivity summaries, so a finite operational universe is frozen to matrix spectral functions, Fisher/population norms, scale-shape summaries, regularized precision, stage geometry, neuron-level projected-drive gates, and standard accuracy links. Every coefficient and the candidate choice use Exp1 only. Exp2 is evaluated without coefficient refitting at the official 3, 4, 5, and 6-stage binnings.

For each stage,

$$
H_k=N_k^{-1}S_k^TS_k,\qquad J_k=N_kH_k,\qquad X_k=H_k/\tau_{\mathrm{Exp1}},
$$

and the common calibration form is

$$
\widehat z_{kj}=a_j+\sum_m b_m x_{m,kj},\qquad
\widehat A_{kj}=\ell^{-1}(\widehat z_{kj}).
$$

- Enumerated producer-link candidates: `636`
- Distinct producers: `128`
- Links: `identity, probit, logit, cloglog, arcsine`
- Exp1-only scale `tau`: `0.0035268913`
- Selection: leave-one-Exp1-stage-out accuracy RMSE; ties prefer fewer parameters.

### Exp1-selected equation

- Producer: `spectral ratio saturation alpha=0.1`
- Formula: `$a_j+b sqrt(e_j^T X(X+0.1I)^-1 e_j)$`
- Link: `arcsine`
- Parameters: `4`
- Exp1 leave-one-stage-out RMSE: `0.044492`
- Mean Exp2 robustness-binning RMSE: `0.047437`

| Exp2 binning | RMSE | MAE | Pearson r | Spearman rho |
|---:|---:|---:|---:|---:|
| 3 | 0.045649 | 0.042100 | 0.841431 | 0.666667 |
| 4 | 0.048259 | 0.043410 | 0.816197 | 0.748252 |
| 5 | 0.047183 | 0.040025 | 0.846420 | 0.778571 |
| 6 | 0.048658 | 0.043033 | 0.839989 | 0.766133 |

### Projected-drive threshold test

The verbal `strength x alignment` factors are not separately identifiable under this projected-drive definition. For neuron row $v_{nk}$ and named factor projector $P_j=e_je_j^T$ they collapse exactly to the dimensionless projected drive

$$
s_{nk}=\frac{\lVert v_{nk}\rVert}{\sqrt{\tau}},\qquad
a_{nkj}=\frac{\lVert P_jv_{nk}\rVert}{\lVert v_{nk}\rVert},\qquad
d_{nkj}=s_{nk}a_{nkj}=\frac{|S_{k,nj}|}{\sqrt{\tau}}.
$$

The primary gate is applied before population aggregation:

$$
\gamma_{nkj}=\sigma[\beta(d_{nkj}-\theta)],\qquad
R_{kj}=\left[N_k^{-1}\sum_n(d_{nkj}\gamma_{nkj})^2\right]^{1/2},
$$

$$
h(\widehat A_{kj})=a_j+bR_{kj}.
$$

Here $\tau$ is recomputed from Exp1 training stages in every fold. The finite grid is $\beta\in\{1,2,4,8\}$ and $\theta\in\{0.25,0.5,0.75,1,1.5\}$. The no-gate row is exactly `SPD power p=1`; threshold-only removes amplitude, while the additive control gives projected RMS and mean gate activation separate coefficients.

| Model | Exp1-selected specification | Link | Params | Exp1 CV RMSE | Mean Exp2 RMSE |
|---|---|---|---:|---:|---:|
| No gate | SPD power p=1 | logit | 4 | 0.049888 | 0.059045 |
| Projected-drive gate | neuron projected-drive gate beta=8 theta=0.75 | logit | 4 | 0.048781 | 0.057462 |
| Threshold only | threshold-only gate beta=8 theta=1.5 | probit | 4 | 0.050231 | 0.057914 |
| Additive drive + threshold | projected drive plus threshold beta=8 theta=1 | probit | 5 | 0.050281 | 0.059503 |

Relative to the no-gate projected drive, the Exp1-selected primary gate changes RMSE by `+0.001107` on Exp1 CV and `+0.001583` on the frozen Exp2 readout; positive values favor the gate.

This is a post-discussion discovery test on released pseudopopulation rows. It tests a selectivity-to-decoder calibration surrogate, not a synaptic threshold, effective connectivity, or causal routing mechanism.

### Every producer, best Exp1-selected link

This table retains every nonduplicated producer family. The best link for each row is chosen only by Exp1 leave-one-stage-out error; the Exp2 columns are readouts, not selection inputs.

| Producer | Best link | Params | Exp1 CV RMSE | Mean Exp2 RMSE | K=3 | K=4 | K=5 | K=6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| spectral ratio saturation alpha=0.1 | arcsine | 4 | 0.044492 | 0.047437 | 0.045649 | 0.048259 | 0.047183 | 0.048658 |
| spectral ratio saturation alpha=0.3 | logit | 4 | 0.044994 | 0.047016 | 0.044868 | 0.047891 | 0.047069 | 0.048235 |
| Fisher total | probit | 4 | 0.045272 | 0.064086 | 0.072276 | 0.060963 | 0.061851 | 0.061254 |
| SPD power p=-2 | arcsine | 4 | 0.045299 | 0.048933 | 0.047965 | 0.050819 | 0.047123 | 0.049824 |
| relative precision stretch | logit | 4 | 0.045437 | 0.030212 | 0.026504 | 0.031867 | 0.032032 | 0.030443 |
| SPD power p=-1 | logit | 4 | 0.045601 | 0.045933 | 0.044536 | 0.046969 | 0.045298 | 0.046929 |
| regularized precision lambda=0.1 p=2 | probit | 4 | 0.045791 | 0.045693 | 0.044604 | 0.047159 | 0.044466 | 0.046542 |
| spectral ratio saturation alpha=1 | arcsine | 4 | 0.046116 | 0.048678 | 0.046520 | 0.049274 | 0.049248 | 0.049670 |
| SPD power p=-0.5 | logit | 4 | 0.046224 | 0.047153 | 0.045612 | 0.047968 | 0.046969 | 0.048061 |
| precision resolvent lambda=0.1 | logit | 4 | 0.046253 | 0.046053 | 0.044546 | 0.046974 | 0.045713 | 0.046977 |
| regularized precision lambda=0.1 p=1 | logit | 4 | 0.046253 | 0.046053 | 0.044546 | 0.046974 | 0.045713 | 0.046977 |
| spectral exponential saturation beta=1 | identity | 4 | 0.046334 | 0.048135 | 0.045802 | 0.048422 | 0.049057 | 0.049260 |
| regularized precision lambda=0.1 p=0.5 | logit | 4 | 0.046884 | 0.048245 | 0.046696 | 0.049058 | 0.048159 | 0.049067 |
| matrix log | logit | 4 | 0.047168 | 0.050174 | 0.048628 | 0.050969 | 0.050199 | 0.050899 |
| precision resolvent lambda=0.3 | logit | 4 | 0.047202 | 0.047852 | 0.046314 | 0.048710 | 0.047711 | 0.048675 |
| spectral ratio saturation alpha=3 | probit | 4 | 0.047590 | 0.053245 | 0.051503 | 0.054009 | 0.053561 | 0.053906 |
| shifted matrix log epsilon=0.1 | logit | 4 | 0.047775 | 0.051501 | 0.050026 | 0.052337 | 0.051499 | 0.052140 |
| regularized precision lambda=1 p=2 | arcsine | 4 | 0.048163 | 0.049117 | 0.047661 | 0.049837 | 0.049110 | 0.049858 |
| spectral exponential saturation beta=0.3 | logit | 4 | 0.048295 | 0.055249 | 0.053718 | 0.056110 | 0.055390 | 0.055777 |
| SPD power p=0.5 | logit | 4 | 0.048402 | 0.054336 | 0.052905 | 0.055263 | 0.054366 | 0.054810 |
| shifted matrix log epsilon=0.3 | logit | 4 | 0.048637 | 0.053618 | 0.052292 | 0.054517 | 0.053531 | 0.054131 |
| neuron projected-drive gate beta=8 theta=0.75 | logit | 4 | 0.048781 | 0.057462 | 0.055626 | 0.058598 | 0.057759 | 0.057863 |
| spectral ratio saturation alpha=10 | logit | 4 | 0.048907 | 0.056763 | 0.055354 | 0.057743 | 0.056807 | 0.057146 |
| Fisher strict | probit | 1 | 0.048947 | 0.050711 | 0.055554 | 0.048731 | 0.049054 | 0.049504 |
| regularized precision lambda=1 p=1 | logit | 4 | 0.049008 | 0.053308 | 0.052069 | 0.054202 | 0.053109 | 0.053853 |
| precision resolvent lambda=1 | logit | 4 | 0.049008 | 0.053308 | 0.052069 | 0.054202 | 0.053109 | 0.053853 |
| neuron projected-drive gate beta=8 theta=0.5 | logit | 4 | 0.049025 | 0.058020 | 0.056532 | 0.059112 | 0.058095 | 0.058339 |
| neuron projected-drive gate beta=8 theta=1 | logit | 4 | 0.049070 | 0.056203 | 0.054277 | 0.057378 | 0.056588 | 0.056569 |
| neuron projected-drive gate beta=4 theta=0.75 | logit | 4 | 0.049186 | 0.056964 | 0.055223 | 0.058123 | 0.057239 | 0.057272 |
| neuron projected-drive gate beta=4 theta=0.5 | logit | 4 | 0.049189 | 0.057696 | 0.056115 | 0.058825 | 0.057844 | 0.058001 |
| spectral exponential saturation beta=0.1 | logit | 4 | 0.049308 | 0.057747 | 0.056414 | 0.058795 | 0.057732 | 0.058048 |
| Fisher-occupancy geometric mix alpha=0.75 | logit | 4 | 0.049405 | 0.085069 | 0.082537 | 0.089221 | 0.081492 | 0.087024 |
| neuron projected-drive gate beta=4 theta=0.25 | logit | 4 | 0.049417 | 0.058273 | 0.056849 | 0.059402 | 0.058303 | 0.058537 |
| shape only | logit | 4 | 0.049509 | 0.086268 | 0.083135 | 0.090479 | 0.083152 | 0.088304 |
| neuron projected-drive gate beta=4 theta=1 | probit | 4 | 0.049517 | 0.056335 | 0.054526 | 0.057535 | 0.056770 | 0.056509 |
| regularized precision lambda=1 p=0.5 | logit | 4 | 0.049552 | 0.055455 | 0.054334 | 0.056412 | 0.055206 | 0.055870 |
| neuron projected-drive gate beta=8 theta=0.25 | logit | 4 | 0.049590 | 0.058638 | 0.057326 | 0.059771 | 0.058583 | 0.058872 |
| relative information stretch | logit | 4 | 0.049696 | 0.028400 | 0.026801 | 0.030615 | 0.028663 | 0.027520 |
| Fisher-occupancy geometric mix alpha=0.25 | logit | 4 | 0.049781 | 0.087570 | 0.083955 | 0.091831 | 0.084868 | 0.089626 |
| neuron projected-drive gate beta=2 theta=0.25 | logit | 4 | 0.049873 | 0.057824 | 0.056369 | 0.059032 | 0.057874 | 0.058023 |
| SPD power p=1 | logit | 4 | 0.049888 | 0.059045 | 0.057788 | 0.060207 | 0.058949 | 0.059236 |
| neuron projected-drive gate beta=2 theta=0.5 | logit | 4 | 0.050080 | 0.057431 | 0.055932 | 0.058682 | 0.057510 | 0.057600 |
| population L1 | logit | 4 | 0.050172 | 0.071426 | 0.092639 | 0.061495 | 0.066418 | 0.065151 |
| shifted matrix log epsilon=1 | logit | 4 | 0.050178 | 0.057678 | 0.056666 | 0.058731 | 0.057360 | 0.057954 |
| threshold-only gate beta=8 theta=1.5 | probit | 4 | 0.050231 | 0.057914 | 0.053970 | 0.058421 | 0.059969 | 0.059296 |
| projected drive plus threshold beta=8 theta=1 | probit | 5 | 0.050281 | 0.059503 | 0.057751 | 0.060136 | 0.059977 | 0.060147 |
| neuron projected-drive gate beta=2 theta=0.75 | arcsine | 4 | 0.050472 | 0.057603 | 0.056031 | 0.058946 | 0.057840 | 0.057596 |
| neuron projected-drive gate beta=8 theta=1.5 | identity | 4 | 0.050663 | 0.056694 | 0.054937 | 0.058045 | 0.057670 | 0.056124 |
| projected drive plus threshold beta=4 theta=1 | identity | 5 | 0.050810 | 0.060309 | 0.058495 | 0.061027 | 0.060977 | 0.060738 |
| precision resolvent lambda=3 | logit | 4 | 0.050881 | 0.059101 | 0.058251 | 0.060221 | 0.058625 | 0.059308 |
| spectral exponential saturation beta=3 | identity | 4 | 0.051010 | 0.046979 | 0.045302 | 0.048146 | 0.046680 | 0.047789 |
| precision accessibility | logit | 4 | 0.051021 | 0.052359 | 0.052438 | 0.053408 | 0.051096 | 0.052494 |
| neuron projected-drive gate beta=2 theta=1 | identity | 4 | 0.051072 | 0.058296 | 0.056664 | 0.059840 | 0.058645 | 0.058034 |
| neuron projected-drive gate beta=1 theta=0.25 | identity | 4 | 0.051268 | 0.059455 | 0.058191 | 0.061066 | 0.059421 | 0.059143 |
| neuron projected-drive gate beta=1 theta=0.5 | identity | 4 | 0.051605 | 0.059314 | 0.058146 | 0.060992 | 0.059181 | 0.058938 |
| shifted matrix log epsilon=3 | logit | 4 | 0.051652 | 0.061412 | 0.060635 | 0.062693 | 0.060845 | 0.061475 |
| neuron projected-drive gate beta=4 theta=1.5 | identity | 4 | 0.051854 | 0.057425 | 0.055914 | 0.058982 | 0.058008 | 0.056798 |
| projected drive plus threshold beta=8 theta=0.75 | identity | 5 | 0.051863 | 0.060508 | 0.058924 | 0.061257 | 0.061006 | 0.060843 |
| Fisher-occupancy arithmetic mix alpha=0.75 | logit | 4 | 0.051980 | 0.088244 | 0.087579 | 0.093147 | 0.083136 | 0.089115 |
| neuron projected-drive gate beta=1 theta=0.75 | identity | 4 | 0.052013 | 0.059180 | 0.058137 | 0.060934 | 0.058919 | 0.058729 |
| regularized precision lambda=10 p=2 | logit | 4 | 0.052051 | 0.062113 | 0.061405 | 0.063449 | 0.061448 | 0.062150 |
| projected drive plus threshold beta=2 theta=1 | identity | 5 | 0.052140 | 0.060485 | 0.058757 | 0.061354 | 0.061055 | 0.060774 |
| threshold-only gate beta=4 theta=1.5 | arcsine | 4 | 0.052336 | 0.058593 | 0.054479 | 0.058955 | 0.060678 | 0.060261 |
| regularized precision lambda=10 p=1 | logit | 4 | 0.052390 | 0.062977 | 0.062277 | 0.064391 | 0.062277 | 0.062964 |
| precision resolvent lambda=10 | logit | 4 | 0.052390 | 0.062977 | 0.062277 | 0.064391 | 0.062277 | 0.062964 |
| projected drive plus threshold beta=4 theta=0.75 | identity | 5 | 0.052439 | 0.060566 | 0.059021 | 0.061461 | 0.060994 | 0.060787 |
| projected drive plus threshold beta=2 theta=1.5 | identity | 5 | 0.052456 | 0.060320 | 0.058201 | 0.061084 | 0.061202 | 0.060794 |
| neuron projected-drive gate beta=1 theta=1 | identity | 4 | 0.052489 | 0.059070 | 0.058186 | 0.060910 | 0.058648 | 0.058536 |
| regularized precision lambda=10 p=0.5 | logit | 4 | 0.052567 | 0.063408 | 0.062709 | 0.064862 | 0.062690 | 0.063370 |
| projected drive plus threshold beta=2 theta=0.75 | identity | 5 | 0.052655 | 0.060575 | 0.058973 | 0.061528 | 0.061027 | 0.060773 |
| projected drive plus threshold beta=1 theta=1 | arcsine | 5 | 0.053037 | 0.059728 | 0.058215 | 0.060666 | 0.060010 | 0.060022 |
| projected drive plus threshold beta=1 theta=0.75 | arcsine | 5 | 0.053065 | 0.059744 | 0.058266 | 0.060696 | 0.059998 | 0.060017 |
| projected drive plus threshold beta=4 theta=1.5 | arcsine | 5 | 0.053077 | 0.058760 | 0.056455 | 0.059345 | 0.059756 | 0.059484 |
| projected drive plus threshold beta=1 theta=0.5 | arcsine | 5 | 0.053135 | 0.059756 | 0.058308 | 0.060723 | 0.059983 | 0.060008 |
| projected drive plus threshold beta=1 theta=1.5 | arcsine | 5 | 0.053167 | 0.059687 | 0.058098 | 0.060612 | 0.060022 | 0.060017 |
| projected drive plus threshold beta=1 theta=0.25 | arcsine | 5 | 0.053231 | 0.059761 | 0.058339 | 0.060747 | 0.059965 | 0.059995 |
| projected drive plus threshold beta=2 theta=0.5 | arcsine | 5 | 0.053271 | 0.059792 | 0.058429 | 0.060775 | 0.059957 | 0.060006 |
| neuron projected-drive gate beta=2 theta=1.5 | identity | 4 | 0.053307 | 0.057889 | 0.056707 | 0.059698 | 0.057834 | 0.057318 |
| SPD power p=2 | logit | 4 | 0.053555 | 0.068056 | 0.067065 | 0.069830 | 0.067571 | 0.067757 |
| threshold-only gate beta=2 theta=1.5 | arcsine | 4 | 0.053568 | 0.059563 | 0.055759 | 0.060032 | 0.061163 | 0.061297 |
| neuron projected-drive gate beta=1 theta=1.5 | identity | 4 | 0.053585 | 0.059051 | 0.058572 | 0.061082 | 0.058212 | 0.058339 |
| projected drive plus threshold beta=2 theta=0.25 | logit | 5 | 0.053811 | 0.059281 | 0.058033 | 0.060346 | 0.059226 | 0.059520 |
| threshold-only gate beta=1 theta=1.5 | logit | 4 | 0.054288 | 0.060418 | 0.057149 | 0.061228 | 0.061352 | 0.061942 |
| projected drive plus threshold beta=8 theta=1.5 | logit | 5 | 0.054325 | 0.057699 | 0.055319 | 0.058282 | 0.058694 | 0.058502 |
| projected drive plus threshold beta=4 theta=0.5 | logit | 5 | 0.054380 | 0.059291 | 0.058072 | 0.060349 | 0.059219 | 0.059524 |
| Fisher-occupancy arithmetic mix alpha=0.25 | logit | 4 | 0.054719 | 0.089818 | 0.085207 | 0.094154 | 0.088261 | 0.091649 |
| matrix exponential alpha=0.25 | logit | 4 | 0.054795 | 0.067632 | 0.066848 | 0.069569 | 0.066710 | 0.067399 |
| projected drive plus threshold beta=4 theta=0.25 | logit | 5 | 0.055360 | 0.058916 | 0.057634 | 0.060114 | 0.058819 | 0.059099 |
| population L4 | identity | 4 | 0.055448 | 0.061209 | 0.065065 | 0.062340 | 0.058649 | 0.058783 |
| threshold-only gate beta=1 theta=1 | logit | 4 | 0.055617 | 0.061541 | 0.058176 | 0.062628 | 0.062268 | 0.063093 |
| projected drive plus threshold beta=8 theta=0.5 | logit | 5 | 0.055895 | 0.059167 | 0.057959 | 0.060269 | 0.059068 | 0.059371 |
| projected drive plus threshold beta=8 theta=0.25 | logit | 5 | 0.056040 | 0.058253 | 0.056810 | 0.059557 | 0.058192 | 0.058451 |
| threshold-only gate beta=1 theta=0.75 | logit | 4 | 0.056311 | 0.062174 | 0.058756 | 0.063412 | 0.062794 | 0.063734 |
| matrix exponential alpha=0.5 | logit | 4 | 0.056324 | 0.069899 | 0.068963 | 0.072144 | 0.068901 | 0.069589 |
| threshold-only gate beta=2 theta=1 | logit | 4 | 0.056416 | 0.062420 | 0.058687 | 0.063702 | 0.063152 | 0.064138 |
| threshold-only gate beta=1 theta=0.5 | logit | 4 | 0.056999 | 0.062815 | 0.059347 | 0.064204 | 0.063330 | 0.064381 |
| threshold-only gate beta=4 theta=1 | logit | 4 | 0.057032 | 0.062906 | 0.058687 | 0.064233 | 0.063790 | 0.064914 |
| spectral condition | logit | 4 | 0.057079 | 0.080729 | 0.081281 | 0.089177 | 0.073483 | 0.078973 |
| threshold-only gate beta=1 theta=0.25 | logit | 4 | 0.057664 | 0.063441 | 0.059927 | 0.064972 | 0.063856 | 0.065009 |
| threshold-only gate beta=2 theta=0.75 | logit | 4 | 0.058159 | 0.064158 | 0.060358 | 0.065839 | 0.064557 | 0.065878 |
| threshold-only gate beta=8 theta=1 | identity | 4 | 0.058331 | 0.062919 | 0.057872 | 0.063892 | 0.064564 | 0.065347 |
| global intercept | logit | 1 | 0.058509 | 0.087417 | 0.084376 | 0.090797 | 0.085736 | 0.088758 |
| population Linf | identity | 4 | 0.059046 | 0.066993 | 0.068871 | 0.069576 | 0.064208 | 0.065317 |
| spectral effective rank | logit | 4 | 0.059180 | 0.086550 | 0.087276 | 0.094149 | 0.079099 | 0.085675 |
| threshold-only gate beta=4 theta=0.75 | logit | 4 | 0.059813 | 0.065694 | 0.061546 | 0.067687 | 0.065858 | 0.067685 |
| threshold-only gate beta=2 theta=0.5 | logit | 4 | 0.060070 | 0.065895 | 0.062063 | 0.067947 | 0.065984 | 0.067585 |
| matrix exponential alpha=1 | logit | 4 | 0.060138 | 0.074028 | 0.072627 | 0.076917 | 0.072921 | 0.073648 |
| threshold-only gate beta=8 theta=0.75 | logit | 4 | 0.060143 | 0.065920 | 0.061443 | 0.067869 | 0.066079 | 0.068287 |
| stage progression | logit | 4 | 0.061319 | 0.081168 | 0.075907 | 0.085376 | 0.079473 | 0.083915 |
| threshold-only gate beta=2 theta=0.25 | logit | 4 | 0.062005 | 0.067511 | 0.063681 | 0.069871 | 0.067340 | 0.069154 |
| correlation power p=2 | logit | 4 | 0.062819 | 0.102795 | 0.095143 | 0.109489 | 0.101984 | 0.104564 |
| axis intercepts | logit | 3 | 0.062839 | 0.086199 | 0.083602 | 0.089987 | 0.083970 | 0.087235 |
| correlation power p=0.5 | logit | 4 | 0.062913 | 0.104551 | 0.096971 | 0.111308 | 0.104030 | 0.105893 |
| correlation power p=-0.5 | logit | 4 | 0.062977 | 0.106000 | 0.098474 | 0.112857 | 0.105651 | 0.107019 |
| correlation power p=-1 | logit | 4 | 0.063010 | 0.106654 | 0.099141 | 0.113563 | 0.106384 | 0.107528 |
| correlation power p=-2 | logit | 4 | 0.063076 | 0.108040 | 0.100505 | 0.115091 | 0.107930 | 0.108632 |
| scale plus shape | logit | 5 | 0.064049 | 0.051578 | 0.049740 | 0.052475 | 0.051662 | 0.052436 |
| threshold-only gate beta=4 theta=0.5 | logit | 4 | 0.064781 | 0.069067 | 0.065194 | 0.071723 | 0.068520 | 0.070832 |
| Fisher isotropic | logit | 4 | 0.067984 | 0.059024 | 0.070972 | 0.053944 | 0.057784 | 0.053396 |
| threshold-only gate beta=4 theta=0.25 | logit | 4 | 0.069413 | 0.072545 | 0.068919 | 0.075640 | 0.071580 | 0.074043 |
| threshold-only gate beta=8 theta=0.5 | logit | 4 | 0.069974 | 0.070849 | 0.067276 | 0.073786 | 0.069700 | 0.072633 |
| scale only | logit | 4 | 0.074383 | 0.048371 | 0.048429 | 0.048552 | 0.049964 | 0.046540 |
| log-volume reference | logit | 4 | 0.074383 | 0.048371 | 0.048429 | 0.048552 | 0.049964 | 0.046540 |
| threshold-only gate beta=8 theta=0.25 | logit | 4 | 0.075014 | 0.076584 | 0.073408 | 0.080032 | 0.075153 | 0.077743 |
| Fisher-occupancy arithmetic mix alpha=0.5 | logit | 4 | 0.076361 | 0.086475 | 0.083678 | 0.090274 | 0.084387 | 0.087561 |
| AIRM reference | logit | 4 | 0.080668 | 0.168763 | 0.171110 | 0.177881 | 0.163214 | 0.162845 |
| spectral trace | logit | 4 | 0.087176 | 0.058375 | 0.055700 | 0.059652 | 0.060506 | 0.057641 |
| AIRM plus log-volume | cloglog | 5 | 0.233360 | 0.149045 | 0.152216 | 0.156130 | 0.144569 | 0.143264 |

### Alternate fixation-bias partition robustness

The Exp1-main fitted coefficients are applied without refitting to the authors' alternate fixation-bias stage assignment and matching decoder cache. This reuses the same experiment and is a robustness check, not a new cohort.

| Producer | Link | RMSE | MAE | Pearson r | Spearman rho |
|---|---|---:|---:|---:|---:|
| Fisher total | probit | 0.026120 | 0.020873 | 0.805066 | 0.776224 |
| neuron projected-drive gate beta=8 theta=0.75 | logit | 0.027065 | 0.021253 | 0.791329 | 0.804196 |
| neuron projected-drive gate beta=8 theta=1 | logit | 0.027282 | 0.021449 | 0.786330 | 0.783217 |
| neuron projected-drive gate beta=4 theta=0.75 | logit | 0.027336 | 0.021370 | 0.785546 | 0.804196 |
| neuron projected-drive gate beta=4 theta=0.5 | logit | 0.027357 | 0.021241 | 0.785334 | 0.804196 |
| spectral exponential saturation beta=0.3 | logit | 0.027368 | 0.021459 | 0.786416 | 0.804196 |
| spectral ratio saturation alpha=3 | probit | 0.027370 | 0.021803 | 0.787243 | 0.804196 |
| precision resolvent lambda=1 | logit | 0.027429 | 0.021547 | 0.787439 | 0.804196 |
| regularized precision lambda=1 p=1 | logit | 0.027429 | 0.021547 | 0.787439 | 0.804196 |
| shifted matrix log epsilon=0.3 | logit | 0.027431 | 0.021578 | 0.786707 | 0.804196 |
| SPD power p=-1 | logit | 0.028690 | 0.023654 | 0.772887 | 0.755245 |
| spectral ratio saturation alpha=0.1 | arcsine | 0.029607 | 0.025016 | 0.753287 | 0.755245 |
| relative precision stretch | logit | 0.033138 | 0.025778 | 0.764434 | 0.853147 |
| relative information stretch | logit | 0.035401 | 0.028443 | 0.752759 | 0.825175 |

### Cross-surface discovery winner

- Producer: `relative precision stretch`
- Formula: `$a_j+b sqrt(g_k(e_j,e_j)/g_1(e_j,e_j))$`
- Link: `logit`
- Mean Exp2-binning RMSE: `0.030212`
- Alternate fixation-bias RMSE: `0.033138`
- Equal-weight two-surface mean RMSE: `0.031675`

This combined ranking is descriptive and post-discovery, but it identifies the equation that remains accurate across rule-generalization binnings and the alternate fixation-bias partition instead of optimizing only one surface.

The four Exp2 binnings reuse the same underlying sessions, so agreement across them is robustness to binning rather than four independent replications. The producer universe was expanded after inspecting earlier Exp2 results, making this a discovery tournament. A fresh session-level cohort is required for confirmation.

## Geometry trajectory decomposition

The covariance path is decomposed into determinant scale, determinant-normalized anisotropy, fixed-chart spectral diagnostics, effective rank, and successive AIRM motion. Ordered-projector displacement is retained only as a chart-dependent diagnostic because eigenvalue crossings or small gaps make individual lower eigenvectors unstable. Dominant-axis angle and eigengap are the interpretable rotation checks.

| Dataset | Stage | Scale change | Anisotropy | Effective rank | Ordered-projector displacement | Dominant angle (deg) | Dominant gap | Min gap | AIRM from stage 1 | Successive AIRM | Axis cost ratio | det-normalized eigenvalues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Exp1 | 1 | +0.000000 | 0.516050 | 2.857577 | 0.000000 | 0.000 | 0.453594 | 0.050722 | 0.000000 | 0.000000 | 1.105470 | `[1.52239, 0.83184, 0.78965]` |
| Exp1 | 2 | +0.089842 | 0.570167 | 2.869525 | 1.048878 | 43.816 | 0.058999 | 0.058999 | 0.588230 | 0.588230 | 1.321073 | `[1.30019, 1.22348, 0.62863]` |
| Exp1 | 3 | +0.327537 | 0.753691 | 2.703012 | 0.888980 | 36.618 | 0.576646 | 0.109599 | 0.921645 | 0.867526 | 1.543679 | `[1.84361, 0.7805, 0.69496]` |
| Exp1 | 4 | -0.198946 | 1.017375 | 2.498166 | 1.327995 | 46.982 | 0.660738 | 0.245585 | 1.136529 | 1.044212 | 1.854845 | `[2.25826, 0.76614, 0.57799]` |
| Exp2 | 1 | +0.000000 | 1.341505 | 2.250959 | 0.000000 | 0.000 | 0.722476 | 0.434653 | 0.000000 | 0.000000 | 1.633749 | `[2.84246, 0.78885, 0.44597]` |
| Exp2 | 2 | -0.618148 | 0.642538 | 2.814595 | 1.331619 | 38.697 | 0.339197 | 0.339197 | 1.608826 | 1.608826 | 1.344957 | `[1.55366, 1.02666, 0.62692]` |
| Exp2 | 3 | -0.317268 | 1.149061 | 2.509028 | 1.481309 | 84.918 | 0.512139 | 0.512139 | 1.669442 | 1.083347 | 1.996842 | `[2.18088, 1.06397, 0.43096]` |
| Exp2 | 4 | -0.623766 | 0.796253 | 2.678518 | 1.478679 | 75.632 | 0.564303 | 0.215639 | 1.966164 | 0.778724 | 1.532882 | `[1.88671, 0.82203, 0.64477]` |

## Pooled-information-weighted cross-task cosine

For the two official stimulus-set selectivity matrices, a separately pooled second-moment weighting gives

$$
\cos_{H^{-1}}(S_1,S_2)=\frac{\operatorname{tr}(S_1H^{-1}S_2^T)}{\sqrt{\operatorname{tr}(S_1H^{-1}S_1^T)\operatorname{tr}(S_2H^{-1}S_2^T)}}.
\qquad H=(S_1^TS_1+S_2^TS_2)/(2N).
$$

The Euclidean and pooled-information columns summarize all three factor coordinates. This $H^{-1}$ is built from both task matrices and is not the separately discovered stage metric $g_k=C_k^{-1}$. The named-axis columns follow the variables actually computed in the authors' `figure_4.py`.
These are parallel descriptive readouts, not a demonstrated equality between single-neuron alignment and a metric eigenspace.

| Stage | Early Euclidean cos | Early pooled-information cos | Late Euclidean cos | Late pooled-information cos | Early colour | Early context | Late shape | Late XOR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.079536 | -0.050117 | 0.120709 | 0.180770 | -0.149823 | 0.442327 | 0.149701 | 0.399433 |
| 2 | 0.201041 | 0.188715 | 0.316220 | 0.311941 | 0.238010 | 0.288946 | 0.321860 | 0.379541 |
| 3 | 0.328019 | 0.205775 | 0.326460 | 0.327477 | 0.519529 | 0.094229 | 0.229029 | 0.561382 |
| 4 | 0.212505 | 0.160412 | 0.295404 | 0.374014 | 0.368520 | 0.074412 | 0.475854 | 0.610862 |

## Subspace-conditioned routing hypothesis

The pasted routing proposal contains a useful hypothesis but its additive formulas need two corrections. A weighted sum of projectors is generally not itself a projector, and an additive metric update need not remain SPD. A finite, typed replacement is

$$
z(c)=\operatorname{softmax}(Ac+b),\qquad
U(z)=\operatorname{qf}\left(U_0+\sum_a z_aB_a\right),\qquad
\Pi(z)=U(z)U(z)^T,
$$

where the `qf` input must have full column rank. If a soft continuous gate rather than a literal subspace is intended, use $G(z)=U\operatorname{diag}(\sigma(Kz+d))U^T$ and call it a gain operator, not a projector.

In coordinate-free form, let $\mathcal S(z)$ be a dimensionless $g_0$-self-adjoint endomorphism and define

$$
g_z(u,v)=g_0\!\left(\exp(\mathcal S(z))u,v\right),\qquad
\mathcal S(z)=\sum_a z_a\mathcal S_a.
$$

This guarantees $g_z\succ0$. Metric deformation alone still does not select a destination or generate motion. One explicit fixed-chart bridge is

$$
D_z=\Pi_zg_z^{-1}\Pi_z+\epsilon(I-\Pi_z)g_z^{-1}(I-\Pi_z),\qquad
\dot x=-\kappa D_z\nabla_xV(x,z)+f_\perp(x,z),\quad \epsilon>0.
$$

The target-dependent potential $V$ and any non-gradient drift $f_\perp$ are additional hypotheses. Thus global search can be amortized into a learned controller/value field, but its computational cost is moved offline or into $z(c)$ and $V$; it is not proven to disappear.

### Observable PFC middle-link discovery test

A time- and coordinate-matched check uses fold 0 of the official 100-150 ms task matrices. Let $X_k$ be task 1, $Y_k$ task 2, and $C_k=\operatorname{Cov}(X_k)$. The two stage-level observables are

$$
d_k=d_{\mathrm{AI}}(C_1,C_k),\qquad
q_k=\frac{\langle X_k,Y_k\rangle_F}{\|X_k\|_F\|Y_k\|_F}.
$$

- $d_k$: `[0.0, 0.455685, 0.850545, 0.558887]`
- $q_k$: `[0.120709, 0.31622, 0.32646, 0.295404]`
- Pearson $r(d,q)$: `0.908400`
- Stagewise independent task-2 released-row shuffles: one-sided `p=0.044598`, two-sided `p=0.088496` over `20000` draws

This is the strongest available same-cache middle-link result: larger task-1 geometry displacement accompanies stronger matched task-2 alignment. The shuffle tests neuron-row correspondence conditional on four released pseudopopulation stages; it is post-discovery and is not an animal/session population test.

The time-matched shape/XOR control does not reproduce the relationship:

- accessibility: `[[-0.0, -0.0], [0.048827, 0.016377], [0.182237, 0.146587], [0.206628, -0.035472]]`
- official alignment: `[[0.149701, 0.399433], [0.32186, 0.379541], [0.229029, 0.561382], [0.475854, 0.610862]]`
- Pearson: `0.015998`; author-null two-sided `p=0.967033` over `1000` draws

Therefore the released data contain a positive 3D discovery signal, but not a uniform axis-by-axis routing law. Two numerical corrections to the pasted interpretation are required: `-0.149823 -> 0.368520` is computed from the authors' early **colour** variables even though their exported plot labels that column as context, and `0.149701 -> 0.475854` is late **shape-selectivity alignment**. Neither series is metric alignment. The pooled-information cosine above is another same-cache statistic and must not be renamed as $g_k$.

No primary-chart context coupling is reported: the 70-100 ms `[set, set*context, context]` metric and the separate Fig. 4 selectivity design do not supply an exact same-coordinate bridge.

### External evidence boundary

- [Tafazoli et al.](https://doi.org/10.1038/s41586-025-09805-2) directly support shared, task-selectively engaged sensory and motor subspaces; they do not fit a metric tensor.
- [Binish et al.](https://doi.org/10.1038/s41593-026-02290-4) directly support a low-dimensional PFC-M1 communication subspace predictive of context-dependent action; they do not measure geodesic or routing cost.
- [Gonzalez et al.](https://doi.org/10.1038/s41586-026-10481-z) support task/state-dependent hippocampal-retrosplenial communication subspaces and sleep reactivation; they do not identify $g_z$.
- The primary dendrite study is [Maristany de Las Casas et al., Science](https://doi.org/10.1126/science.adx4358), not the cited Nature Neuroscience research highlight. It supports local rule-dependent dendritic gating, not a global Riemannian router.

The evidence therefore supports `shared subspace engagement` and a PFC `relative precision deformation` candidate separately. The combined chain $c\to\Pi_z\to g_z\to\dot x$ remains a falsifiable model, not an observed biological identity.

## Equation availability boundary

| Equation family | Status on this official local release |
|---|---|
| $S^TQ^{-p}S$ with measured residual $Q$ | UNAVAILABLE: trial residual covariance is not released in the local processed cache |
| $(\Sigma+\lambda I)^{-p}$ and spectral functions | EXECUTED in the finite tournament |
| Fisher-occupancy arithmetic and affine-invariant geometric mixtures | EXECUTED as fixed-chart sensitivities |
| Increment mobility $[\operatorname{Cov}(r_{t+1}-r_t)+\lambda I]^{-1}$ | UNAVAILABLE: raw neural time series/spikes are absent locally |
| Relative stretch, scale-shape, effective rank, eigenbasis rotation | EXECUTED |
| Pooled-information-weighted cross-task cosine | EXECUTED on the official task1/task2 selectivity matrices; distinct from $g_k$ |
| Late 3D $d_{\mathrm{AI}}$ to cross-task alignment | EXECUTED as a post-discovery released-row conditional test |
| State-dependent curvature/geodesic trajectory | UNAVAILABLE: only stagewise constant summary matrices are present |
| Structural $W\rightarrow g$ | UNAVAILABLE: no structural connectivity $W$ |

## Geometry and official behavior summaries

For each stage, the table compares behavior with distance from the first-stage covariance and signed log-volume change:

$$
d_k=d_{\mathrm{AI}}(C_1,C_k),\qquad v_k=\tfrac13\log\det(C_1^{-1}C_k).
$$

| Dataset | Endpoint | Stages | Spearman(d, behavior) | Spearman(v, behavior) |
|---|---|---:|---:|---:|
| Exp1 | fixation-break ratio | 4 | 1.000000 | -0.200000 |
| Exp1 | colour switch | 4 | 1.000000 | -0.200000 |
| Exp1 | shape switch | 4 | 0.400000 | -0.800000 |
| Exp1 | hierarchical switch | 4 | 1.000000 | -0.200000 |
| Exp2 | fixation-break ratio | 3 | 1.000000 | -1.000000 |
| Exp2 | fixation-break ratio | 4 | 0.800000 | -0.400000 |
| Exp2 | fixation-break ratio | 5 | 0.700000 | -0.700000 |
| Exp2 | fixation-break ratio | 6 | 0.771429 | -0.485714 |

These behavior correlations use only 3-6 ordered stage means and receive no p-value. They are external endpoint sensitivities, not trial-level neural mediation tests.

## Held-out prediction check

For the primary four-stage Exp2 comparison, a zero-mean Gaussian with the full stage-specific metric was fitted on four folds and scored on the unseen fifth fold. The table reports `alternative NLL - full-metric NLL`, so positive values favor the full stage-specific metric. The 200 x 5 folds are repeated prediction checks, not independent biological samples.

| Alternative | Mean held-out NLL penalty (nat/row) | Folds won by full metric |
|---|---:|---:|
| pooled full | 0.148638 | 833/1000 |
| stage diagonal | 0.092249 | 740/1000 |
| stage spherical | 0.231698 | 882/1000 |

## Matched functional readouts

The metric-axis cost change is `log((e_i^T g_last e_i)/(e_i^T g_first e_i))`. Decoder p-values use the authors' 1,000 learning-epoch-reassignment nulls, not the row permutation above.

| Experiment | Named axis | Metric log-cost change | Decoder change | Author LER p |
|---|---|---:|---:|---:|
| Exp1 | colour | +0.615599 | -0.133333 | 0.030969 |
| Exp1 | shape | +0.469348 | -0.141327 | 0.000999 |
| Exp1 | XOR | -0.565714 | +0.038350 | 0.539461 |
| Exp2 | set | +1.282997 | -0.175000 | 0.000999 |
| Exp2 | set*context (XOR2) | +0.426475 | -0.074575 | 0.028971 |
| Exp2 | context | -0.258498 | -0.017687 | 0.590410 |

Across the six pre-named axes, accessibility change `-Delta log cost` and decoder change have Pearson `r=0.952609` with 5/6 matching signs. This is an exploratory alignment summary only; stages and task axes are not exchangeable inferential units, so no permutation p-value is assigned.

## Input integrity

- `selectivity_coefficients_exp1_140_1504stages.pickle`: `a5c0b1ad9b6f0b533449b3983b553b49fbeb12fb084e19843433f627f528bfac`
- `selectivity_coefficients_exp1_fixbias_140_1504stages.pickle`: `bfb52ec0f14fdfc7313ed48eba2ebfc2e1d51a33524e79760dc625accf6dd862`
- `selectivity_coefficients_exp2_70_100_3stages.pickle`: `dc79fbe1b3eef4973d837266c7cec36b0127e570b7d5b04b8adfa1efd895d5c0`
- `selectivity_coefficients_exp2_70_100_4stages.pickle`: `a0bf42f992ac4b5575ef44103f14a6107651e4f5d45c22755f7c37c15be5315f`
- `selectivity_coefficients_exp2_70_100_5stages.pickle`: `f9445dfb7786f26c138931078ebcece97d0906159e7df3390bb7f2dbc7797eca`
- `selectivity_coefficients_exp2_70_100_6stages.pickle`: `6284ccedff1d80110cc4f70c259bcd72e5741f42812fefa58e25b5857e618896`
- `exp1_decoding_collocked_50_150_4stages.pickle`: `11e429d2c207ef2ddb6ee6e080ef671bec115057564c7f319a849202dd7e8206`
- `exp1_decoding_shapelocked_100_150_4stages.pickle`: `6539e4d510792c35531d9da4f7cab8963b72f7abac7c4b7735c46b6e10f6b4c2`
- `exp1_beh_terminations4stages.pickle`: `c041005c75abae6e28cf63b1e096188c78077306424a0ff1a445f78135ff8e7d`
- `exp1_switch_costs4stages.pickle`: `c6e9fb2bd9c41cf355a626819b68dd333fd407214e3cce1195f9edd913ff15b7`
- `exp2_selectivity_dat_early_50_100_late_100_150_stages_4.pickle`: `b64f6fa81b0362957465e09e9f47cfce14fed93ca35448a2ef186deed110bc30`
- `exp1_decoding_fixbias_collocked_50_150_4stages.pickle`: `2d1c5a365ebc5a3d83b79e9d48e83e056ebbdd2c4599817e0fe93642570d2dce`
- `exp1_decoding_fixbias_shapelocked_100_150_4stages.pickle`: `291a52fb48d3757a61ea22f37cb831fcb30cad9d504de6d163d7b77524323804`
- `exp2_decoding_time_avg_3stages_50_100.pickle`: `29ec0446292aa9b870b3c059a637c3239274341d4bbeb03402f11e1eb855de3f`
- `fixation_breaks_prop_exp2_3stages.pickle`: `207e91c24db3c11393c871b226b6b5d94f33cce83f4153fc6fe45bd1ced3ebc0`
- `exp2_decoding_time_avg_4stages_50_100.pickle`: `d36f587accf5aa9a71470b13753b5ecb6d9f66fb0f2d7060c0bbb0125a1561db`
- `fixation_breaks_prop_exp2_4stages.pickle`: `72ebf08ed946cea6ed8b5b0b6efb93254a46d733d724be1f00c38e2ba297377f`
- `exp2_decoding_time_avg_5stages_50_100.pickle`: `7f812f4de04db0deeba1b1c6b57fcbfa62b25228d303f9b7affdde3d6d7541cd`
- `fixation_breaks_prop_exp2_5stages.pickle`: `508497258fdba26ed2c13ce67362b1d020e2e5c45eb373feee5703028a4a3cfb`
- `exp2_decoding_time_avg_6stages_50_100.pickle`: `75aeda585841a54cfc4b8f87c64ded3fc074ac41d3003243bbecd1114e9e7261`
- `fixation_breaks_prop_exp2_6stages.pickle`: `3b8bbf4aeb5d271bd83e9d58a619ac98c17989c5e7e5729c4b6358e90e39e868`

## Scope

This is an official real-data test of stage-specific inverse-covariance geometry and a separately derived Fisher-pullback decoder bridge on released PFC selectivity pseudopopulations. It is not a mock-data result.

The released rows do not retain session or animal identifiers, and the Exp1 cache's two apparent folds are exact duplicates. Therefore row exchangeability and row-fold prediction are sensitivity analyses, not animal-population or held-out-session inference.

Every equation supported by these released stagewise 3D selectivity summaries is evaluated above. The Fisher pullback uses the explicitly stated homoscedastic z-score approximation and absorbs the missing residual-noise scale into kappa. Process-noise reachability, controllability Gramians, fully noise-calibrated or state-dependent Fisher fields, curvature, geodesic trajectory prediction, directed action, structural W producers, and the causal chain Delta W -> Delta g -> Delta x require raw trajectories, perturbation channels, spatial fields, or connectivity that are absent here. Substituting fabricated inputs for those equations would be a mock analysis, so they are deliberately not computed.
