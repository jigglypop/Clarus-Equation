# Routes — C3 binding/latch transition families

Status: COMPLETE

LANE: route exploration (ce-route-explorer). No closure, no promotion and no first-principles claim is made here; escalation is the business of ce-closure-gate / ce-status-auditor.

## 1. Target statement (one sentence)

Find learnable transition-plus-readout families $h_{t+1}=F_\theta(h_t,x_t)$, $\hat y=\operatorname{sign} f_\theta(h_T)$, real-valued and dimensionless, on the fixed budget $\dim h \le 20$ over the frozen v13 balanced-split domain $x_t\in\mathbb{R}^{20}$, $t=0..T-1$, $T\in\{4,8\}$, such that (i) a signal tick is captured without loss, (ii) it is retained with closed-interval eigenvalue exactly $1$, and (iii) the readout expresses the bilinear form $c^{\top}\hat W b$ of two separately stored latents; every reported quantity is an accuracy in $[0,1]$ against the reference gru20 $\approx 0.889$ (iid panels) and $0.551$ (heldout).

## 2. Canon-forced vs free

- Forced by canon (00-contract plus the v13 ledger): the episode generator and the 20-channel raw encoding (frozen); the protocol (Adam $lr=0.01$, $wd=10^{-4}$, clip $1.0$, 200 epochs, full batch, BCEWithLogits, 192 train / 256 evaluation episodes, balanced split); the five panels; the state budget $n=20$; retention $r=1$ permitted in the closed interval.
- Free: the algebraic form of the write gate; the algebraic form of the binding (outer product, circular convolution, diagonal factor basis, routing simplex); the slot partition of the 20 state scalars; the readout feature map.

## 3. New axiom introduced by this lane (exactly one, explicit)

**A-SAL (salience is an even functional of the drive).** The write gate is built from an even functional of the input,
$$g_t=\sigma\!\left(a\,\lVert m\odot x_t\rVert^{2}+b\right),$$
rather than from a linear form $g_t=\sigma(u^{\top}x_t+\beta)$.

Why this is forced rather than fitted: the bits tick carries per-episode $\pm1$ signs, so for any fixed $u$ the projection $u^{\top}x_0$ changes sign across episodes and a sigmoid-of-linear gate cannot open on the bits tick for all bit patterns, while it can open on the context tick (one-hot, single sign). This is a structural obstruction, not an optimisation accident, and it matches the v10/v13 failure signature exactly: the signal is never latched, hence the $T=8$ collapse. The energy $\lVert x_t\rVert^{2}$ is sign-blind and separates $\lVert x_0\rVert^{2}\approx16$ and $\lVert x_1\rVert^{2}\approx1$ from noise ticks, $\lVert x_t\rVert^{2}\approx 20\sigma^{2}\le0.13$.

The axiom is shared by candidates K, L, M, N; candidates A-J and the reference use no new axiom. Because it was formulated after inspecting the first result table, every candidate that uses it is marked target-aware below.

## 4. Candidate table

$n$ = recurrent state scalars; params = exact torch parameter count at the tested width; panel entries are seed means over 9000-9005 (6 seeds), protocol of section 2. "marg. stab." asks whether closed-interval eigenvalue $1$ is structural (holds for every parameter value) or only learned.

| # | route | transition | readout | params | $n$ | marg. stab. | tgt-aware | id | noise | horiz | comb | held | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R | gru20 reference | GRU | linear | 2541 | 20 | learned | no | 0.891 | 0.891 | 0.890 | 0.889 | 0.551 | reference |
| A | salience-latch + bilinear | $s'=(1-g)s+g\tanh(Ux)$, $g=\sigma(u^{\top}x+\beta)$ | $\langle W,s_c s_b^{\top}\rangle$ | 315 | 12 | structural | no | 0.786 | 0.763 | 0.739 | 0.683 | 0.473 | weak |
| B | HRR circular-convolution binding | $h'=(1-g)h+g\,(h\circledast\rho(x)+\tanh(Ux))$ | $\langle w,h\rangle$ | 350 | 8 | structural | no | 0.812 | 0.739 | 0.762 | 0.648 | 0.561 | weak |
| C | key-value slot, simplex key | $M'=M+g_m\,\alpha(x)s^{\top}$, $\alpha=\mathrm{softmax}(W_kx)$ | $\langle W,\operatorname{vec}M\rangle$ | 219 | 20 | structural | no | 0.805 | 0.783 | 0.796 | 0.740 | 0.583 | weak |
| D | fast-weight outer product | as C with free key $\alpha=W_kx$ | $\langle W,\operatorname{vec}M\rangle$ | 219 | 20 | structural | no | 0.755 | 0.715 | 0.766 | 0.714 | 0.507 | weak |
| E | orthogonal rotation + quadratic | $h'=Rh+Bx$, $R$ block rotation, no gate | $\langle w,h\rangle+h^{\top}Qh$ | 237 | 8 | structural | no | 0.911 | 0.873 | 0.767 | 0.752 | 0.667 | weak-plus |
| F | gated integrator + quadratic | $h'=h+g\tanh(Bx)$, one slot | $\langle w,h\rangle+h^{\top}Qh$ | 254 | 8 | structural | no | 0.876 | 0.852 | 0.880 | 0.827 | 0.544 | weak (C1(ii) control) |
| G | oracle-gate latch + bilinear | A with $g$ = thresholded channel-group energy, 0 gate dof | $\langle W,s_c s_b^{\top}\rangle$ | 315 | 12 | structural | yes (oracle) | 0.979 | 0.958 | 0.978 | 0.956 | 0.952 | diagnostic bound, not a route |
| H | latch + linear readout | A transition | $\langle w,[s_b;s_c]\rangle$ | 295 | 12 | structural | no | 0.686 | 0.658 | 0.639 | 0.601 | 0.424 | C2 control |
| I | tapped delay line + bilinear | nilpotent shift, no gate, $\lambda=0$ | $\langle W,\tau_k\tau_{k-1}^{\top}\rangle$ | 126 | 20 | absent by construction | no | 0.980 | 0.973 | 0.502 | 0.497 | 0.954 | axis-separating control |
| J | factored multiplicative RNN | $h'=(1-g)h+g\tanh(Ux+W_h(\rho(x)\odot Vh))$ | $\langle w,h\rangle$ | 478 | 8 | structural | no | 0.832 | 0.791 | 0.737 | 0.657 | 0.574 | weak |
| K | energy-gate latch + bilinear | A with A-SAL gate | $\langle W,s_c s_b^{\top}\rangle$ | 317 | 12 | structural | yes | 0.889 | 0.885 | 0.891 | 0.887 | 0.566 | weak on heldout |
| L | energy-gate HRR binding | B with A-SAL gate | $\langle w,h\rangle$ | 351 | 8 | structural | yes | 0.988 | 0.964 | 0.975 | 0.905 | 0.964 | GO-candidate |
| M | energy-gate key-value slot | C with A-SAL gates | $\langle W,\operatorname{vec}M\rangle$ | 221 | 20 | structural | yes | 0.858 | 0.826 | 0.803 | 0.783 | 0.824 | GO-candidate (weaker iid) |
| N | energy-gate latch, asymmetric gate init | K with per-slot random mask init | $\langle W,s_c s_b^{\top}\rangle$ | 317 | 12 | structural | yes | 0.906 | 0.904 | 0.907 | 0.901 | 0.632 | weak-plus, bimodal |

Ranking, low dof and strong cross-prediction first: $\mathrm{L}\succ \mathrm{M}\succ \mathrm{N}\succ \mathrm{K}\succ \mathrm{E}\succ \mathrm{C}\succ \mathrm{F},\mathrm{J},\mathrm{B},\mathrm{D},\mathrm{A}$. G, H, I are controls, not routes.

## 5. Pre-registered replication on fresh seeds

L, M, K, N were constructed after inspecting the first table, so they are target-aware. A fresh seed set 9100-9107 (8 seeds, identical protocol, no re-tuning, no re-selection) was run as confirmation.

| route | id | noise | horiz | comb | held | held $[\min,\max]$ |
|---|---|---|---|---|---|---|
| L energy-HRR | 0.994 | 0.974 | 0.984 | 0.919 | 0.972 | $[0.840,1.000]$ |
| M energy-KV | 0.853 | 0.838 | 0.810 | 0.791 | 0.737 | $[0.625,0.871]$ |
| K energy-latch | 0.925 | 0.918 | 0.918 | 0.909 | 0.703 | $[0.367,1.000]$ |
| N asym energy-latch | 0.914 | 0.903 | 0.906 | 0.890 | 0.668 | $[0.328,1.000]$ |
| G oracle gate (control) | 0.945 | 0.924 | 0.950 | 0.926 | 0.877 | $[0.605,1.000]$ |
| I delay line (control) | 0.973 | 0.970 | 0.500 | 0.501 | 0.930 | $[0.773,1.000]$ |
| B linear-gate HRR | 0.879 | 0.808 | 0.836 | 0.698 | 0.684 | $[0.551,0.980]$ |
| R gru20 | 0.891 | 0.887 | 0.893 | 0.892 | 0.563 | $[0.375,0.711]$ |

Reference thresholds recomputed on this same run ($0.95\times$ gru20 gives $0.846/0.843/0.848/0.847$ on id/noise/horizon/combined; the contract's G3 asks heldout $\ge0.90$): L clears all four iid thresholds and the heldout threshold; K and N clear the four iid thresholds and miss heldout; M clears id only. This lane records the numbers and does not evaluate the gates; the scored run is the business of the implementation stage (30/31).

## 6. Degrees of freedom and look-elsewhere

- Structural choices per candidate: widths ($d_b,d_c,d,k$) fixed once at the state budget and never swept; gate bias init fixed at $-1$ for all learned-gate routes; no learning-rate, epoch or width tuning was performed on any candidate.
- Post-hoc changes actually made: (1) the oracle gate G was corrected from a single shared energy threshold to per-channel-group thresholds after it produced chance accuracy by mutual overwrite; (2) A-SAL was introduced after the first table (routes K, L, M); (3) the mask initialisation of N. Three post-hoc moves total.
- Look-elsewhere size: 14 configurations trained on the main seed set plus 1 (N) plus 8 on the replication set, i.e. 15 distinct architectures scored, of which 4 use the single new axiom. No selection over seeds, no selection over epochs, no early stopping anywhere.

## 7. Per-candidate notes and killing falsification tests

Detailed implementations and raw per-seed records live in `artifacts/v14_route_toy.py`, `artifacts/v14_routes_main.json`, `artifacts/v14_routes_replication.json`, `artifacts/v14_routes_N.json`.

**A / K / N (salience latch + explicit bilinear readout).** Transition $s'=(1-g)s+g\tanh(Ux)$ is a convex combination, so $g\to0$ gives eigenvalue exactly $1$ structurally; no learned parameter can break it. Gradient path: because the closed interval is an exact identity map, no BPTT decay occurs; the only long-range credit assignment needed is on the gate itself. A fails because the linear gate cannot detect the sign-varying bits tick (A-SAL); K fixes the gate and recovers all four iid panels to reference level, but its two gates open on the same ticks, so the context write partly overwrites the bits latch and the state is not factorised, which is precisely the C1(ii) condition; heldout stays at $0.566$. N breaks the gate symmetry by initialisation and moves heldout to $0.632$ (main) / $0.668$ (replication) with per-seed range $[0.33,1.00]$: the family can find the separated solution but the landscape is bimodal. **Killing test:** if a per-slot gate that is architecturally forbidden to open on the same tick (for example a mutual-exclusion or refractory constraint) still leaves heldout below $0.90$ at 8 seeds, the latch-plus-outer-product family is falsified as the mechanism and the fault lies in the readout capacity instead.

**B / L (multiplicative recurrent binding by circular convolution).** $\rho=\delta$ is the identity of circular convolution, so $g\to0$ leaves $h$ pointwise unchanged: marginal stability is structural on two counts. A linear readout on a convolution-bound state is already a bilinear form, $\langle w,u\circledast v\rangle=u^{\top}C(w)v$ with $C(w)$ circulant, and because $u=U x_0$ and $v$ enter through learned encoders the effective form $U^{\top}C(w)V$ spans general matrices; the readout therefore needs only $d$ free parameters instead of $d_b d_c$. That is why L generalises where K does not: binding is a product, not an overwrite, so the two writes cannot erase each other, and the circulant constraint removes the readout freedom that lets K memorise the 24 train cells. L is the lowest-dof route that reaches the reference on every iid panel and $0.96$-$0.97$ on heldout. **Killing test:** replace the circulant binding by an unconstrained bilinear readout of equal parameter count and keep the same energy gate; if heldout stays at $0.96$ the circulant structure is irrelevant and the claim reduces to A-SAL alone. Second killing test: at $T=16$ or $T=32$ the closed-interval identity predicts no degradation beyond the noise term; any decay falsifies the structural-marginal-stability account.

**C / D / M (gated outer-product memory).** $M'=M+g\,\alpha s^{\top}$ is a pure integrator, eigenvalue exactly $1$ with no parameter dependence at all, and the readout on $\operatorname{vec}M$ is bilinear in the two latents by construction. Difference from a one-head mini attention: the softmax here is over a fixed slot inventory of size $k$ and never over time, and the retention across ticks is carried by the integrator rather than by re-reading a stored sequence, so memory cost is $O(kd)$ independent of $T$. The simplex key (C, M) generalises better than the free key (D) at equal parameter count, consistent with the normalisation acting as a capacity constraint. M reaches heldout $0.824$ / $0.737$ but its iid panels sit $0.03$-$0.09$ below reference, which the write-then-stage ordering explains: the staging latch must already hold the bits when the context tick arrives, so a single mis-timed gate costs the whole episode. **Killing test:** if forcing $k=4$ slots with a hard one-hot key (straight-through) does not lift the iid panels to reference, the routing family is falsified for this operating point.

**E (orthogonal recurrence + quadratic readout, no gate).** All eigenvalues have unit modulus by construction, so retention is structural without any gate; both latents live in one state and the quadratic readout supplies the product. It reaches the best heldout of the no-new-axiom routes ($0.667$) and matches reference on id, but degrades on horizon and combined ($0.767/0.752$) exactly as the mechanism predicts: with no write gate, the noise ticks integrate, so the signal-to-noise ratio falls like $T^{-1/2}$. **Killing test:** the predicted degradation is a specific function of $T$ and $\sigma$; measuring accuracy at $T\in\{4,8,16\}$ and $\sigma\in\{0.04,0.08\}$ and finding no such scaling falsifies the account, and adding an energy gate to E should restore the horizon panels. This is the cheapest untested cross-prediction in the set.

**F and H (controls for C1(ii) and C2).** F stores both latents in one additive state and reads a quadratic form; it holds up on the iid panels ($0.827$-$0.880$) but sits at $0.544$ on heldout, i.e. at the gru20 level. H keeps the two-slot latch but replaces the product by a linear readout and is the worst route in the entire set on heldout ($0.424$, below chance because the linear rule learned on 24 cells is anti-correlated on the 8 held-out ones). Together these support the C2 direction: with a latched but purely concatenated state, heldout is not merely capped, it is actively misled; a product term is what changes the sign of the outcome.

**G and I (the two axes, isolated).** G supplies the correct gate for free and keeps everything else from route A: $0.979/0.958/0.978/0.956$ and heldout $0.952$. I supplies the product readout and correct slot separation but replaces marginal stability by a nilpotent shift: heldout $0.954$ at $T=4$, and exactly $0.502/0.497$ on the $T=8$ panels because the signal has shifted out of the buffer. The two controls separate the contract's two bottlenecks cleanly and quantitatively: composition (heldout) is settled by the bilinear readout over separated latents, while horizon is settled by the retention mechanism, and neither substitutes for the other. G additionally shows that the whole gap between route A and near-perfect accuracy is attributable to gate credit assignment alone, at fixed state budget and fixed readout family.

**J (factored multiplicative RNN).** The product is formed inside the transition in a learned diagonal factor basis rather than by circulant convolution or an outer product. It is the largest route (478 params) and among the weakest; the tanh on the candidate makes the written value input-dependent in a way that the closed interval cannot undo, and the linear gate defect of A applies unchanged. **Killing test:** substituting the A-SAL gate into J should reproduce L-like numbers if the factor basis is equivalent to the circulant one; if it does not, the specific binding algebra matters and that is a testable structural distinction.

## 8. What is still missing

- No route here has been checked at $T>8$; every claim about structural marginal stability predicts flat accuracy in $T$ up to the noise term, and that prediction is untested.
- The A-SAL gate as implemented reads a raw input energy. A version that is a surprise or prediction-error functional (energy of $x_t$ relative to a running expectation) would carry the same evenness property without the fixed threshold scale, and is the obvious generalisation to a domain where the signal amplitude is not known; it was not built or tested in this lane.
- The observed bimodality of K and N across seeds is not characterised; whether it is a two-basin landscape or a slow-mode optimisation artefact is undetermined, and 6-8 seeds cannot separate them.
- The C2 direction is supported numerically by H and F but no lower bound is proved here; that is the math lane's item.

## 9. Reproduction

```
.venv/Scripts/python.exe _workspace/ce/agi-v14-binding-design-20260812/artifacts/v14_route_toy.py \
  --seeds 9000,9001,9002,9003,9004,9005 --epochs 200 \
  --output _workspace/ce/agi-v14-binding-design-20260812/artifacts/v14_routes_main.json
```

Torch 2.11.0+cpu, numpy 2.3.5, single thread, no GPU. The script imports the frozen generator `reality_stone.clarus.local_cloud_v13_benchmark` and the frozen 20-channel encoding only; no repo file was modified by this lane.

Status: COMPLETE
