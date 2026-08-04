# EulerCE 2-bit attention — refined formulation

Cleaned-up math, made consistent with the implementation
(`reality_stone/clarus/ce_euler.py`) and with the measured results
(`experiments/RESULTS_ood_length.md`). The refinement sharpens the old doc
claim: the length-OOD tier is governed **entirely by the decay bit**, not by
rotation.

## 1. Head as a 2-bit operator

Each head $h$ commits to a bit pair $(\pi_h, e_h)\in\{0,1\}^2$:
$\pi_h$ turns rotation on, $e_h$ turns distance decay on. These are the
operational residue of $\{e,\pi,i\}$: $\{\pi,i\}\Rightarrow$ rotation
generator $e^{i\theta}$, $\{e\}\Rightarrow$ exponential decay.

**Rotation.** With per-pair frequencies $\omega_r = b^{-2r/d}$
($r=0,\dots,d/2-1$, base $b=10^4$), define the rotation of a vector $u$ at
position $m$:
$$
\rho_m(u) \;=\; R(\pi_h\, m\,\omega)\,u,\qquad
R(\phi)=\bigoplus_{r}\begin{pmatrix}\cos\phi_r&-\sin\phi_r\\ \sin\phi_r&\cos\phi_r\end{pmatrix}.
$$
$\pi_h=0$ collapses $R$ to the identity (no positional rotation).

**Decay.** A causal, distance-linear logit penalty with per-head length
$\xi_h>0$ (init $\xi_h=N_{\text{train}}/8$, learnable):
$$
\beta_h(i,j) \;=\; -\,e_h\,\frac{|i-j|}{\xi_h}.
$$
$e_h=0$ removes it. This matches the code mask
$m[h,i,j]=-|i-j|\,e_h/\xi_h$ for $j\le i$, else $-\infty$.

**Unified score.** For query $i$, key $j\le i$:
$$
\boxed{\;\ell_h(i,j)=\frac{\langle \rho_i(q_i),\,\rho_j(k_j)\rangle}{\sqrt{d}}
\;-\;e_h\,\frac{i-j}{\xi_h}\;},\qquad
a_h(i,\cdot)=\operatorname{softmax}_{j\le i}\ell_h(i,j).
$$

## 2. The taxonomy (and its literature anchors)

| $(\pi_h,e_h)$ | rotation | decay | method | length tier (measured) |
|:---:|:---:|:---:|:---|:---:|
| $(0,0)$ | – | – | NoPE [Kazemnejad'23] | **T2** (+32%) |
| $(0,1)$ | – | ✓ | ALiBi [Press'22] | **T1** (−9%) |
| $(1,0)$ | ✓ | – | RoPE [Su'21] | **T2** (+505%) |
| $(1,1)$ | ✓ | ✓ | xPos [Sun'23] / EulerCE | **T1** (−9%) |

(3-seed, train 64 → eval 2048, §RESULTS_ood_length.)

## 3. Central lemma — the decay bit alone governs length-OOD

**Claim.** Under bounded content logits, $e_h=1$ gives a receptive field
whose size is independent of sequence length $N$; $e_h=0$ does not.

**Proof sketch (localization bound).** Assume $|\langle\rho_i(q_i),\rho_j(k_j)\rangle|/\sqrt d \le B$
(true for norm-bounded $q,k$; rotation is orthogonal so it preserves the bound).
With $e_h=1$, for distance $\Delta=i-j$,
$$
a_h(i,j)\;=\;\frac{e^{\ell_h(i,j)}}{\sum_{j'\le i}e^{\ell_h(i,j')}}
\;\le\; \frac{e^{B-\Delta/\xi_h}}{e^{\ell_h(i,i)}}
\;\le\; e^{2B}\,e^{-\Delta/\xi_h}.
$$
Hence the attention mass beyond a window $D$ is
$$
\sum_{\Delta>D} a_h(i,j)\;\le\;\frac{e^{2B}}{1-e^{-1/\xi_h}}\;e^{-D/\xi_h},
$$
a bound **independent of $N$**. Choosing $D=\xi_h\log(e^{2B}/\varepsilon)$
caps the out-of-window mass at $\varepsilon$ for every $N$ — train/test
length invariance.

With $e_h=0$ the logit has no $\Delta$ term, so the bound fails: the softmax
ranges over all $N$ keys and far-token mass is $\Theta((N-D)/N)$, which does
**not** vanish as $N$ grows past $N_{\text{train}}$. $\square$

**Corollary (severity within T2).** Among $e_h=0$ heads, rotation makes it
much worse: $\text{rope}(\pi{=}1)\gg\text{nope}(\pi{=}0)$, measured
$+505\%\gg+32\%$.

*Mechanism — tested and corrected.* A natural guess is rotary-phase aliasing
at unseen positions $j>N_{\text{train}}$. We tested it directly
(`experiments/aliasing_probe.py`): position interpolation, which squeezes
all eval phases back into $[0,N_{\text{train}})$, removes only **11%** of
RoPE's degradation ($+629\%\to+559\%$). So phase-range aliasing is **not**
the dominant cause. The data instead reinforces the lemma: the operative
variable is localization. Without the decay term the softmax still ranges
over all $N$ content scores, and rotation injects position-dependent
perturbations into those scores that, lacking any far-token suppression,
corrupt more than NoPE's positionless dilution. **Position correction alone
(NTK/YaRN-style) does not rescue a decay-free head; only the decay bit
does.** (Caveat: in this implementation `pos` drives both rotation and
decay, so interpolating an ALiBi head also collapses its distance penalty —
the ALiBi-interp number is an artifact, not evidence about ALiBi.)

## 4. Refinement vs. the old doc

The old `19_OOD_Generalization.md` framed it as "pure rotation is the lone
Tier-2 type, so 3 of 4 head-types are usable (capacity $\log_2 3$)". The
measurement does not support that: **NoPE ($e{=}0$) also fails.** The clean,
data-consistent statement is the factorization

$$
\text{length-robust}(h)\iff e_h=1,
$$

i.e. the rotation bit is **orthogonal** to extrapolation. Effective
length-robust capacity is therefore **1 bit (the decay bit)**, not
$\log_2 3$. Rotation still adds in-distribution expressivity (relative-phase
matching), which is why $(1,1)$ xPos is the natural default: keep rotation
for expressivity, keep decay for extrapolation.
