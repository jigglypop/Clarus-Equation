# Revision 01 — bootstrap support-minimum correction

Date: 2026-08-23  
Trigger: post-development independent implementation audit  
Outcome direction: strictly more conservative; cannot turn a STOP into PASS

The frozen contract requires the relative *minimum* generalized eigenvalue on
the evaluation support for every slice-cluster bootstrap draw. The first
implementation used the within-draw 5th percentile before taking the across-draw
2.5th percentile. That was weaker than the written contract.

The implementation now records

$$
T_b=\min_{z\in\mathcal S}
\frac{\lambda_{\min}(g_{\rm resp}^{(b)}(z),g_{\rm ref})}
{\lambda_{\max}(g_{\rm resp}^{(b)}(z),g_{\rm ref})}
$$

for draw $b$, then applies the already frozen lower 2.5% quantile and
$10^{-4}$ threshold. Seeds (`83201`, `83202`), data, split, response map,
threshold and all predictive results are unchanged. The correction was selected
from contract text, not outcome magnitude. Both strata had already returned
`RANK_UNIDENTIFIED`; this correction can only preserve or strengthen that result.
Confirmation remained unopened before and after the correction.
