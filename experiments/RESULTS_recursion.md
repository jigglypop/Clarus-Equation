# RecursiveEulerCEBlock — first evidence

First measurements for the weight-tied self-recursion (ClarusCell
fixed-point) claim, which previously had none. Produced by
`experiments/recursion_probe.py` (tiny char-LM, block=64, single seed).

## Results

| config | block params | compute | PPL | mean depth |
|--------|-----:|----:|-----:|-----:|
| rec@1 (baseline) | 49K | 1 | 4.719 | 1.0 |
| rec@2 | 49K | 2 | 4.250 | 2.0 |
| rec@4 | 49K | 4 | 4.096 | 4.0 |
| rec@8 | 49K | 8 | 4.080 | 8.0 |
| **untied-4 (4x params)** | 198K | 4 | **3.642** | 1.0 |
| rec@8 + fp_loss | 49K | 8 | 3.990 | 8.0 |
| rec@8 tol=1e-2 (halting) | 49K | 8 | 4.080 | 8.0 |

## What this shows / doesn't

**Claim (1) recursion buys compute without params — SUPPORTED, with strong
diminishing returns.** rec@8 beats rec@1 by -13.5% PPL at identical
parameters. But almost all of the gain arrives by rec@2 (-9.9%); rec@4->rec@8
adds only ~0.4%. So self-recursion adds effective compute, but saturates
fast (2-4 iterations).

**Claim (2) approaches the untied-depth ceiling — PARTIAL.** rec@8 closes
only 59% of the rec@1 -> untied-4 gap while using 1/4 the parameters. So
weight-tied recursion is parameter-efficient but does NOT replace real depth:
4x params (untied-4 = 3.642) still wins clearly over rec@8 (4.080).

**fixed_point_loss helps slightly.** rec@8 + fp_loss = 3.990 < plain rec@8
4.080. The self-consistency regularizer is a small net positive here.

**tol-halting did NOT trigger.** With tol=1e-2 the mean halting depth stayed
at the max (8.0) — the relative change ||F(h')-F(h)||/||h|| never dropped
below 1e-2, so the loop never converged to a fixed point in this setup. The
"halts at a fixed point" story is not demonstrated; the block keeps moving.

## Honest bottom line
Recursion is real (adds compute at zero params) but modest (saturates by
~2-4 iters, doesn't match added parameters, doesn't converge to a fixed
point under tol). This is a directional first result on a tiny char-LM,
not proof it helps on the workloads the product targets (agent drift,
multi-step reasoning). Those remain untested.
