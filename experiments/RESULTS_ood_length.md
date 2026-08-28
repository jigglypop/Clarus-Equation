# Length-OOD reproduction — results

Reproducible backing for the table in `paper/7_AGI/19_OOD_Generalization.md`,
which previously had no runnable script. Produced by
`experiments/ood_length_repro.py`.

## Setup (honest scope)
- tiny char-level LM, d_model=64, 4 heads, 2 layers (~127K params; larger
  than the doc's "~30K" because vocab=220 inflates the embedding)
- corpus: repo `.py` files, 600K chars, char-level, 90/10 split
- train block = 64; eval lengths 64/128/256/512/1024 (up to **16×**, not the
  doc's 32×); **single seed (0)**
- metric: next-char PPL, degradation vs eval@64

## Measured degradation — single seed, eval to 1024 (16x)

| head  |  128  |  256  |  512  |  1024  | tier |
|-------|------:|------:|------:|-------:|:----:|
| alibi | -0.7% | +0.6% | -2.6% |  +0.1% |  T1  |
| xpos  | -1.0% | +1.6% | -1.6% |  +0.6% |  T1  |
| nope  | +10%  | +32%  | +45%  | +60.8% |  T2  |
| rope  | +36%  | +246% | +455% | +663%  |  T2  |

## Scaled — 3 seeds, eval to 2048 (full 32x), mean +/- std

Run: `SEEDS=0,1,2 STEPS=500 EVAL_LENS=64,256,1024,2048 python -m experiments.ood_length_repro`

| head  |  256  |  1024  |  2048 (mean +/- std) | tier |
|-------|------:|-------:|---------------------:|:----:|
| alibi |  -2.3% |  -3.7% |   **-9.1% +/- 5.7**  |  T1  |
| xpos  |  -2.0% |  -3.9% |   **-8.8% +/- 5.4**  |  T1  |
| nope  | +17.9% | +31.7% |   +31.7% +/- 10.2    |  T2  |
| rope  | +179%  | +466%  |  **+505% +/- 41.4**  |  T2  |

Thesis SUPPORTED across 3 seeds at the full 32x: decay-bearing heads are
length-invariant (slightly *improve*), pure rotation is catastrophic. The
NoPE/Tier-1 doc claim is again refuted (Tier 2, consistent across seeds).

## What this confirms / refutes

**CONFIRMED (strongly).** The central thesis holds: heads carrying a decay
bit (ALiBi, xPos) are essentially invariant to 16× length extrapolation
(~0%), while pure rotation (RoPE) is catastrophic (+663%, far worse than the
doc's +47%). The decay bit — not rotation — is what buys length-OOD
robustness. The 2-bit taxonomy's predictive core stands.

**REFUTED (here).** The doc placed NoPE in Tier 1 (+7%, "extrapolates 7x
better than RoPE", citing Kazemnejad 2023). This setup does NOT reproduce
that: NoPE degrades +60.8% (Tier 2). With no positional signal at all, the
tiny model gets worse as context grows. So the doc's specific NoPE claim is
not supported by this reproduction.

## Caveats (do not over-read)
- single seed, tiny model, char-level on a code corpus; directional only.
- 16× not 32×; full reproduction of the doc requires eval@2048 and the doc's
  exact corpus/params, which are not in the repo.
- this neither proves nor claims SOTA; it is a falsifiable check that turns
  the doc's unbacked table into one measured, partially-corrected result.
