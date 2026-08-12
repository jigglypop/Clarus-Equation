# Implementation

Status: COMPLETE

## Scope / authorization

- Parent task: add `GatedLocalCloudV13B` (strict convex-combination transition +
  spectral cap sigma <= 1.0 on all recurrent-path matrices), `--model v13b`
  runner option, v13b tests, and a 16-seed dev run. No modification of the
  existing `GatedLocalCloudV13` class; runner logic unchanged except candidate
  registration/threading. No prior 20-audit exists for this line (v13 was also
  implemented without one); the parent task message defines the approved scope
  and the marginal-stability relaxation carries over the same explicit user
  authorization recorded for v13.

## Changes

1. `reality_stone/python/reality_stone/clarus/gated_local_cloud_v13.py`
   - New class `GatedLocalCloudV13B` (existing `GatedLocalCloudV13` untouched):
     - Transition is a strict convex combination `h' = (1-g) (*) h + g (*) h~`,
       `g = sigmoid(W_g x + b_g)` (input-driven, as in v13),
       `h~ = tanh(W_in x + W_rec h + cross + U(h (*) c))`.
     - All six recurrent-path matrices (local/cloud rec, both cross matrices,
       both interaction matrices U) are spectrally capped:
       `W_eff = W / max(sigma(W), 1)`.
     - Cap mechanism: initially power iteration (8 steps, persistent u
       buffers) per the task instruction; the first 16-seed run showed the
       estimate underestimating sigma by up to 5.6% after training (capped
       norm up to 1.0563 on one seed's cloud_rec), violating the intended
       sigma <= 1.0. Replaced by the exact top singular value (4x4 svdvals,
       no_grad, used as constant divisor per standard SN convention); the cap
       now holds exactly (observed max sigma = 1.000000 across all 16 seeds).
       The interim run's JSON is preserved in
       `artifacts/local_cloud_v13b_dev16_interim_power_iteration.json`.
     - `lipschitz_report()`: exact sigma of each effective matrix,
       `structural_bound_*` fields (coarse per-branch candidate bound
       sigma(rec)+sigma(cross)+sigma(U) <= 3, valid because the convex
       combination keeps |state|_inf <= 1 from zero init), `certified: False`
       retained.
     - Parameter count 197 vs v13's 205 (within +-30%; v13's 8 retention
       params are replaced by the convex combination).
   - New `train_gated_v13b` (identical regime: Adam lr 0.01, wd 1e-4,
     clip 1.0, full-batch BCE, deterministic).
2. `examples/agi/local_cloud_v13_run.py`
   - `VARIANTS` registry + `--model {v13,v13b}` (default v13); `variant`
     threaded through `evaluate_seed`/`evaluate_development` (candidate row
     key, G1/G2/G3 references, model-count integrity check). Gate key names
     unchanged (registered names). `learned_retention_summary` now summarizes
     whatever numeric keys the variant's lipschitz report exposes (v13 and
     v13b expose different observational keys). New exploration-log entries
     for the v13b design and the power-iteration -> exact-sigma replacement.
     Default v13 behavior is unchanged (verified by the pre-existing tests).
3. `tests/test_local_cloud_v13.py`
   - 8 new tests: g~=0 freezes state exactly (h'=h), g~=1 yields the pure
     candidate (closed form from zero state), |state|_inf <= 1 invariant,
     spectral cap on forced expansion (5*I) and after training (<= 1+1e-5),
     structural-but-uncertified report shape and bound range [1, 3],
     parameter budget (197, +-30% of 205), training determinism (model_hash),
     v13b end-to-end through the development harness.

## Invariants checked

- reality_stone/clarus: minimal-form limit honored — at g -> 0 the transition
  reduces to identity (h'=h), and from zero state with g -> 1 to a plain
  bounded readout of tanh(W_in x); no canonical state-dimension promotion
  (same 20-scalar local/cloud budget as V10-V13); no F1-F4 bypass; no STDP
  involvement; CodeMap consulted (no v13 entries exist; none added — out of
  approved scope).
- physics gates: no threshold or task changed after seeing results
  (G1-G4 identical to the v13 dev16 registration; same seeds 9000-9015, same
  episode counts/epochs); the failing result is reported as STOP; the interim
  power-iteration run (which happened to score higher on every panel) is
  preserved and reported, not silently discarded, and was replaced only
  because it violated the specified sigma <= 1.0 cap, not because of its
  accuracy.
- No target/label/oracle signal enters the transition; readout at final tick
  only.

## Result summary (honest)

16-seed dev run (`artifacts/agi/local_cloud_v13b_dev16.json`): **STOP**.
G1 false (all panels), G2 false (id/noise CIs straddle 0; horizon/combined/
heldout means negative vs elman3), G3 false (heldout 0.322), G4 true.
Baselines reproduce the v13 dev16 values bit-exactly (same seeds/data).

| panel    | v13b (exact cap) | v13b (interim PI) | v13    | v10    | elman3 | elman20 | gru20  |
|----------|------------------|-------------------|--------|--------|--------|---------|--------|
| id       | 0.7815           | 0.8147            | 0.8164 | 0.6138 | 0.7825 | 0.8601  | 0.8567 |
| noise    | 0.7588           | 0.7925            | 0.8022 | 0.5583 | 0.7534 | 0.8599  | 0.8569 |
| horizon  | 0.5618           | 0.5874            | 0.5286 | 0.5437 | 0.6433 | 0.6064  | 0.8586 |
| combined | 0.5513           | 0.5784            | 0.5288 | 0.5234 | 0.6196 | 0.6023  | 0.8567 |
| heldout  | 0.3220           | 0.3362            | 0.3108 | 0.3499 | 0.3667 | 0.4382  | 0.4270 |

Reading: the structural cap eliminated the v13 Lipschitz blow-up (observed
loose bound 7-9 -> structural bound exactly 3.0, all sigma = 1.0) and
partially recovered horizon/combined (+0.03-0.06 over v13), but v13b still
trails elman3 there (~ -0.06 to -0.08) and trails gru20 everywhere except
being level on nothing; id/noise regressed ~0.03-0.04 vs v13. The interim
(slightly leaky) cap scored higher on every panel than the exact cap,
suggesting the binding sigma = 1.0 constraint costs capacity; that
observation is recorded here as-is, hypothesis level only. Convex
combination + spectral caps fixed stability but did not close the gap to
GRU-20; the diagnosed defect is not fully explained by transition gain.

---

# Round 3 addendum (2026-08-12): balanced split + spectral_cap relaxation (v13c)

Status: COMPLETE

## Scope / authorization

Parent task (round 3) with two confirmed diagnoses: (1) math-verified —
the compositional complement-pair holdout is anti-generalizing (labels
undetermined by train with probability 3/4, in which case every
margin/logistic/NN learner outputs the opposite label; ideal-learner ceiling
0.25); (2) observed — the exact sigma = 1.0 cap was binding (leaky cap ~1.056
scored higher on every panel), and the user explicitly authorized relaxing
the constraint. Approved scope: `split="balanced"` benchmark mode,
`spectral_cap` parameter on `GatedLocalCloudV13B` (default 1.0, behavior
unchanged), `v13c = v13b(spectral_cap=1.25)` registration, `--split` runner
option, validation tests, and a 16-seed balanced run.

## Changes

1. `reality_stone/python/reality_stone/clarus/local_cloud_v13_benchmark.py`
   - `cell_label(context, index)`: ground-truth label of a cell under the
     frozen V10 rule (public helper for the identifiability tests).
   - `holdout_cells_balanced(seed)`: per context one +1-labelled and one
     -1-labelled cell, never a bitwise-complement pair; seed-deterministic
     (rng tag 778, distinct from compositional's 777).
   - `generate_episodes_v2(..., condition_split="balanced")`: same 24/8
     train/eval cell mechanics as compositional (rng tags 131/241 vs 130/240);
     "iid" and "compositional" code paths untouched (delegation and tags
     identical).
   - `panel_configs_v13(..., heldout_split={"compositional","balanced"})`,
     default "compositional" (original panels byte-for-byte).
2. `reality_stone/python/reality_stone/clarus/gated_local_cloud_v13.py`
   - `GatedLocalCloudV13B(spectral_cap=1.0)`: cap divisor becomes
     max(sigma/cap, 1), so sigma(W_eff) <= spectral_cap exactly; at the
     default 1.0 this equals the original max(sigma, 1) bit-for-bit
     (verified: identical model hash, and seed-9000 dev16 row reproduced
     bit-exactly). `lipschitz_report` gains `"spectral_cap"`;
     `train_gated_v13b` gains the pass-through kwarg. Still
     `certified: False`; structural bound scales to 3*cap.
3. `examples/agi/local_cloud_v13_run.py`
   - `VARIANTS["v13c"] = functools.partial(train_gated_v13b, spectral_cap=1.25)`.
   - `--split {compositional,balanced}` (default compositional) threaded to
     candidate training, the heldout panel, and the recorded holdout cells.
   - `--model` now also accepts a comma-separated list; each candidate gets
     its own registered G1-G4 block (`per_variant`), trained with the same
     seed (seed*100+50) it would get solo, so solo runs reproduce
     bit-for-bit. Single-variant output schema unchanged (legacy flat keys,
     `v13_ledger`). Two new exploration-log entries record the balanced-split
     rationale and the cap value 1.25 (fixed before the run).
4. `tests/test_local_cloud_v13.py`: 8 new tests — balanced holdout
   determinism/label-balance/non-complement (21 seeds incl. 9000-9015),
   24/8 disjointness + eval label balance, logistic-regression
   identifiability of every held-out pair from the 6 train cells (all run
   seeds, all contexts), heldout_split selection/validation, spectral_cap
   capping at 1.25 + pass-through of sub-cap matrices + rejection of 0/inf,
   default-cap bit-identity vs original and difference vs cap 1.25, v13c
   registry check, multi-variant balanced harness end-to-end.

## Invariants checked

- Frozen files untouched (`local_cloud_kernel.py`, `local_cloud_benchmark.py`,
  `local_cloud_ood_benchmark.py`, `learnable_small_gain_local_cloud.py`,
  docs 29/30, v10-v12 workspaces) — confirmed via git status.
- "compositional" mode and the existing dev16 artifacts preserved unmodified
  (negative results stay on record); compositional path verified bit-exact
  (seed-9000 v13b row: panel accuracies, state hash, holdout cells all match
  `artifacts/agi/local_cloud_v13b_dev16.json`).
- No threshold changed after seeing results: G1/G2/G3/G4 definitions, seeds
  9000-9015, episode counts, epochs identical to the registered dev16
  protocol; cap value 1.25 and the balanced split were fixed before the run.
- No target/label/oracle signal enters any transition; minimal-form limits
  of v13b unchanged by the cap (cap=1.0 default is bit-identical); no
  canonical state-dimension promotion (same 20-scalar budget).

## Result summary (honest)

`artifacts/agi/local_cloud_v13c_balanced_dev16.json`: **STOP for all three
candidates** (v13, v13b, v13c). G4 true everywhere; G1/G2/G3 false for all.
Details in the round-3 addendum of 31-validation.md.
