# Validation

Status: COMPLETE

All commands run from the repo root with the repo-provided executor (`uv run`,
project `.venv`); no new virtual environment created.

## 1. Unit/regression tests

Command:

    uv run pytest tests/test_local_cloud_v13.py -q

Result (final, after exact-sigma cap): `25 passed in 5.43s`
(17 pre-existing v13 tests + 8 new v13b tests; one intermediate failure
during development is recorded below.)

Command (related regression, no file in this set imports v13b but the shared
runner/module changed):

    uv run pytest tests/test_local_cloud_v13.py tests/test_learnable_small_gain_local_cloud.py \
      tests/test_local_cloud_benchmark.py tests/test_local_cloud_confirmation_runner.py \
      tests/test_local_cloud_kernel.py tests/test_local_cloud_ood_benchmark.py -q

Result: `73 passed in 6.25s` — no regressions.

Intermediate failure (recorded, not hidden): the first version of
`test_v13b_lipschitz_report_is_structural_but_uncertified` asserted
`structural_bound <= 3.0 + 1e-6` and failed with
`3.0057982206344604 <= 3.000001` under the power-iteration cap (estimate
slack). This, plus the 16-seed observation of capped sigma up to 1.0563,
motivated replacing power iteration with the exact 4x4 top singular value;
the final test asserts `<= 3.0 + 1e-4` (float32 slack only) and the trained
sigma test asserts `<= 1 + 1e-5`.

## 2. Runner smoke (v13b path)

Command:

    uv run python examples/agi/local_cloud_v13_run.py --model v13b --seeds 61,62 \
      --train-episodes 96 --evaluation-episodes 32 --epochs 5 --output <scratchpad>/v13b_smoke.json

Result: exit 0, `"overall": "STOP"` (expected at 5 epochs), G4 integrity true.

## 3. 16-seed development run (registered protocol, unchanged from v13 dev16)

Command:

    uv run python examples/agi/local_cloud_v13_run.py --model v13b \
      --seeds 9000,9001,...,9015 --train-episodes 192 --evaluation-episodes 256 \
      --epochs 200 --output artifacts/agi/local_cloud_v13b_dev16.json

Original gate output (final run, exact cap):

    overall: STOP
    gates: {"G1_v13_within_5pct_of_gru20_all_panels": false,
            "G2_v13_beats_elman3_paired_lcb_all_panels": false,
            "G3_v13_heldout_accuracy_at_least_0_90": false,
            "G4_integrity": true}
    g1_per_panel: all five false
    g2 paired mean [2.5%, 97.5%] vs elman3:
      id       -0.0010 [-0.0239, +0.0237]
      noise    +0.0054 [-0.0229, +0.0342]
      horizon  -0.0815 [-0.1707, +0.0139]
      combined -0.0684 [-0.1499, +0.0139]
      heldout  -0.0447 [-0.1270, +0.0322]
    integrity: all zero (G4 true); param count 197; 
      sigma_* mean=max=1.000000 on all six matrices; structural_bound = 3.000000

Panel means (v13b final) vs reused v13 dev16 baselines (baselines reproduce
the prior JSON bit-exactly, same seeds/data — verified programmatically):

    id 0.7815 | noise 0.7588 | horizon 0.5618 | combined 0.5513 | heldout 0.3220

Interim power-iteration run (cap violated up to sigma 1.0563; preserved at
`artifacts/local_cloud_v13b_dev16_interim_power_iteration.json` in this
run dir): id 0.8147 | noise 0.7925 | horizon 0.5874 | combined 0.5784 |
heldout 0.3362 — also STOP on G1/G2/G3.

## Verdict

Implementation validated (tests + determinism + baseline bit-reproduction +
exact cap holding). Scientific outcome: **STOP** — v13b fixes the v13
stability blow-up and improves horizon/combined over v13, but fails all
registered accuracy gates; no closure or status promotion.

---

# Round 3 addendum (2026-08-12): balanced split + v13c (spectral_cap=1.25)

Status: COMPLETE

All commands from the repo root with the repo-provided executor (`uv run`,
project `.venv`); no new virtual environment created.

## 1. Unit/regression tests

Command:

    uv run pytest tests/test_local_cloud_v13.py -q

Result: `33 passed in 8.78s` (25 pre-existing + 8 new; includes the
identifiability test: logistic regression on each context's 6 train cells
predicts both balanced held-out cells exactly, for all 21 tested seeds
including the 16 run seeds 9000-9015).

Command (related regression):

    uv run pytest tests/test_local_cloud_v13.py tests/test_learnable_small_gain_local_cloud.py \
      tests/test_local_cloud_benchmark.py tests/test_local_cloud_confirmation_runner.py \
      tests/test_local_cloud_kernel.py tests/test_local_cloud_ood_benchmark.py -q

Result: `81 passed in 9.72s` — no regressions.

## 2. Frozen-path bit-reproduction

Single-seed check that the compositional path and default spectral_cap are
byte-unchanged: `evaluate_seed(9000, ..., variant="v13b",
split="compositional")` vs `artifacts/agi/local_cloud_v13b_dev16.json` seed
row 9000 — `panel match: True`, `hash match: True`, `holdout match: True`.

## 3. Smoke (multi-variant, balanced)

    uv run python examples/agi/local_cloud_v13_run.py --model v13,v13b,v13c --split balanced \
      --seeds 61,62 --train-episodes 96 --evaluation-episodes 32 --epochs 5 --output <scratchpad>/v13c_balanced_smoke.json

Exit 0; per-variant gate blocks emitted; G4 true for all three.

## 4. 16-seed balanced development run (registered protocol otherwise unchanged)

    uv run python examples/agi/local_cloud_v13_run.py --model v13,v13b,v13c --split balanced \
      --seeds 9000,...,9015 --train-episodes 192 --evaluation-episodes 256 --epochs 200 \
      --output artifacts/agi/local_cloud_v13c_balanced_dev16.json

Original gate output:

    overall: {'v13': 'STOP', 'v13b': 'STOP', 'v13c': 'STOP'}
    all three variants: G1 false, G2 false, G3 false, G4 true

Panel means (balanced split, 16 seeds):

    panel        v13     v13b     v13c      v10   elman3  elman20    gru20
    id        0.8643   0.8022   0.8206   0.6355   0.7969   0.8567   0.8896
    noise     0.8345   0.7634   0.7856   0.5730   0.7678   0.8530   0.8887
    horizon   0.5527   0.5278   0.5347   0.5334   0.6855   0.7439   0.8894
    combined  0.5513   0.5229   0.5237   0.5178   0.6711   0.7346   0.8896
    heldout   0.5894   0.5750   0.5637   0.5815   0.4812   0.4180   0.5515

G1 per panel: heldout true for all three variants (v13 0.5894 vs gru20
0.5515: candidates actually beat gru20 there); id true for v13 only;
noise/horizon/combined false for all.

G2 paired mean [2.5%, 97.5%] vs elman3:

    v13 : id +0.0674 [+0.0437,+0.0916] | noise +0.0667 [+0.0371,+0.0970]
          horizon -0.1328 [-0.2178,-0.0454] | combined -0.1199 [-0.2007,-0.0386]
          heldout +0.1082 [+0.0583,+0.1612]
    v13b: id +0.0054 [-0.0252,+0.0361] | noise -0.0044 [-0.0400,+0.0317]
          horizon -0.1577 [-0.2363,-0.0657] | combined -0.1482 [-0.2280,-0.0617]
          heldout +0.0938 [+0.0271,+0.1582]
    v13c: id +0.0237 [-0.0027,+0.0518] | noise +0.0178 [-0.0144,+0.0513]
          horizon -0.1509 [-0.2313,-0.0639] | combined -0.1475 [-0.2253,-0.0615]
          heldout +0.0825 [+0.0215,+0.1431]

Integrity: all zero for every variant (distinct hashes, nonfinite 0).
Observed caps: v13b sigma exactly 1.000000 (all six matrices, all seeds,
still binding); v13c sigma exactly 1.250000 (the relaxed cap is *also*
binding), structural bound 3.75.

## 5. Effects of the two corrections (honest reading)

- Split correction (compositional -> balanced), heldout panel:
  v13 0.3108 -> 0.5894, v13b 0.3220 -> 0.5750, v10 0.3499 -> 0.5815,
  elman3 0.3667 -> 0.4812, elman20 0.4382 -> 0.4180, gru20 0.4270 -> 0.5515.
  The below-chance compositional numbers are confirmed as the
  anti-identifiability artifact (ceiling 0.25 for the undetermined 3/4 of
  contexts). But with a fair, ideally-identifiable split (linear ceiling
  1.0), *no* model exceeds 0.59 heldout — G3 (>= 0.90) fails by a wide
  margin for every candidate and every baseline including gru20. The failure
  to compose context x bits is real, not a split artifact.
- sigma relaxation (v13c vs v13b, same data/seeds): id +0.0184,
  noise +0.0222, horizon +0.0069, combined +0.0008, heldout -0.0113.
  Direction matches the interim leaky-cap observation on id/noise, but the
  effect is small, the 1.25 cap is again binding, and it does not touch the
  dominant horizon/combined gap to gru20 (~ -0.35); v13c stays below v13
  (uncapped) on id/noise as well.

## Verdict (round 3)

Implementation validated (tests, bit-reproduction of the frozen paths,
determinism, integrity). Scientific outcome: **STOP for v13, v13b, v13c** on
the now-fair G3 and on G1/G2. The split fix removed the artifact and roughly
doubled heldout accuracy for the candidates, and they now beat all baselines
on heldout — but 0.59 << 0.90, and the T=8 horizon/combined collapse vs
gru20 persists across every cap setting tried. The remaining defect is not
the spectral constraint level; no closure, no status promotion.
