# Validation

Status: COMPLETE

## Fresh development block

The locked development runner was executed once on 256 fresh OOD seeds
`79100..79355`.  The block has zero overlap with every V1--V7 registered data
role.  It is development evidence because the repository does not yet provide
immutable registration-before-implementation chronology for a successor.

The candidate is

\[
\widehat Y=P+0.7868543064870357(S-P).
\]

## Model means

All values are mean seed-level normalized H20 path RMSE; lower is better.

| Model | Mean RMSE |
|---|---:|
| parent-anchored sparse shrinkage | `0.548432992` |
| symmetric dense shrinkage | `0.548593999` |
| V5 sparse parent | `0.554138980` |
| zero-bridge shrinkage | `0.558917837` |
| frozen V7 consensus | `0.560358794` |
| frozen V7 no-sparse consensus | `0.576249521` |
| persistence | `0.584028707` |
| stable adaptive dense | `0.631480396` |

## Paired checkpoints

| Comparison | Mean improvement | Paired 95% lower | Seed wins |
|---|---:|---:|---:|
| versus V5 parent | `+0.005705988` | `+0.001105704` | `55.08%` |
| versus persistence | `+0.035595715` | `+0.017896861` | `59.38%` |
| versus zero-bridge shrinkage | `+0.010484845` | `+0.002195676` | `51.95%` |
| versus frozen V7 consensus | `+0.011925803` | `+0.003578677` | `58.59%` |
| versus frozen V7 no-sparse | `+0.027816529` | `+0.014477156` | `61.33%` |

The symmetric dense geometric error ratio was `0.999690077`, with paired
log-ratio interval `[-0.000814014,+0.000194072]`.  Thus sparse and dense are
effectively tied; the result does not establish sparse-specific superiority.

## Stability and integrity

| Component | Maximum pathwise radius | Fraction above `0.98` |
|---|---:|---:|
| sparse | `0.781419962` | `0` |
| symmetric dense | `0.821638175` | `0` |
| zero bridge | `0.781419962` | `0` |
| external adaptive comparator | `1.311462100` | `17.19%` |

- maximum observed state index: `80`;
- future observation reads: `0`;
- nonfinite predictions: `0`;
- historical locked test opened: `false`.

The fresh block therefore reproduces the parent and persistence improvements,
zero-bridge contribution, dense noninferiority, finiteness, leakage, and
retained-component stability checkpoints.  This is a fresh development PASS,
not a confirmatory endpoint or an AGI claim.

