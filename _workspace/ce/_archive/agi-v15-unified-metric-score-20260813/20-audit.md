# AGI V15 scored validation: formal status audit

Status: COMPLETE

Gate: PASS

The gate permits the frozen score and report to be finalized. It does **not**
mean that the finite core or AGI gate passed. The system under test remained at
SHA-256 `0599FC3B212F924424DE0675266881F8F1A6611D880382533708CD55F2529BE4`.

## 1. Claim ledger

| ID | Exact status | Audited claim |
|---|---|---|
| F1 | [theorem] | Joint affine transport preserves local quadratic, finite edge, and fixed-topology path costs. |
| F2 | [no-go theorem] | Fixed-chart spectral clipping is not generally affine covariant. |
| F3 | [no-go theorem] | The implemented static undirected Riemannian cost is direction symmetric. |
| F4 | [no-go theorem] | A reflection-symmetric source-free diamond admits no equivariant singleton goal selector. |
| F5 | [no-go theorem] | Finite endpoint tensors do not identify the intervening continuum metric. |
| held-out score | [numerical result] | Correctness on 256 preregistered ordinary-scale synthetic seeds. |
| coordinate metamorphism | [numerical result] | Readout consistency under generated affine coordinate changes and node permutations. |
| scale fixture | [numerical result: failure] | Public `shortest_path` does not terminate on a positive $10^{-16}$-scale fixture. |
| oracle utility | [numerical result] | With the environment metric supplied directly, the implemented cost objective selects the synthetic optimum. |
| A1--A4 | [incomplete] | No executable autonomous learning, environment loop, delayed credit, or learned compute-matched OOD test exists. |

All score ratios and thresholds are dimensionless. Relative error divides like
units, and normalized regret divides a cost difference by the optimum cost.
Dimensionlessness is necessary for a valid threshold but does not establish
semantic or physical validity.

## 2. Closure findings

F1--F5 each have a complete symbolic argument and a concrete reproduction, so
the mathematical subscore may be reported as `MATH PASS 5/5`. F2--F5 are
limitations, not positive intelligence achievements. No numerical result is
used as a substitute for those proofs.

The positive-scale counterexample is a complete implementation counterexample
to scale-robust termination. Therefore any unconditional claim that the frozen
finite core handles every valid positive finite graph is removed. Ordinary
random-seed success is retained only with its tested scale qualification.

The static metric's direction-symmetry and source-free goal no-go prevent
promotion to irreversible world dynamics or autonomous goal formation. No
parent AGI claim survives merely because oracle cost optimization succeeds.

## 3. Scorer audit

The independent scorer uses Floyd--Warshall for reference shortest costs and an
exhaustive two-branch evaluator for utility. Its connected graph generator,
permutation mapping, and regret calculation were independently checked.

The following limits must accompany the results:

1. Affine coordinates are generated with the SUT's `affine_chart_change`, so the
   randomized metamorphic result validates downstream readout consistency, not
   an independent implementation of tensor transport. F1 supplies the separate
   symbolic proof.
2. Randomized goal comparison deliberately shares the declared numerical tie
   tolerance. It validates that convention, not every possible tie policy.
3. The scale fixture has two exact optimal paths, not a unique direct optimum.
   The runner's direct-path check is unnecessarily strict. The observed result
   is nontermination, so relaxing the path choice cannot change the failure.
4. The runner reports the frozen hash separately rather than combining it into
   `finite_core_go`. The observed hash exactly matches the contract, so this
   aggregation flaw does not invalidate this execution but must be fixed before
   reusing the runner for a different artifact.
5. Affine/permutation tests are coordinate metamorphisms of the same semantic
   instances; they are not task-distribution OOD evidence.
6. The utility environment and V15 arm use the same oracle metric formula. The
   identity baseline has less information. This is objective-alignment and
   privileged-information utility, not a fair learned-model or AGI comparison.

## 4. Gate decision

There is no open P0 against publishing the qualified score. The scorer limits
are explicit and the scale counterexample is retained rather than repaired or
discarded. Therefore the research/report gate is `PASS`.

The result gates remain separate:

- mathematical gate: eligible for 5/5;
- ordinary-scale randomized gate: eligible for PASS;
- positive-scale robustness gate: forced FAIL;
- aggregate finite-core gate: forced STOP;
- oracle objective-alignment gate: eligible for PASS with the oracle qualifier;
- autonomous AGI gate: forced STOP unless A1--A4 all acquire executable scored
  evidence.

