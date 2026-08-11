# Decisive route

## What is preserved

- Loop 8H–8L numbers, seeds, tests, and reports remain checkpoints.
- Their soft support, causal pending order, finite-topology validator, paired
  confidence bounds, and normalization checks remain reusable infrastructure.
- Loop 8K and 8L remain explicit HMM/POMDP comparison arms.

## What is no longer assumed

- no oracle-sized context vector;
- no built-in XOR action law;
- no claim that context-to-action complete bipartite wiring is hierarchical;
- no claim that a hazard posterior is computed by basal ganglia;
- no action-only softmax STN term;
- no claim that reverse vector inhibition is reverse graph message passing.

## Experiment A: topology before recurrence

Use 4 goals and 4 reusable subactions, giving 16 compound actions. Train on 12
balanced pairs and reserve a Latin-square complement of 4 unseen recombinations.
Remove hidden context, switching, feedback recurrence, and hazard inference.

Matched arms:

1. atomic 16-way flat policy;
2. strict tree with a duplicated subaction head under each goal;
3. reconvergent shared-subaction DAG;
4. two-head factorized-flat control;
5. untied or identity-permuted DAG.

The shared-DAG claim passes only if it beats the strict tree and destroyed-DAG
controls on untouched OOD seeds, preserves ID performance, shows correct shared
node lesion locality, and beats the factorized-flat control. Equality to the
factorized control identifies factorization—not DAG topology—as the useful cause.

## Experiment B: STN causal role

Add an explicit HOLD choice and score low- versus high-conflict trials. Compare
full STN-to-HOLD, STN off, and common-offset-only STN. The common-offset arm must
be numerically identical to STN off. Full STN must selectively change HOLD rate
or latency under conflict without arbitrarily reversing action identity.

## Experiment C: local delayed credit

Only after A and B pass, add signed TD eligibility. Shuffle reward, trace timing,
edge responsibility, and TD sign independently. Held-out return must disappear
under the corresponding causal break; mere weight movement is not success.

## Experiment D: restore cortical recurrence

Finally connect the successful PFC–MD checkpoint and train its state by next-event
prediction. Compare intact, reset, and time-shuffled cortical state with the same
frozen BG selector. No hazard grid is tuned. This separates state inference from
selection instead of letting either module solve the other's benchmark.
