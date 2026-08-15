# AGI V15 scored validation harness

Status: COMPLETE

## 1. Scope

No product implementation was changed. The system under test remained frozen at
the contract hash. This run added only reproducible research artifacts:

- `artifacts/verify_scored_math.py`: F1--F5 killing fixtures;
- `artifacts/run_scored_benchmark.py`: held-out finite correctness,
  coordinate/permutation metamorphism, positive-scale termination, oracle
  navigation utility, and explicit A1--A4 aggregation;
- `artifacts/scored-results.json`: first scored stdout captured without removing
  failures;
- `artifacts/verify_adversarial_numeric.py` and
  `artifacts/adversarial-numeric-results.json`: unique-path scale and extreme
  finite-input probes added without modifying the SUT.

## 2. Independent calculations

- Edge weights are reimplemented directly from the endpoint-average quadratic
  expression.
- Reference all-pairs costs use vectorized Floyd--Warshall rather than the SUT's
  Dijkstra implementation.
- Oracle utility evaluates both branch costs independently and then scores the
  SUT and identity-selected branches in the fixed environment cost.
- The positive-scale call runs in a daemon thread with a one-second boundary so
  a predecessor cycle is recorded as a failure rather than hanging the whole
  validation process.
- Random generators are local `numpy.random.Generator` instances keyed only by
  the preregistered integer seed.

## 3. Explicit limitations

The scorer limitations recorded in `20-audit.md` are part of the implementation
contract. In particular, affine transport generation and randomized tie
tolerance are not fully independent of the SUT convention, the scale direct
path is not uniquely optimal, and the oracle utility comparison is
information-asymmetric. None changes the observed scale nontermination or the
A1--A4 score.

The SUT defect was intentionally not fixed because the user asked for an actual
score. Repair and confirmatory rerun must be a new run with a new frozen hash.
