# Route decisions

Status: COMPLETE

- Keep: prefix posterior observer, OOD prefix loading adaptation, internal
  correction injection, covariance diagnostics, sparse/dense symmetry.
- Remove from the active core: fast residual mode. It auto-collapsed because
  fold support was 4/8 and its persistence estimate was unstable.
- Do not add: uncertainty output gain. The posterior mean already performs
  measurement-error shrinkage; another gain would recreate V8.
- Defer: regime switching, memory, graph adaptation, hypothesis beams, and
  planning. The base observer has not yet earned a fresh development block.

The next technically coherent revision is not a third residual mode. It is a
rank-one posterior observer with hierarchical episode calibration or a better
measurement-noise model aimed at reducing the three losing folds without
changing the frozen graph or opening new data.
