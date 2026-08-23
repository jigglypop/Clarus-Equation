# BA-TR11 validation

Focused test:

`.codex\hooks\python.cmd pytest tests/test_runtime_learned_metric_curvature.py -q -p no:cacheprovider`

Result: `4 passed` in 3.60 s.

Frozen BA-TR10 development-weight probe: 16/16 passed and status
`CURVATURE_MEMORY_IDENTITY_REJECTED`. Maximum analytic-vs-central-FD Jacobian
error was `7.8464e-11`. The largest hidden-row-permutation curvature residual
over all seeds and points was `5.8981e-17`. Every learned code produced a
nonzero nonlinear curvature signature; the minimum across-seed maximum
absolute curvature was `2.1172e-5`. A general hidden rotation kept the origin
metric fixed while changing nonlinear curvature by at least `.0418386`.

No ridge, output, decoder, target, reward, or runtime endpoint was used.

