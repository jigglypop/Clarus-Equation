# BA-TR14 validation

Focused command:

```text
.codex\hooks\python.cmd pytest tests/test_runtime_experience_delayed_binding.py -q -p no:cacheprovider
```

Result: `3 passed` in `3.75 s`.

Development command:

```text
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_experience_delayed_binding_benchmark --input _workspace\ce\brainruntime-local-stochastic-binding-20260822\artifacts\development-results.json --output _workspace\ce\brainruntime-experience-delayed-binding-20260822\artifacts\development-results.json
```

The benchmark intentionally returned exit code 1 because the frozen all-seed
gate failed. The result artifact was written successfully.

Development result:

- Status: `EXPERIENCE_DELAYED_BINDING_STOP`
- Pass: `14/16`
- Accuracy distribution: fourteen `1.0`, one `.75`, one `.50`
- Mean learned accuracy: `.953125`
- Strongest control accuracy: `.25`
- Raw/install maximum error: `0`
- Installed block Frobenius norm: `1.603962779045105` for every seed
- Minimum hidden Gram margin: `5.642228476396132e-7`
- Minimum cue-only target margin: `4.4512203203339595e-6`
- Stores remained zero and snapshots order-independent.

Seeds `98503` and `98504` failed the frozen absolute readout threshold of
`1e-5`. Their correct output coordinates were still the unique maxima, but
some margins were `4.45e-6` to `6.55e-6`. The same attenuation caused their
target-shuffle learned-map checks to abstain. No threshold, gain, horizon, or
learning-rate retuning is allowed on these development rows.

