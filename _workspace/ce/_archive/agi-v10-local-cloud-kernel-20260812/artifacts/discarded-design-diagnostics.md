# Discarded design diagnostics

Status: NON-EVIDENCE

These target-aware engineering probes used disposable seeds and may never be reported as
development or confirmation evidence.

| probe | full | local only | cloud only | no memory | disposition |
|---|---:|---:|---:|---:|---|
| additive, 8-step, 64/64 | 0.515625 | 0.578125 | 0.406250 | 0.359375 | rejected: no interaction and unfair semantic mixture |
| bilinear, 8-step, 512/512 | 0.468750 | 0.490234 | 0.484375 | 0.464844 | rejected: signal decayed before readout |
| bilinear, 4-step, biased context, 512/512 | 0.636719 | 0.619141 | 0.511719 | 0.476563 | rejected: context matrix had a marginal local-bit shortcut |
| balanced context, equal local retention, 512/512 | 0.564453 | 0.519531 | 0.476563 | 0.490234 | rejected: weak interaction |
| frozen-candidate structure, 512/512 | 0.650391 | 0.482422 | 0.480469 | 0.496094 | design diagnostic only; all seeds burned |

The sequence documents why the old approach was not simply rerun. It also establishes
look-elsewhere exposure: gains, horizon, context coding, and task form were viewed. A future
development registration must freeze them and use entirely fresh seed roles. Confirmation
cannot reuse any seed from this file.
