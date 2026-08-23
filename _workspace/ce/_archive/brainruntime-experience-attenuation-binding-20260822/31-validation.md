# BA-TR15 validation

Focused validation:

```text
.codex\hooks\python.cmd pytest tests/test_runtime_experience_attenuation_binding.py -q -p no:cacheprovider
```

Result: `3 passed` in `4.15 s`.

Fresh input generation passed for calibration `102001` and all 16 development
seeds `102101..102116`. The frozen calibration then passed:

- compensated margin `6.57624623272568e-5`;
- compensation-off margin `9.394371772941668e-6`;
- identity controls at most `.25`;
- raw/install error `0`.

Frozen development result:

- Status: `ATTENUATION_BINDING_DEVELOPMENT_GO`
- Compensated pass count: `16/16`
- Compensation-off pass count: `14/16`
- Minimum compensated margin: `3.990212280768901e-5`
- Minimum compensation-off margin: `2.493882675480563e-6`
- Maximum identity-control accuracy: `.5`
- Target-shuffle reproduced its experienced mapping: `16/16`
- Packet-amplitude shuffle: `14/16`; minimum margin `6.4260784711223096e-6`
- Raw write norm range: `1.603962779045105` to `13.26716423034668`
- Compensation factor range: `1` to `16`
- Edge-cap hits: `0`
- Maximum raw/install error: `0`
- Every per-row timing, support, cutoff, order, and snapshot gate passed.

The packet-shuffle result is informative: it normally preserves identity but
fails exactly two weak-channel rows at the absolute margin. Thus the result is
about reliable amplitude compensation, not discovery of source identity.

