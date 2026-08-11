# Development implementation

Status: COMPLETE

The predecessor kernel remained frozen. This run added deterministic seed-level evaluation,
seed-block bootstrap, factorial interaction, intact-readout lesion scoring, integrity counters,
and a hash-bound one-shot runner.

Registered files:

| file | SHA-256 |
|---|---|
| `local_cloud_kernel.py` | `1F157E0CB9C4B41EFAD3FAEB934AF5502ED39C281F9AD7BEEB64ADEA75756ADD` |
| `local_cloud_benchmark.py` | `BD5F75C82D4CED5C91C631F041C1231951D2BCB0D444D4944CE21B657DF13319` |
| `test_local_cloud_kernel.py` | `9D75CD55AAA2952C143B0DE6BBE58E5DD929A686AB5E041E4C89610E36C8BA34` |
| `test_local_cloud_benchmark.py` | `4BF68A07570C18E0B0F3CED300B83FD4E3D8D83473B55E57D77C5D2EFFE52073` |
| `local_cloud_development_run.py` | `F8CD919E076CE15A5CEF63401C93B9CD9552F13D2E08E924DAD88105983AB6DE` |

Every scored arm has 20 recurrent scalars and the same declared upper bound of 76 scalar
multiplies per tick. Dynamic coefficients are zeroed by ablation rather than replaced with arm-
specific learned modules. Effective ridge degrees of freedom and coefficient norms are reported
instead of being called equal capacity.
