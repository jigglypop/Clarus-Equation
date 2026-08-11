# V10 local/cloud kernel implementation report

Status: COMPLETE

## Outcome

The failed repeated weak-shell route was not rerun. It was replaced by a new isolated
transition kernel in which four local recurrent states and one shared state update together.
The only exported evidence feature is the resulting twenty-dimensional recurrent state.
There is no label/posterior/HOLD shortcut.

The implementation gate passes:

- bounded bilinear local/cloud interaction;
- global weighted-sup small-gain certificate, `q = 0.9355555555555556 < 0.95`;
- exact deterministic transition composition;
- equal twenty-feature full/local-only/cloud-only/no-memory arms;
- train-only frozen ridge readout;
- cross-cut and local/cloud reset at the actual transition;
- `36 passed`, Ruff check/format pass, dimensionless checker exit `0`.

## What was rejected

Four intermediate task/mechanism variants were rejected because they exposed additive-only
failure, excessive decay, a marginal context shortcut, or weak interaction. Their values and
seeds are preserved in `artifacts/discarded-design-diagnostics.md` as non-evidence. This is the
look-elsewhere ledger, not a scoreboard.

## Exact status

| claim | status |
|---|---|
| declared bounded map is contractive | conditional theorem + unit verification |
| transition composes as a deterministic kernel | implemented and tested |
| local/cloud state causally changes features | implemented lesion tests |
| full model improves over both factorial controls | untested in registered development |
| whole brain is a nested SCC tower | untested biological hypothesis |
| system is AGI | not claimed and not tested |

## Next authorized loop

The next step is a hash-bound, fresh-seed development registration for the frozen task and four
arms. It must report paired confidence intervals, the factorial interaction, lesion losses,
state count, effective ridge degrees of freedom, coefficients, and MAC separately. Confirmation
may open only after the development gate passes. No current runtime default is changed.
