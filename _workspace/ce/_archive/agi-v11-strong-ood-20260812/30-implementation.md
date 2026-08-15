# V11 implementation

Status: COMPLETE

Added a raw-sequence evaluator that feeds identical 20-scalar observations to V10, Elman-20,
GRU-20, and Elman-3. Learned comparators use deterministic full-batch Adam under a frozen 100-
epoch schedule. ID, noise, horizon, and combined panels are generated from the same label rule.

The compute ledger is explicit: V10 `76`, Elman-3 `72`, Elman-20 `820`, GRU-20 `2460` estimated
multiplies per tick. Parameter counts are respectively `30`, `79`, `861`, and `2541`.

Pre-run validation: 51 focused/dimensionless tests, Ruff check, and format check passed.
