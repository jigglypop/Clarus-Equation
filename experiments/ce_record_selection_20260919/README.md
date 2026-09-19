# CE-RS1 publication: residual Higgs response and physical records

This additive directory publishes the record-selection follow-up to main commit
`6e473c28099c300045d732719468ba0b8cdbfdf1`.

- `REPORT_ko.md`: canonical publication of definitions, conditional proofs and limits.
- `verify_record_selection.py`: standalone source, including 37 regression tests.
- `verification_rerun.json`: diagnostics from the publication-time rerun.
- `publication.json`: source hashes and distinction between published files and source archives.
- `source/CE_input.txt`: unchanged original common-spectrum source.
- `PREVIOUS_WORK.md`: preceding research, separated from the new rerun.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_record_selection.py --output rerun.json
```

The scalar finite-regulator phase comparison, conditional-record source and
controlled oscillator-history calculations are conditional results. They do not
select an absolute outcome, all constants, the cosmic initial state or the full
Standard Model. No observational fitting or new joint-likelihood success is claimed.
Original archive statements about Git refer to their historical run, not this
publication. Archive hashes are provenance, not evidence that archive bytes were
uploaded. Earlier ground-state computations are not counted as rerun here.
