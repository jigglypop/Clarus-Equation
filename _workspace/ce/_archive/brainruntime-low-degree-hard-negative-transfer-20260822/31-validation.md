# BA-TR29 validation

Status: `COMPLETE / DEVELOPMENT_GO / CONFIRMATION_SEALED`.

Revision 1 passed all 16 development seeds and all 400 rotating folds.
Maximum quadratic query error was `2.3753434051713483e-15`.  Minimum
quadratic-versus-affine content separation was `0.020168611635363657`.

The affine hard-negative baseline selected its own decoy `400/400` and the
true packet `0/400`.  On the independent nearest-content panels it selected
truth `57/400`; the worst per-seed fraction was `0.24`, below the frozen
`0.25` ceiling.  Minimum hard and nearest binding margins were
`0.010084295757264147` and `0.012199633825917749`.  Minimum runtime route
separation was `0.009731593623802815`.  Maximum hard/nearest norm ratios were
`1.1681207385703258` and `1.155893900046496`.  All 400 association shuffles
rejected before endpoint, and every one-shot/store/control gate passed.

Focused combined validation returned `24 passed` with two existing PyTorch
sparse-CSR warnings.  No confirmation seed was opened.
