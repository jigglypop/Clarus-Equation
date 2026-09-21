# CE-SC1: state metric, canonical map, and the common-force obstruction

[Chapter 36](../../paper/후속연구_기록과_상태선택/36_관계상태의_정준사상과_계량몫의_반례.md)
contains the continuum proof and scope.

    .venv\Scripts\python.exe experiments/ce_state_canonical_20260921/derive_state_map.py
    .venv\Scripts\python.exe experiments/ce_state_canonical_20260921/verify_full_gradient.py

The exact symbolic calculation constructs a normalized 13-component real state
whose six amplitude variations span every spatial metric variation. It verifies
the cotangent one-form in that linearized chart, then tests a distinct proposal:
using the normalized quantum state's local Berry symplectic structure directly.
In the full 26-component product-state example two smeared metric components
have bracket 3*pi^3*ell^4/(4*hbar), although canonical metric coordinates commute.

A second 52-component construction has identical positive spatial metrics over
an entire patch and full metric-variation rank in both branches, but different
gauge curvature norms. Thus identifying all states with the same metric loses
information needed by a common gauge action.

The independent full-gradient script evaluates the complete 26-component
projector and second derivatives at three exact points. It includes directions
normal to the product-state submanifold; the continuum identity and integral
are proved in the chapter and checked by the main symbolic calculation.

These results do not derive a microscopic gravitational momentum, the Lorentzian
constraint algebra, quantum closure, or observed gauge fields. They specify why
two direct replacements for the assumptions of chapter 35 do not yet suffice.
