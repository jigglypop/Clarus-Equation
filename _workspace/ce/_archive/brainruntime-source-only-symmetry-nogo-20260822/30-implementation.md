# Implementation

Status: COMPLETE

BA-TR7 is an isolated source-only probe. It restores the frozen BA-TR6 substrate, resets state, pulses only one source payload at tick zero, and observes through tick three for delay two. It records activation and exact-delay eligibility without pulsing hidden or output coordinates and without invoking a decoder.

The probe also reverses the four hidden threshold profiles on a cloned runtime. First-arrival activation and eligibility remain unchanged, separating smooth activation symmetry from later bit/lifecycle coordinate bias.
