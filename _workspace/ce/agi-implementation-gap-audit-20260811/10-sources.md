# Sources

Status: COMPLETE

This audit is repository-local. No external scientific claim was promoted.

Canonical priority:

1. `docs/7_AGI/8_Roadmap.md` sections 0.4, 1.1, 1.2
2. `docs/7_AGI/14_BrainRuntimeSpec.md` sections 14.5, 14.6
3. `docs/7_AGI/18_CodeMap.md` sections 10, 12
4. Concrete code and tests under `reality_stone/python/reality_stone/clarus/` and `tests/`
5. Locked experiment artifacts under `artifacts/`, `results/`, and AGI development folders

Document conflicts found:

- `13_Verification.md` and late parts of `17_AgentLoop.md` still call STDP and neuromodulation unimplemented. Newer canonical files show STDP is wired but ineffective, while neuromodulation is implemented as an isolated mapping but only partly integrated.
- `9_LLM.md` describes GaugeLatticeV2 as implemented, while current Roadmap and CodeMap classify perturbative channel mixing and cross-frequency coupling as missing from the canonical runtime.
- Several LLM paths in `9_LLM.md` refer to removed legacy artifacts and cannot be treated as current implementation evidence.

