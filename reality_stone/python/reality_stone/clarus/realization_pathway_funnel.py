"""Hard-gate funnel for physically distinct spatial-folding pathways."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealizationPathway:
    name: str
    theory_scope: str
    explicit_action: bool
    negative_stress_derived: bool
    self_consistent_backreaction: bool
    ambient_shortcut: bool
    complete_linear_stability: bool
    engineering_scale_bridge: bool
    fatal_veto: bool
    verdict: str
    decisive_next_gate: str

    @property
    def physical_gate_count(self) -> int:
        return sum(
            (
                self.explicit_action,
                self.negative_stress_derived,
                self.self_consistent_backreaction,
                self.ambient_shortcut,
                self.complete_linear_stability,
                self.engineering_scale_bridge,
            )
        )

    @property
    def full_realization_pass(self) -> bool:
        return self.physical_gate_count == 6 and not self.fatal_veto


def spatial_folding_realization_funnel() -> tuple[RealizationPathway, ...]:
    """Return broad candidate classes without assigning unsupported odds."""

    candidates = (
        RealizationPathway(
            name="Einstein-Maxwell charged-fermion long wormhole",
            theory_scope="known 4D semiclassical model",
            explicit_action=True,
            negative_stress_derived=True,
            self_consistent_backreaction=True,
            ambient_shortcut=False,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="PHYSICAL MODEL / NOT A SHORTCUT",
            decisive_next_gate="CE map is irrelevant unless ambient-shortcut requirement is relaxed",
        ),
        RealizationPathway(
            name="AdS two-boundary double-trace wormhole",
            theory_scope="controlled holographic model",
            explicit_action=True,
            negative_stress_derived=True,
            self_consistent_backreaction=True,
            ambient_shortcut=False,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="CONTROL MODEL / WRONG ASYMPTOTICS",
            decisive_next_gate="derive an asymptotically flat local interaction",
        ),
        RealizationPathway(
            name="material-boundary static Casimir throat",
            theory_scope="QED boundary ansatz plus GR target",
            explicit_action=True,
            negative_stress_derived=True,
            self_consistent_backreaction=False,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="PHYSICAL SCALE FAIL",
            decisive_next_gate="supply a reflecting boundary at the required subnuclear scale",
        ),
        RealizationPathway(
            name="dynamic Casimir multimode squeezed pulse",
            theory_scope="laboratory quantum electrodynamics",
            explicit_action=True,
            negative_stress_derived=False,
            self_consistent_backreaction=False,
            ambient_shortcut=False,
            complete_linear_stability=True,
            engineering_scale_bridge=True,
            fatal_veto=True,
            verdict="REAL EFFECT / NOT A STATIC GRAVITY SOURCE",
            decisive_next_gate="derive sustained renormalized negative stress, not emitted photons",
        ),
        RealizationPathway(
            name="CE nonminimal scalar xi R Phi^2",
            theory_scope="current CE effective action term",
            explicit_action=True,
            negative_stress_derived=False,
            self_consistent_backreaction=False,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="REFUTED AS STANDALONE HEALTHY SOURCE",
            decisive_next_gate="requires additional exotic matter or a theory beyond scalar-tensor",
        ),
        RealizationPathway(
            name="CE massive vacuum polarization",
            theory_scope="CE semiclassical estimate",
            explicit_action=False,
            negative_stress_derived=False,
            self_consistent_backreaction=False,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="REFUTED AS MACROSCOPIC HEAVY-FIELD SOURCE",
            decisive_next_gate="identify a massless or collective CE sector",
        ),
        RealizationPathway(
            name="beyond-Horndeski wormhole",
            theory_scope="external modified-gravity extension",
            explicit_action=True,
            negative_stress_derived=True,
            self_consistent_backreaction=True,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=False,
            verdict="HIGH-ENERGY STABLE CONTROL / REALITY INCOMPLETE",
            decisive_next_gate=(
                "close slow tachyons, GR weak-field limit, luminal GW, and scale bridge"
            ),
        ),
        RealizationPathway(
            name="thin-shell cut-and-paste wormhole",
            theory_scope="classical GR junction construction",
            explicit_action=True,
            negative_stress_derived=False,
            self_consistent_backreaction=True,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=False,
            verdict="GEOMETRY EXACT / SCALE-FREE EDGE QFT REFUTED",
            decisive_next_gate=(
                "derive a stable massive or anisotropic defect QFT with the junction EoS"
            ),
        ),
        RealizationPathway(
            name="CE non-material topological boundary",
            theory_scope="proposed CE-native extension",
            explicit_action=False,
            negative_stress_derived=False,
            self_consistent_backreaction=False,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="REFUTED AS PURE SOURCE / EDGE QFT REQUIRED",
            decisive_next_gate="specify a dynamical edge theory and derive its renormalized stress",
        ),
        RealizationPathway(
            name="phantom or wrong-sign kinetic field",
            theory_scope="ghost matter",
            explicit_action=True,
            negative_stress_derived=True,
            self_consistent_backreaction=True,
            ambient_shortcut=True,
            complete_linear_stability=False,
            engineering_scale_bridge=False,
            fatal_veto=True,
            verdict="REJECTED / GHOST",
            decisive_next_gate="none without a ghost-free UV completion",
        ),
    )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (not candidate.fatal_veto, candidate.physical_gate_count),
            reverse=True,
        )
    )
