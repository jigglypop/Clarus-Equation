import math

from reality_stone.clarus.llm_pre_eq import (
    CandidateAnswer,
    ClaimAudit,
    ClaimAxisEvidence,
    ClaimGraphEdge,
    ClaimResidualVerifier,
    ClaimResidualVerifierConfig,
    EvidenceClaim,
    LabeledCandidateSet,
    PreEqVerifier,
    PreEqVerifierConfig,
    ResidualAnswerCandidate,
    ResidualClaim,
    evaluate_labeled_sets,
)


def test_pre_eq_verifier_selects_low_defect_over_high_prior_hallucination() -> None:
    verifier = PreEqVerifier(PreEqVerifierConfig(beta=2.0, min_gap=0.4))
    candidates = (
        CandidateAnswer(
            "Fluent but false.",
            prior_weight=0.8,
            supported_claims=1,
            contradicted_claims=1,
            unsupported_claims=1,
        ),
        CandidateAnswer("Grounded answer.", prior_weight=0.2, supported_claims=4),
    )

    decision = verifier.select(candidates)

    assert decision.selected_index == 1
    assert decision.selected_text == "Grounded answer."
    assert not decision.abstained
    assert decision.energy_gap > 0.4
    assert decision.posterior[1] > decision.posterior[0]


def test_pre_eq_verifier_abstains_when_manifest_gap_is_small() -> None:
    verifier = PreEqVerifier(PreEqVerifierConfig(beta=2.0, min_gap=1.0))
    candidates = (
        CandidateAnswer("Maybe A.", prior_weight=0.5, supported_claims=1),
        CandidateAnswer("Maybe B.", prior_weight=0.5, supported_claims=1),
    )

    decision = verifier.select(candidates)

    assert decision.abstained
    assert decision.selected_index is None
    assert decision.reason == "gap_below_threshold"
    assert math.isclose(decision.energy_gap, 0.0)


def test_pre_eq_verifier_abstains_when_all_candidates_have_high_defect() -> None:
    verifier = PreEqVerifier(PreEqVerifierConfig(beta=2.0, min_gap=0.0, max_energy=0.9))
    candidates = (
        CandidateAnswer("Bad A.", prior_weight=0.6, contradicted_claims=1),
        CandidateAnswer("Bad B.", prior_weight=0.4, unsupported_claims=3),
    )

    decision = verifier.select(candidates)

    assert decision.abstained
    assert decision.reason == "energy_above_threshold"


def test_labeled_set_metrics_show_hallucination_reduction() -> None:
    verifier = PreEqVerifier(
        PreEqVerifierConfig(beta=2.0, min_gap=0.4, max_energy=3.0, min_manifest_posterior=0.45)
    )
    cases = (
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer("False high prior.", 0.7, supported_claims=1, contradicted_claims=1),
                CandidateAnswer("True grounded.", 0.2, supported_claims=4),
                CandidateAnswer("Weak answer.", 0.1, unsupported_claims=1),
            ),
            correct_index=1,
        ),
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer("Wrong confident.", 0.6, unsupported_claims=2),
                CandidateAnswer("Right with evidence.", 0.3, supported_claims=3),
                CandidateAnswer("Wrong contradicted.", 0.1, contradicted_claims=1),
            ),
            correct_index=1,
        ),
        LabeledCandidateSet(
            candidates=(
                CandidateAnswer("Unsupported A.", 0.5, unsupported_claims=2),
                CandidateAnswer("Unsupported B.", 0.3, unsupported_claims=2),
                CandidateAnswer("Insufficient evidence.", 0.2, supported_claims=1, uncertainty_flags=1),
            ),
            correct_index=2,
        ),
    )

    metrics = evaluate_labeled_sets(verifier, cases)

    assert metrics.total == 3
    assert metrics.correct == 3
    assert metrics.hallucinated == 0
    assert metrics.baseline_correct == 0
    assert metrics.baseline_hallucinated == 3
    assert metrics.exact_accuracy == 1.0
    assert metrics.baseline_accuracy == 0.0
    assert metrics.defect_baseline_accuracy == 1.0


def test_beta_zero_recovers_prior_posterior_and_map_prior_selection() -> None:
    verifier = PreEqVerifier(PreEqVerifierConfig(beta=0.0, min_gap=0.0, max_energy=10.0))
    candidates = (
        CandidateAnswer("high prior high defect", prior_weight=0.8, contradicted_claims=2),
        CandidateAnswer("low prior low defect", prior_weight=0.2, supported_claims=4),
    )

    decision = verifier.select(candidates)

    assert decision.selected_index == 0
    assert math.isclose(decision.posterior[0], 0.8)
    assert math.isclose(decision.posterior[1], 0.2)


def test_large_beta_concentrates_on_low_defect_candidate() -> None:
    verifier = PreEqVerifier(PreEqVerifierConfig(beta=20.0, min_gap=0.0, max_energy=10.0))
    candidates = (
        CandidateAnswer("high prior high defect", prior_weight=0.99, contradicted_claims=1),
        CandidateAnswer("low prior low defect", prior_weight=0.01, supported_claims=4),
    )

    decision = verifier.select(candidates)

    assert decision.selected_index == 1
    assert decision.posterior[1] > 0.999


def test_defect_components_sum_to_clipped_energy() -> None:
    verifier = PreEqVerifier()
    candidate = CandidateAnswer(
        "mixed candidate",
        supported_claims=2,
        unsupported_claims=3,
        contradicted_claims=1,
        instruction_violations=1,
        self_contradictions=1,
        uncertainty_flags=2,
    )

    components = verifier.defect_components(candidate)
    raw = sum(components.values())

    assert math.isclose(verifier.defect_energy(candidate), max(0.0, raw))
    assert all(value >= 0.0 for value in components.values())
    assert components["coverage"] > 0.0


def test_no_evidence_candidate_receives_nonzero_defect() -> None:
    verifier = PreEqVerifier()
    no_claim_candidate = CandidateAnswer("No auditable claims.")

    assert verifier.defect_energy(no_claim_candidate) > 0.0


def test_claim_audit_folds_atomic_labels_into_candidate_counts() -> None:
    audit = ClaimAudit(
        claims=(
            EvidenceClaim("A is supported.", "supported"),
            EvidenceClaim("B is missing.", "unsupported"),
            EvidenceClaim("C is false.", "contradicted"),
            EvidenceClaim("D is also supported.", "SUPPORTED"),
        ),
        instruction_violations=1,
        uncertainty_flags=1,
    )

    candidate = audit.to_candidate("answer", prior_weight=0.4)

    assert candidate.supported_claims == 2
    assert candidate.unsupported_claims == 1
    assert candidate.contradicted_claims == 1
    assert candidate.instruction_violations == 1
    assert candidate.uncertainty_flags == 1


def test_posterior_gap_can_force_abstention() -> None:
    verifier = PreEqVerifier(
        PreEqVerifierConfig(
            beta=0.0,
            min_gap=0.0,
            max_energy=10.0,
            min_manifest_posterior=0.0,
            min_posterior_log_gap=0.5,
        )
    )
    candidates = (
        CandidateAnswer("A", prior_weight=0.5, supported_claims=1),
        CandidateAnswer("B", prior_weight=0.5, supported_claims=1),
    )

    decision = verifier.select(candidates)

    assert decision.abstained
    assert decision.reason == "posterior_gap_below_threshold"


def test_policy_can_require_posterior_map_to_match_defect_minimizer() -> None:
    verifier = PreEqVerifier(
        PreEqVerifierConfig(
            beta=0.1,
            min_gap=0.0,
            max_energy=10.0,
            min_manifest_posterior=0.0,
            require_defect_minimizer=True,
        )
    )
    candidates = (
        CandidateAnswer("High prior but worse", prior_weight=0.99, unsupported_claims=1),
        CandidateAnswer("Low prior but cleaner", prior_weight=0.01, supported_claims=2),
    )

    decision = verifier.select(candidates)

    assert decision.defect_min_index == 1
    assert decision.abstained
    assert decision.reason == "posterior_not_defect_minimizer"


def test_unknown_claim_label_raises() -> None:
    audit = ClaimAudit(claims=(EvidenceClaim("ambiguous", "maybe"),))

    try:
        audit.to_candidate("answer")
    except ValueError as exc:
        assert "unknown claim label" in str(exc)
    else:
        raise AssertionError("unknown claim label did not raise")


def test_rust_and_numpy_posterior_state_match_when_available() -> None:
    verifier = PreEqVerifier(PreEqVerifierConfig(beta=2.5))
    candidates = (
        CandidateAnswer("A", prior_weight=0.7, supported_claims=1, unsupported_claims=2),
        CandidateAnswer("B", prior_weight=0.2, supported_claims=4),
        CandidateAnswer("C", prior_weight=0.1, contradicted_claims=1),
    )

    try:
        np_prior, np_energy, np_posterior, np_backend = verifier.posterior_state(
            candidates,
            backend="numpy",
        )
        rust_prior, rust_energy, rust_posterior, rust_backend = verifier.posterior_state(
            candidates,
            backend="rust",
        )
    except RuntimeError:
        return

    assert np_backend == "numpy"
    assert rust_backend == "rust"
    assert np_prior.tolist() == rust_prior.tolist()
    assert np_energy.tolist() == rust_energy.tolist()
    assert np_posterior.tolist() == rust_posterior.tolist()


def _axis(axis: str, value: float, reference: float, *, family: str) -> ClaimAxisEvidence:
    return ClaimAxisEvidence(
        axis=axis,
        value=value,
        reference=reference,
        sigma=0.5,
        source_reliability=1.0,
        source_family=family,
    )


def test_claim_residual_verifier_selects_grounded_answer_over_prior_trap() -> None:
    verifier = ClaimResidualVerifier(
        ClaimResidualVerifierConfig(beta=2.0, min_gap=0.0, min_manifest_posterior=0.1)
    )
    wrong = ResidualAnswerCandidate(
        "CE exactly proves RH.",
        prior_weight=0.8,
        claims=(
            ResidualClaim(
                "CE exactly proves RH.",
                (
                    _axis("support", 1.0, 0.0, family="paper"),
                    _axis("support", 1.0, 0.0, family="docs"),
                ),
            ),
        ),
        required_slots=1,
        covered_slots=1,
    )
    right = ResidualAnswerCandidate(
        "CE uses Riemann structure as an engineering axiom, not a proof.",
        prior_weight=0.2,
        claims=(
            ResidualClaim(
                "CE uses Riemann structure as an engineering axiom, not a proof.",
                (
                    _axis("support", 1.0, 1.0, family="paper"),
                    _axis("support", 1.0, 1.0, family="docs"),
                ),
            ),
        ),
        required_slots=1,
        covered_slots=1,
    )

    decision = verifier.select((wrong, right), backend="numpy")

    assert decision.selected_index == 1
    assert decision.accepted_claims == (
        "CE uses Riemann structure as an engineering axiom, not a proof.",
    )
    assert decision.posterior[1] > decision.posterior[0]


def test_claim_residual_verifier_rejects_single_source_claim() -> None:
    verifier = ClaimResidualVerifier()
    candidate = ResidualAnswerCandidate(
        "Only one source supports this.",
        claims=(
            ResidualClaim(
                "Only one source supports this.",
                (ClaimAxisEvidence("support", 1.0, 1.0, source_family="same"),),
            ),
        ),
    )

    state = verifier.answer_state(candidate)

    assert state.accepted_fraction == 0.0
    assert state.claim_states[0].effective_sources < verifier.config.min_effective_sources
    assert state.claim_states[0].independence_penalty > 0.0


def test_claim_residual_verifier_penalizes_signed_graph_incoherence() -> None:
    verifier = ClaimResidualVerifier()
    coherent = ResidualAnswerCandidate(
        "Coherent pair.",
        claims=(
            ResidualClaim("A", (_axis("truth", 1.0, 1.0, family="a"), _axis("truth", 1.0, 1.0, family="b"))),
            ResidualClaim("B", (_axis("truth", 1.0, 1.0, family="a"), _axis("truth", 1.0, 1.0, family="b"))),
        ),
        graph_edges=(ClaimGraphEdge(0, 1, relation=1),),
    )
    incoherent = ResidualAnswerCandidate(
        "Incoherent pair.",
        claims=(
            ResidualClaim("A", (_axis("truth", 1.0, 1.0, family="a"), _axis("truth", 1.0, 1.0, family="b"))),
            ResidualClaim("not A", (_axis("truth", 0.0, 1.0, family="a"), _axis("truth", 0.0, 1.0, family="b"))),
        ),
        graph_edges=(ClaimGraphEdge(0, 1, relation=1),),
    )

    coherent_state = verifier.answer_state(coherent)
    incoherent_state = verifier.answer_state(incoherent)

    assert incoherent_state.action > coherent_state.action
    assert incoherent_state.claim_states[0].graph_energy > coherent_state.claim_states[0].graph_energy


def test_claim_residual_rust_and_numpy_match_when_available() -> None:
    verifier = ClaimResidualVerifier(ClaimResidualVerifierConfig(beta=2.5, min_gap=0.0))
    candidates = (
        ResidualAnswerCandidate(
            "wrong",
            prior_weight=0.7,
            claims=(
                ResidualClaim(
                    "wrong",
                    (
                        _axis("support", 1.0, 0.0, family="a"),
                        _axis("support", 1.0, 0.0, family="b"),
                    ),
                ),
            ),
        ),
        ResidualAnswerCandidate(
            "right",
            prior_weight=0.3,
            claims=(
                ResidualClaim(
                    "right",
                    (
                        _axis("support", 1.0, 1.0, family="a"),
                        _axis("support", 1.0, 1.0, family="b"),
                    ),
                ),
            ),
        ),
    )
    states = verifier.answer_states(candidates)

    try:
        np_prior, np_actions, np_posterior, np_backend = verifier._posterior_state(
            candidates,
            states,
            backend="numpy",
        )
        rust_prior, rust_actions, rust_posterior, rust_backend = verifier._posterior_state(
            candidates,
            states,
            backend="rust",
        )
    except RuntimeError:
        return

    assert np_backend == "numpy"
    assert rust_backend == "rust"
    assert np_prior.tolist() == rust_prior.tolist()
    assert np_actions.tolist() == rust_actions.tolist()
    assert np_posterior.tolist() == rust_posterior.tolist()


def test_claim_residual_empty_claim_candidate_uses_same_action_components() -> None:
    verifier = ClaimResidualVerifier(ClaimResidualVerifierConfig(beta=1.0))
    candidates = (
        ResidualAnswerCandidate("empty", claims=(), required_slots=1, covered_slots=0),
        ResidualAnswerCandidate(
            "supported",
            claims=(
                ResidualClaim(
                    "supported",
                    (
                        _axis("support", 1.0, 1.0, family="a"),
                        _axis("support", 1.0, 1.0, family="b"),
                    ),
                ),
            ),
        ),
    )
    states = verifier.answer_states(candidates)
    prior, actions, posterior, backend = verifier._posterior_state(
        candidates,
        states,
        backend="numpy",
    )

    assert backend == "numpy"
    assert prior.tolist() == [0.5, 0.5]
    assert actions.tolist() == [states[0].action, states[1].action]
    assert posterior[1] > posterior[0]
