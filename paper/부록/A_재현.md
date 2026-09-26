# 부록 A 재현

이 부록은 본문의 계산을 다시 돌리는 방법과, 조마다 그 계산을 구현한 모듈을 적는다. 모든 명령은 저장소 루트에서 실행한다.

## A.1 실행

```powershell
python -B -m examples.physics.rendering.ce_rendering_registry
python -B -m pytest -p no:cacheprovider tests -q
python -B -m test_support.markdown_math paper
```

첫 줄은 공동 채점을, 둘째 줄은 필수 테스트 전체를, 셋째 줄은 원고의 수식 표기 검사를 실행한다. 필수 테스트는 사전 등록 파일의 자기 해시와 잠긴 모듈의 해시도 검사한다. 테스트 통과는 구현의 회귀 증거이며 물리 주장의 증명이 아니다.

## A.2 조와 모듈

경로는 저장소 루트 기준이다. 한 조에 여러 모듈이 있으면 모두 적었다.

### 2장 렌더링 공리

- 조 2.2.2, 2.2.3, 2.4.5, 2.5.1–2.5.3: examples/physics/rendering/ce_rendering_registry.py (core(), calibrated_alpha_s(), rendering_amplitude)
- 조 2.3.3: examples/physics/rendering/ce_rendering_record_update.py
- 조 2.3.4–2.3.6: examples/physics/rendering/ce_rendering_bool.py, ce_rendering_staircase.py, ce_rendering_boundary.py
- 조 2.4.1–2.4.3: examples/physics/rendering/ce_rendering_e4_gauge.py
- 조 2.4.6, 2.7.1, 2.7.2: examples/physics/rendering/ce_rendering_e4_anchor.py
- 조 2.4.7: examples/physics/rendering/ce_rendering_e4_shape.py
- 조 2.6.2: examples/physics/rendering/ce_rendering_one_event.py
- 조 2.6.3: examples/physics/rendering/ce_rendering_distinction.py
- 조 2.6.4, 2.6.9: examples/physics/rendering/ce_rendering_pole.py
- 조 2.6.5, 2.6.8: examples/physics/rendering/ce_rendering_event_scale.py
- 조 2.6.6: examples/physics/rendering/ce_rendering_balance.py
- 조 2.6.7: examples/physics/rendering/ce_rendering_ruler.py
- 조 2.7.2 (하나의 자기): examples/physics/rendering/ce_rendering_single_self.py
- 조 2.7.3: examples/physics/rendering/ce_rendering_boundary_loop.py
- 조 2.7.4: examples/physics/rendering/ce_rendering_biased_coin.py
- 조 2.7.5: examples/physics/rendering/ce_rendering_fp_ladder.py, ce_rendering_present_share.py
- 조 2.8.3: examples/physics/rendering/ce_rendering_core_four.py, ce_rendering_formula_status.py

### 3장 게이지 구조와 물질

- 조 3.1.2: examples/physics/rendering/ce_rendering_gauge.py (gauge_structure_checks); 고정 검사 tests/test_rendering_registry.py::test_stage_preserving_symmetry_is_the_sm_gauge_group
- 조 3.1.4: examples/physics/rendering/ce_rendering_gauge.py (hypercharge, fock_operator, gauge_structure_checks); 고정 검사 tests/test_rendering_registry.py::test_stage_preserving_symmetry_is_the_sm_gauge_group
- 조 3.2.2: examples/physics/rendering/ce_rendering_gauge.py (multiplet_content, matches_one_generation); 고정 검사 tests/test_rendering_registry.py::test_rendered_stages_give_one_anomaly_free_sm_generation
- 조 3.2.3: examples/physics/rendering/ce_rendering_bool.py (charge_conjugation_on_full_space); 고정 검사 tests/test_rendering_registry.py::test_bool_dictionary_complement_halves_ckm_right_angle_and_charge_conjugation
- 조 3.2.4: examples/physics/rendering/ce_rendering_gauge.py (anomaly_sums); 고정 검사 tests/test_rendering_registry.py::test_rendered_stages_give_one_anomaly_free_sm_generation
- 조 3.2.5: examples/physics/rendering/ce_rendering_gauge.py (unification_sin2); examples/physics/rendering/ce_rendering_e4_trace.py (unweighted_ratio, 차수 가중 $a^{\lvert S\rvert}$·$a^{2\lvert S\rvert}$)
- 조 3.2.6: examples/physics/rendering/ce_rendering_lepton_trace.py (sector_ratios, weighted_range); examples/physics/rendering/ce_rendering_e4_trace.py (class_ii_scan)
- 조 3.3.2: examples/physics/rendering/ce_rendering_generations.py (generation_count); examples/physics/rendering/ce_rendering_gauge.py (anomaly_sums)
- 조 3.3.3: examples/physics/rendering/ce_rendering_generations.py (circulant, ckm_from_circulants); 고정 검사 tests/test_rendering_registry.py::test_cycle_generations_explain_small_quark_and_large_lepton_mixing
- 조 3.3.4: examples/physics/rendering/ce_rendering_generations.py (lepton_mixing_leading); 고정 검사 tests/test_rendering_registry.py::test_cycle_generations_explain_small_quark_and_large_lepton_mixing
- 조 3.3.5: examples/physics/rendering/ce_rendering_registry.py (circulant_eigenvector_drift); 고정 검사 tests/test_rendering_registry.py::test_ouroboros_generator_conserves_probability_and_energy
- 조 3.3.6: examples/physics/rendering/ce_rendering_registry.py (ouroboros_generator, ouroboros_unitary, time_average_occupation); 고정 검사 tests/test_rendering_registry.py::test_ouroboros_generator_conserves_probability_and_energy, test_ouroboros_cycle_renders_every_axis_evenly
- 조 3.4.1: examples/physics/rendering/ce_rendering_ewsb.py (higgs_channel, weak_t3); 고정 검사 tests/test_rendering_registry.py::test_weak_channel_breaking_leaves_electric_charge_and_quantizes_it
- 조 3.4.3: examples/physics/rendering/ce_rendering_ewsb.py (gauge_boson_masses); 고정 검사 tests/test_rendering_registry.py::test_weak_channel_breaking_leaves_electric_charge_and_quantizes_it
- 조 3.4.4: examples/physics/rendering/ce_rendering_ewsb.py (charges_all_states); 고정 검사 tests/test_rendering_registry.py::test_weak_channel_breaking_leaves_electric_charge_and_quantizes_it
- 조 3.5.1: examples/physics/record/dimensional_filter.py (build_filter)
- 조 3.5.2: examples/physics/rendering/ce_rendering_gauge.py (rendering_channel_checks); 고정 검사 tests/test_rendering_registry.py::test_rendering_channel_is_cptp_and_non_signalling
- 조 3.6.1: examples/physics/rendering/ce_rendering_inflation.py (stage_preserving_dimension, inflation_efolds); 고정 검사 tests/test_rendering_registry.py::test_inflation_gauge_count_is_the_stage_preserving_subalgebra

### 4장 섞임과 CP 위상

- 조 4.1.1–4.1.3: examples/physics/rendering/ce_rendering_registry.py (generation_weight, transition_factor); 고정 검사 tests/test_rendering_registry.py::test_generation_weight_gives_the_unique_transition_sign_pattern
- 조 4.1.4: examples/physics/rendering/ce_rendering_boundary_loop.py (alpha_free_relation, verdict); 고정 검사 tests/test_rendering_registry.py::test_boundary_loop_is_linear_and_lepton_ratio_disfavours_fixed_point
- 조 4.2.1, 4.2.2: examples/physics/rendering/ce_rendering_registry.py (rendering_amplitude_closed, flavour_words, rows의 $|V_{us}|$·$|V_{cb}|$ 행, score)
- 조 4.2.4: examples/physics/rendering/ce_rendering_registry.py (colour_winding); 고정 검사 tests/test_rendering_registry.py::test_colour_winding_gives_the_quark_time_channel_only
- 조 4.3.1, 4.3.4: examples/physics/rendering/ce_rendering_registry.py (mixing_matrix, _triangle_angles, ckm_triangle); 고정 검사 tests/test_rendering_registry.py::test_grade_partition_triangle_derives_vub_and_mirror_orientation
- 조 4.3.2, 4.3.3: examples/physics/rendering/ce_rendering_bool.py (complement_symmetry, ckm_triangle_from_partition, dictionary, coverage); 고정 검사 tests/test_rendering_registry.py::test_bool_dictionary_complement_halves_ckm_right_angle_and_charge_conjugation
- 조 4.3.5: examples/physics/rendering/ce_rendering_biased_coin.py (ckm_angles, angle_pulls, universality)
- 조 4.4.1: examples/physics/rendering/ce_rendering_generations.py (lepton_mixing_leading), examples/physics/rendering/ce_rendering_registry.py (순환 평균 cycle_average_occupation)
- 조 4.4.2–4.4.4: examples/physics/rendering/ce_rendering_boundary.py (contrast_vector, best_tbm_overlap, s12_tm1, s12_tm2, tm1_matrix, tm1_angles, tm1_phase_for_s2); 고정 검사 tests/test_rendering_registry.py::test_octant_is_the_boundary_between_me_and_outside
- 조 4.4.5: examples/physics/rendering/ce_rendering_staircase.py (world_heights, truth_bias_for_height), examples/physics/rendering/ce_rendering_registry.py (exterior_channels, distinction_channels)
- 조 4.4.6: examples/physics/rendering/ce_rendering_registry.py (pmns_s2), examples/physics/rendering/ce_rendering_staircase.py (RULES, angles, score_rules); 고정 검사 tests/test_rendering_registry.py::test_distinction_channels_fix_the_pmns_coefficients, test_rendering_staircase_fixes_two_steps_and_leaves_the_octant_step_open
- 조 4.4.7: examples/physics/rendering/ce_rendering_staircase.py (alignment, cos_phi_for)
- 조 4.4.8: examples/physics/rendering/ce_rendering_staircase.py (not_me_pair); 고정 검사 tests/test_rendering_registry.py::test_true_false_worlds_force_the_doubling_staircase_and_the_not_me_mirror
- 조 4.4.9: examples/physics/rendering/ce_rendering_data_sensitivity.py (table), examples/physics/rendering/ce_rendering_staircase.py (data_step_height), examples/physics/rendering/ce_rendering_open_checks.py (rescore_2026)
- 조 4.5.1, 4.5.3: examples/physics/rendering/ce_rendering_registry.py (pmns_matrix, delta_pmns_tm1), examples/physics/rendering/ce_rendering_boundary.py (delta_from_column1, tm1_delta_for_octant); 고정 검사 tests/test_rendering_registry.py::test_tm1_condition_fixes_delta_up_to_circulation
- 조 4.5.2: examples/physics/rendering/ce_rendering_registry.py (delta_pmns_tm1의 기본 방향 = $-\operatorname{sgn}J_{\rm CKM}$)
- 조 4.5.4: examples/physics/rendering/ce_rendering_registry.py (circulant_eigenvector_drift); 고정 검사 tests/test_rendering_registry.py::test_cosmic_cyclic_phase_does_not_move_the_mixing
- 조 4.6.1–4.6.8: experiments/preregistration/rendering_predictions_v26.json (P01–P06, P29, P30); 채점 행 examples/physics/rendering/ce_rendering_registry.py (rows)

### 5장 결합과 질량

- 조 5.1.1: examples/physics/rendering/ce_rendering_registry.py (행 `m_W/m_Z`, 상수 `C_ONSHELL`), examples/physics/rendering/ce_rendering_ruler.py (기록 눈금 $-2.21\sigma$)
- 조 5.1.2: examples/physics/rendering/ce_rendering_registry.py (`alpha_em_inv`), examples/physics/rendering/ce_rendering_core_four.py (강제 부분, 설명 길이)
- 조 5.1.3: examples/physics/rendering/ce_rendering_registry.py (`alpha_em_inv`의 `channel_loop`, `v_over_mpl`), examples/physics/rendering/ce_rendering_bool.py (사건 사전), examples/physics/rendering/ce_rendering_ledger.py (장부 1.6/1.6)
- 조 5.1.4: examples/physics/rendering/ce_rendering_registry.py (행 `alpha_em^-1(M_Z)`), examples/physics/rendering/ce_rendering_ruler.py (기록 눈금 $+7.02\sigma$), examples/physics/rendering/ce_rendering_present_share.py (합규칙 자)
- 조 5.1.5: examples/physics/rendering/ce_rendering_core_four.py (s1·s2)
- 조 5.2.1: 모듈 없음(스킴 차이 약 $-0.98\%$는 2026-09-26/27 감사의 값; 채점 상수는 examples/physics/rendering/ce_rendering_registry.py의 `M_E`, `M_MU`, `M_TAU`)
- 조 5.2.2: examples/physics/rendering/ce_rendering_registry.py (`flavour_words`, 행 `m_mu/m_tau`), examples/physics/rendering/ce_rendering_boundary_loop.py (경계 인자 꼴 시험), examples/physics/rendering/ce_rendering_ruler.py (기록 눈금 $+7.46\sigma$)
- 조 5.2.3: examples/physics/rendering/ce_rendering_registry.py (`koide_me_over_mmu`, 행 `m_e/m_mu`)
- 조 5.2.4: examples/physics/rendering/ce_rendering_boundary_loop.py (렙톤 비의 $\alpha_s$, 고정점 긴장), examples/physics/rendering/ce_rendering_present_share.py (자들과 τ 질량 부산물)
- 조 5.2.5: examples/physics/rendering/ce_rendering_boundary_loop.py ($\alpha_s$ 없는 관계), examples/physics/rendering/ce_rendering_distinction.py (섞임각 정의 판별)
- 조 5.3.1: examples/physics/rendering/ce_rendering_registry.py (`v_over_mpl`), examples/physics/rendering/ce_rendering_core_four.py (설명 길이)
- 조 5.3.2: examples/physics/rendering/ce_rendering_registry.py (행 `v/M_Pl`), examples/physics/rendering/ce_rendering_ledger.py (전약 깊이), examples/physics/rendering/ce_rendering_higgs_weight.py ($F$ 역산), examples/physics/rendering/ce_rendering_ruler.py (기록 눈금 $-7.91\sigma$), examples/physics/rendering/ce_rendering_present_share.py (계층 식 자)
- 조 5.3.3: examples/physics/rendering/ce_rendering_planck_readout.py (세 고리 인자의 θ* 비교), examples/physics/rendering/ce_rendering_reverse_derivations.py (표준 한 고리 인자), examples/physics/rendering/ce_rendering_bool.py (ODD), examples/physics/rendering/ce_rendering_ledger.py (장부 0 bit, 두 깊이)
- 조 5.3.4: examples/physics/rendering/ce_rendering_core_four.py (조용한 통로 가족 216, 지평선 $k^*$)
- 조 5.4.1: examples/physics/rendering/ce_rendering_neutrino.py
- 조 5.4.2: examples/physics/rendering/ce_rendering_nu_audit.py
- 조 5.4.3: examples/physics/rendering/ce_rendering_nu_audit.py
- 조 5.4.4: examples/physics/rendering/ce_rendering_nu_cosmo.py
- 조 5.4.5: examples/physics/rendering/ce_rendering_neutrino.py
- 조 5.4.6: examples/physics/rendering/ce_rendering_neutrino.py, examples/physics/rendering/ce_rendering_nu_audit.py
- 조 5.4.7: examples/physics/rendering/ce_rendering_nu_fit.py (압축 적합), examples/physics/rendering/ce_rendering_nu_lens.py (렌즈 적합), examples/physics/rendering/ce_rendering_nu_lens_sigma8_grid.json ($\sigma_8$ 격자)
- 조 5.5.1: examples/physics/rendering/ce_rendering_registry.py (행 `muon Da_mu x1e11`)
- 조 5.5.2: examples/physics/rendering/ce_rendering_registry.py, examples/physics/rendering/ce_rendering_ledger.py (기준 pull)

### 6장 계보와 경주

- 조 6.1.2: examples/physics/rendering/ce_rendering_registry.py (`core()`, `calibrated_alpha_s()`: $\delta$, $D$)
- 조 6.1.3: examples/physics/rendering/ce_rendering_branching.py (`iterate_to_extinction`, `small_root`)
- 조 6.1.4: examples/physics/rendering/ce_rendering_branching.py (`iterate_to_extinction`, `monte_carlo`)
- 조 6.1.5: examples/physics/rendering/ce_rendering_branching.py (`law_specificity`)
- 조 6.1.6: examples/physics/rendering/ce_rendering_branching.py (`monte_carlo`, `composition`의 Borel 값)
- 조 6.2.1: examples/physics/rendering/ce_rendering_branching.py (`composition`)
- 조 6.2.2: examples/physics/rendering/ce_rendering_probability_weight.py (확률 무게 읽기, $\Omega_b=q$)
- 조 6.3.3: examples/physics/rendering/ce_rendering_race.py (`race_mc`)
- 조 6.3.4: examples/physics/rendering/ce_rendering_branching.py (`composition`), examples/physics/rendering/ce_rendering_registry.py (`core()`의 $\Omega_m$)
- 조 6.3.5: examples/physics/rendering/ce_rendering_higgs_weight.py (`partition_identity`)
- 조 6.3.6: examples/physics/rendering/ce_rendering_race.py (`rules`, `race_mc`의 창 규칙)
- 조 6.4.1: examples/physics/rendering/ce_rendering_registry.py (행 `M_H/M_Z`, 지위 "경험식"), examples/physics/rendering/ce_rendering_ewsb.py (대칭 깨짐 구조; 퍼텐셜 크기 λ, $M_H$는 유도하지 않음을 명시)
- 조 6.4.2: examples/physics/rendering/ce_rendering_mass_rate.py (`scope_checks`)
- 조 6.4.3: examples/physics/rendering/ce_rendering_race.py (`self_first`)
- 조 6.4.4: examples/physics/rendering/ce_rendering_higgs_weight.py (`step_forms`)
- 조 6.4.5: examples/physics/rendering/ce_rendering_higgs_weight.py (`hierarchy_cross_check`)
- 조 6.4.6: examples/physics/rendering/ce_rendering_mass_rate.py (`top_ladder`, `family`)
- 조 6.4.7: examples/physics/rendering/ce_rendering_open_predictions.py (P28), examples/physics/rendering/ce_rendering_higgs_weight.py (`q2_universality`)
- 조 6.4.8: examples/physics/rendering/ce_rendering_mass_rate.py (`top_ladder`)
- 조 6.5.1: examples/physics/rendering/ce_rendering_higgs_cosmos.py
- 조 6.5.2: examples/physics/rendering/ce_rendering_higgs_cosmos.py, examples/physics/rendering/ce_rendering_spread.py (BAO+BBN 판독 이동)
- 조 6.5.4: examples/physics/rendering/ce_rendering_higgs_cosmos.py (P35)
- 조 6.6.1: examples/physics/rendering/ce_rendering_nu_ledger.py (`early_densities`), examples/physics/rendering/ce_rendering_planck_readout.py (플랑크 판독 $h$)
- 조 6.6.2: examples/physics/rendering/ce_rendering_nu_ledger.py, examples/physics/rendering/ce_rendering_registry.py (행 $\omega_bh^2$, $\omega_ch^2$)
- 조 6.6.3: examples/physics/rendering/ce_rendering_nu_ledger.py (P12)
- 조 6.6.4: examples/physics/rendering/ce_rendering_nu_ledger.py (P13)
- 고정 검사: tests/test_rendering_registry.py::test_branching_extinction_derives_the_cosmic_composition (조 6.1.3–6.1.6, 6.2.1), ::test_race_between_self_and_capture_derives_br3_and_higgs_weight (조 6.3.3, 6.3.6, 6.4.3), ::test_higgs_weight_is_the_survival_partition_one_step (조 6.3.5, 6.4.4, 6.4.5), ::test_mass_rate_scope_and_top_ladder (조 6.4.2, 6.4.6), ::test_higgs_mass_predicts_cosmic_matter_fraction (조 6.5.1, 6.5.2), ::test_neutrino_ledger_removes_ce_neutrinos_from_early_cold_matter (조 6.6.1, 6.6.2)

### 7장 빛·중력·지평선

- 조 7.1.1–7.1.4: examples/physics/rendering/ce_rendering_complex_boost.py (정리 C′ 재증명, 반례 C-0, 분기 R), ce_rendering_axiom_proofs.py
- 조 7.1.3 (기록 에너지를 플럭스로 읽는 판본의 기각): examples/physics/rendering/ce_rendering_intermittent.py
- 조 7.1.5, 7.1.6: examples/physics/rendering/ce_rendering_light_limit.py (머리글에 정정 전 문구 잔존), ce_rendering_cycle.py ($H_\Lambda$, $c/H_\Lambda$, 둘레, 주기)
- 조 7.1.5 (텐서 모드): examples/physics/rendering/ce_rendering_generations.py
- 조 7.1.6 (온도, 최종 지평선, 사건 지평선 거리): examples/physics/rendering/ce_rendering_modular_time.py (MK6, MK7, axis_cycle()), ce_rendering_thermal_time.py
- 조 7.1.7: examples/physics/rendering/ce_rendering_ledger.py (플랑크 해상도 가설과 LHAASO 반증 조건)
- 조 7.2.1–7.2.3: examples/physics/rendering/ce_rendering_probability_weight.py (벨 쌍 무신호, BMV), ce_rendering_axiom_proofs.py (정리 A 검산)
- 조 7.3.1, 7.3.2, 7.3.4: examples/physics/rendering/ce_rendering_record_update.py
- 조 7.3.3: examples/physics/rendering/ce_rendering_closure.py (기록 완성 기준, 정수 통로 반례)
- 조 7.4.1–7.4.3: examples/physics/rendering/ce_rendering_horizon_pixel.py
- 조 7.5.1, 7.5.2: examples/physics/rendering/ce_rendering_jacobson.py
- 조 7.6.1: examples/physics/rendering/ce_rendering_modular.py
- 조 7.6.2, 7.6.3: examples/physics/rendering/ce_rendering_joint_render.py
- 조 7.7.1: examples/physics/rendering/ce_rendering_first_event.py (audit()), ce_rendering_pocket_boundary.py (audit())
- 조 7.7.2, 7.7.3: examples/physics/rendering/ce_rendering_core_four.py, ce_rendering_audit.py (30기호)
- 조 7.7.4: examples/physics/rendering/ce_rendering_registry.py (hubble_kms), ce_rendering_planck_readout.py, ce_rendering_derivations.py (판독 인자 없는 $h$의 음향 각도 반례), ce_rendering_ledger.py (두 깊이)
- 조 7.7.5: examples/physics/rendering/ce_rendering_parabolic_lock.py, ce_rendering_horizon_towers.py, ce_rendering_tower_temperature.py, ce_rendering_two_pi_squared.py
- 조 7.7.6: examples/physics/rendering/ce_rendering_horizon_constant.py, ce_rendering_horizon_towers.py (HT-δ)
- 검사(tests/test_rendering_registry.py): test_light_speed_as_rendering_limit_fixes_the_tilt_and_bounds_all_signals, test_complex_boost_reproof_of_light_limit_and_bisector, test_axiom_proofs_qg1_c6_light_limit_and_bisector, test_tensor_modes_travel_at_light_speed_in_the_abs_a_background (조 7.1); test_probability_weight_gravity_is_the_only_no_signalling_reading (조 7.2); test_records_update_the_gravity_source_only_inside_the_light_cone, test_record_completion_is_thermodynamic_and_horizon_is_not_an_integer_channel_count (조 7.3, 7.4.1); test_horizon_pixel_nyquist_nat_counts_a_quarter (조 7.4); test_pixel_entropy_gives_newton_constant_and_kerr_first_law (조 7.5); test_pixel_modular_fluctuation_c_equals_s_and_interferometer_tension, test_joint_rendering_cancels_differential_interferometer_noise (조 7.6); test_horizon_h0_fails_the_acoustic_angle_and_theta_calibration_restores_it, test_planck_unit_loop_transfers_to_the_horizon_readout, test_core_four_are_not_derivable_and_cost_less_as_parameters, test_horizon_thermal_towers_read_the_lead_but_not_the_constant, test_ph5_has_no_thermal_time_basis_and_n_e_needs_stiff_reheating, test_no_ce_native_torus_supplies_the_two_pi_squared, test_circle_free_critical_lock_gives_basel_sum_but_not_the_constant, test_horizon_constant_has_no_natural_mechanism_and_is_unidentifiable (조 7.7)

### 8장 시간축과 허블 판독

- 조 8.1.1: examples/physics/rendering/ce_rendering_planck_readout.py
- 조 8.1.2: examples/physics/rendering/ce_rendering_complex_scale.py
- 조 8.1.3: examples/physics/rendering/ce_rendering_complex_scale.py
- 조 8.1.3: examples/physics/rendering/ce_rendering_open_checks.py
- 조 8.1.4: examples/physics/rendering/ce_rendering_generations.py
- 조 8.2.2: examples/physics/rendering/ce_rendering_complex_boost.py
- 조 8.2.3: examples/physics/rendering/ce_rendering_registry.py
- 조 8.2.4: examples/physics/rendering/ce_rendering_open_checks.py
- 조 8.2.4: examples/physics/rendering/ce_rendering_causal_lock.py
- 조 8.3.2: examples/physics/rendering/ce_rendering_thermal_time.py
- 조 8.3.2: examples/physics/rendering/ce_rendering_modular_time.py
- 조 8.3.4: examples/physics/rendering/ce_rendering_thermal_time.py
- 조 8.3.4: examples/physics/rendering/ce_rendering_cycle.py
- 조 8.3.5: examples/physics/rendering/ce_rendering_thermal_time.py
- 조 8.3.5: examples/physics/rendering/ce_rendering_complex_boost.py
- 조 8.3.5: examples/physics/rendering/ce_rendering_causal_lock.py
- 조 8.3.6: examples/physics/rendering/ce_rendering_thermal_time.py
- 조 8.3.6: examples/physics/rendering/ce_rendering_registry.py
- 조 8.4.1: examples/physics/rendering/ce_rendering_o1_map.py
- 조 8.4.2: examples/physics/rendering/ce_rendering_time_semantics.py
- 조 8.4.3: examples/physics/rendering/ce_rendering_o1_map.py
- 조 8.4.4: examples/physics/rendering/ce_rendering_o1_map.py
- 조 8.4.4: examples/physics/rendering/ce_rendering_time_semantics.py
- 조 8.4.4: examples/physics/rendering/ce_rendering_registry.py
- 조 8.5.1: examples/physics/rendering/ce_rendering_registry.py
- 조 8.5.1: examples/physics/rendering/ce_rendering_o1_map.py
- 조 8.5.1: examples/physics/rendering/ce_rendering_audit.py
- 조 8.5.2: examples/physics/rendering/ce_rendering_ladder.py
- 조 8.5.3: examples/physics/rendering/ce_rendering_ladder.py
- 조 8.5.3: examples/physics/rendering/ce_rendering_jwst_cchp.py
- 조 8.5.4: examples/physics/rendering/ce_rendering_registry.py
- 조 8.5.5: examples/physics/rendering/ce_rendering_o1_map.py
- 조 8.5.6: examples/physics/rendering/ce_rendering_ladder.py
- 조 8.5.6: examples/physics/rendering/ce_rendering_jwst_cchp.py
- 조 8.5.7: examples/physics/rendering/ce_rendering_time_semantics.py
- 조 8.5.8: examples/physics/rendering/ce_rendering_open_predictions.py
- 조 8.5.8: examples/physics/rendering/ce_rendering_spread.py
- 조 8.6.1: examples/physics/rendering/ce_rendering_planck_readout.py
- 조 8.6.1: examples/physics/rendering/ce_rendering_derivations.py
- 조 8.6.2: examples/physics/rendering/ce_rendering_theta_nu.py
- 조 8.6.3: examples/physics/rendering/ce_rendering_bao_ruler.py
- 조 8.6.3: examples/physics/rendering/ce_rendering_nu_ledger.py
- 조 8.6.3: examples/physics/rendering/ce_rendering_o1_map.py
- 조 8.6.4: examples/physics/rendering/ce_rendering_spiral.py
- 조 8.6.5: examples/physics/rendering/ce_rendering_planck_readout.py
- 조 8.6.6: examples/physics/rendering/ce_rendering_nu_ledger.py
- 조 8.6.7: examples/physics/rendering/ce_rendering_derivations.py
- 조 8.6.8: examples/physics/rendering/ce_rendering_spiral.py
- 조 8.7.1: examples/physics/rendering/ce_rendering_gradient.py
- 조 8.7.2: examples/physics/rendering/ce_rendering_gradient.py
- 조 8.7.3: examples/physics/rendering/ce_rendering_open_predictions.py
- 조 8.7.3: examples/physics/rendering/ce_rendering_open_checks.py
- 조 8.7.3: examples/physics/rendering/ce_rendering_weight_epoch.py
- 조 8.7.4: examples/physics/rendering/ce_rendering_gradient.py
- 조 8.7.4: examples/physics/rendering/ce_rendering_ledger.py
- 조 8.7.4: examples/physics/rendering/ce_rendering_sn_holdout.py
- 조 8.7.5: examples/physics/rendering/ce_rendering_spread.py
- 조 8.7.6: examples/physics/rendering/ce_rendering_gradient.py
- 조 8.7.7: examples/physics/rendering/ce_rendering_spread.py
- 조 8.8.1: examples/physics/rendering/ce_rendering_cycle.py
- 조 8.8.1: examples/physics/rendering/ce_rendering_light_limit.py
- 조 8.8.2: examples/physics/rendering/ce_rendering_cycle.py
- 조 8.8.3: examples/physics/rendering/ce_rendering_spiral.py
- 조 8.8.4: examples/physics/rendering/ce_rendering_phase_lock.py
- 조 8.8.4: examples/physics/rendering/ce_rendering_parabolic_lock.py
- 조 8.8.5: examples/physics/rendering/ce_rendering_causal_lock.py
- 조 8.8.6: examples/physics/rendering/ce_rendering_lock_history.py
- 조 8.8.7: examples/physics/rendering/ce_rendering_thermal_time.py
- 조 8.8.7: examples/physics/rendering/ce_rendering_modular_time.py
- 조 8.8.7: examples/physics/rendering/ce_rendering_intermittent.py

### 9장 급팽창·암흑에너지·구조 성장

- 조 9.1.1: examples/physics/rendering/ce_rendering_inflation.py
- 조 9.1.1: examples/physics/rendering/ce_rendering_registry.py (`core()`의 $N_e=18D$)
- 조 9.1.2: examples/physics/rendering/ce_rendering_registry.py ($n_s$, $dn_s/d\ln k$ 채점 행)
- 조 9.1.2: examples/physics/rendering/ce_rendering_open_checks.py (2026 재채점, P-ACT $n_s$)
- 조 9.1.3: examples/physics/rendering/ce_rendering_registry.py (P09, 등록 파일 experiments/preregistration/rendering_predictions_v26.json)
- 조 9.1.4: examples/physics/rendering/ce_rendering_registry.py (P10, 등록 파일 experiments/preregistration/rendering_predictions_v26.json)
- 조 9.2.1: examples/physics/rendering/ce_rendering_registry.py (`scalar_amplitude()`, $A_s$ 채점 행)
- 조 9.2.1: examples/physics/rendering/ce_rendering_core_four.py (유도 시도, 설명 길이)
- 조 9.2.2: examples/physics/rendering/ce_rendering_core_four.py
- 조 9.2.2: examples/physics/rendering/ce_rendering_first_event.py ($H_{\rm inf}$)
- 조 9.3.1: examples/physics/rendering/ce_rendering_tower_temperature.py (`matching()`)
- 조 9.3.2: examples/physics/rendering/ce_rendering_tower_temperature.py (`matching()`의 운동 에너지 지배 요구량)
- 조 9.3.2: examples/physics/rendering/ce_rendering_first_event.py ($N(a_0H_0)$)
- 조 9.4.1: examples/physics/rendering/ce_rendering_w_branch.py
- 조 9.4.2: examples/physics/rendering/ce_rendering_vacuum_harmonic.py
- 조 9.4.2: examples/physics/rendering/ce_rendering_axiom_proofs.py (정리 B 검산)
- 조 9.4.3: examples/physics/rendering/ce_rendering_vacuum_tilt.py
- 조 9.4.5: examples/physics/rendering/ce_rendering_vacuum_tilt.py
- 조 9.4.5: examples/physics/rendering/ce_rendering_theta_nu.py (페르미–디랙 θ*)
- 조 9.4.5: examples/physics/rendering/ce_rendering_gradient.py (그라데이션을 더한 CMB–BAO)
- 조 9.4.5: examples/physics/rendering/ce_rendering_sn_holdout.py (Pantheon 보류 시험)
- 조 9.4.6: examples/physics/rendering/ce_rendering_vacuum_tilt.py (P21)
- 조 9.4.7: examples/physics/rendering/ce_rendering_vacuum_tilt.py (P22)
- 조 9.4.8: examples/physics/rendering/ce_rendering_mimetic_vacuum.py
- 조 9.4.9: examples/physics/rendering/ce_rendering_w3_growth.py
- 조 9.4.9: examples/physics/rendering/ce_rendering_data_sensitivity.py
- 조 9.4.10: examples/physics/rendering/ce_rendering_light_limit.py (`signal_speeds()`)
- 조 9.5.1: examples/physics/rendering/ce_rendering_growth.py
- 조 9.5.1: examples/physics/rendering/ce_rendering_nu_ledger.py
- 조 9.5.1: examples/physics/rendering/ce_rendering_vacuum_tilt.py (진공 기울기 판독의 성장, `s8()`)
- 조 9.5.1: examples/physics/rendering/ce_rendering_open_checks.py (DES Y6 재채점)
- 조 9.5.2: examples/physics/rendering/ce_rendering_vacuum_tilt.py (P16 값 0.8161, `s8()`)
- 조 9.5.2: examples/physics/rendering/ce_rendering_growth.py (무진동 전달함수와 상대 교정)
- 조 9.6.1: examples/physics/rendering/ce_rendering_probability_weight.py ($\Omega_k=0$, Planck 2018 + BAO pull)
- 조 9.6.1: examples/physics/rendering/ce_rendering_first_event.py (DESI DR2 + CMB 자료 상수 `OMEGA_K_DATA`; $-1.91\sigma$는 §43.111의 단순 산술이며 전용 함수 없음)
- 조 9.6.2: examples/physics/rendering/ce_rendering_probability_weight.py (P24)

### 10장 기록과 비관측 상태

- 조 10.1.1–10.1.4: examples/physics/rendering/ce_rendering_intermittent.py (convex_curve, chord_tilt_on_curve, loop_structure, circle_timetable, typical_tilt, flux_variant_chi2)
- 조 10.1.2 (위상 잠금으로 본 흐려짐, 8.8절 인용): examples/physics/rendering/ce_rendering_phase_lock.py (circle_map, laminar_lengths, slip_depth)
- 조 10.1.3, 10.1.4: examples/physics/rendering/ce_rendering_thermal_time.py (phase_histories, cycle_timetable, exact_half_now), ce_rendering_lock_history.py (theta_of_u, score)
- 조 10.1.1 (접선–현 기울기, 원의 기하): examples/physics/rendering/ce_rendering_cycle.py (tangent_chord_tilt, cycle_geometry), ce_rendering_complex_boost.py (record_energy)
- 조 10.2.1–10.2.4: examples/physics/rendering/ce_rendering_boundary_loss.py (variants, candidates, baseline_check, fit, scan, signed_scan, candidate_table, closure, rows43, scan43, supernova, verdict)
- 조 10.2.5: examples/physics/rendering/ce_rendering_boundary_loss.py (geometric_exit)
- 조 10.2.6: examples/physics/rendering/ce_rendering_boundary_loss.py (neither, lab_bound)
- 조 10.3.1–10.3.12: examples/physics/rendering/ce_rendering_modular_time.py (trichotomy, thermal_time, balance, de_sitter_rate, one_domain, axis_cycle, verdict); 속도와 시각표는 ce_rendering_thermal_time.py를 가져다 쓴다
- 조 10.4.1–10.4.3: examples/physics/rendering/ce_rendering_record_dynamics.py (irreversibility, correlation_scan)
- 조 10.4.5: examples/physics/rendering/ce_rendering_record_update.py (dp_event_scale_length, 사건 척도 자발 붕괴의 기각)
- 조 10.4.6: examples/physics/rendering/ce_rendering_record_dynamics.py (zero_point_dx, omega_sn, _gauss_pair_curvature_check, self_gravity)
- 조 10.4.7: experiments/preregistration/rendering_predictions_v26.json (P44), 검사 tests/test_rendering_registry.py::test_rendering_predictions_v26_is_frozen
- 조 10.4.6, 10.5.9 (P23, 확률 무게 원천): examples/physics/rendering/ce_rendering_probability_weight.py (bmv, cosmic_weights)
- 조 10.5.1–10.5.4, 10.5.6: examples/physics/rendering/ce_rendering_first_event.py (lorentz_geometry, branching, logistic_profile, inside_view, inflation_scale, closure, stationary_tick, balance_check, sign_test, audit)
- 조 10.5.5, 10.5.7: examples/physics/rendering/ce_rendering_bubble_capacity.py (candidates, evaluate, chance_in_windows, starobinsky_exit)
- 조 10.5.8–10.5.10: examples/physics/rendering/ce_rendering_pocket_boundary.py (ratios, entropy_identity, g1_front, standard_layer, verdict, audit); qg1_bubble()은 철회된 열린 거품 판을 단언하므로 조 10.5.9의 근거로 쓰지 않는다
- 조 10.6.1–10.6.4: examples/physics/rendering/ce_rendering_record_rate.py (budget, persistence_band, maximal_recording, verdict); curvature_link()는 철회된 곡률 연결 판이다
- 조 10.6.5–10.6.9: examples/physics/rendering/ce_rendering_distinction_end.py (budget_ln, age_gyr, global_end, local_budget, audit)
- 조 10.1–10.6 공통 코어 값: examples/physics/rendering/ce_rendering_registry.py, ce_rendering_planck_readout.py(첫 사건 모듈의 $H_0$)
- 테스트: tests/test_rendering_registry.py — test_intermittent_rendering_blurs_and_reemerges_on_any_convex_cycle(10.1), test_boundary_loss_is_bounded_and_the_vacuum_is_inherited(10.2), test_only_the_neither_state_beyond_the_boundary_makes_a_clock(10.3), test_records_cannot_be_undone_and_unrecorded_crystals_self_gravitate(10.4), test_one_first_event_gives_an_open_bubble_and_only_the_horizon_capacity_closes(10.5.1–10.5.6, 이름은 감사 전 주장), test_curvature_kills_only_small_bubble_capacities_and_the_diffusion_exit_is_flat(10.5.5, 10.5.7), test_one_ratio_sets_the_pocket_boundary_and_determination_spreads_at_light_speed(10.5.8, 10.5.9), test_landauer_budget_makes_the_undetermined_sea_grow_while_crossing_the_band(10.6.1–10.6.4), test_the_mean_record_budget_crosses_the_start_threshold_94_efolds_from_now(10.6.5–10.6.9)
- 해시 잠금: 간헐 렌더링 모듈은 v13부터, 경계 소실·모듈러 시간·기록 동역학 모듈은 v26부터 잠겼다. 첫 사건·거품 용량·주머니 경계·기록률·분별의 끝 모듈(조 10.5–10.6)은 어떤 사전 등록 판본에도 잠기지 않았다

### 11장 공동 채점

- 조 11.1.1: examples/physics/rendering/ce_rendering_registry.py (core, calibrated_alpha_s)
- 조 11.1.1: examples/physics/rendering/ce_rendering_nu_ledger.py (early_densities, rd_and_h)
- 조 11.1.1: examples/physics/rendering/ce_rendering_planck_readout.py (planck_unit_factor)
- 조 11.1.2: examples/physics/rendering/ce_rendering_registry.py (score)
- 조 11.1.2: examples/physics/rendering/ce_rendering_planck_readout.py (score_variant_iv.add)
- 조 11.1.2: examples/physics/rendering/ce_rendering_nu_ledger.py (_rescore)
- 조 11.1.3: examples/physics/rendering/ce_rendering_gradient.py (factor, bao_rows)
- 조 11.1.3: examples/physics/rendering/ce_rendering_vacuum_tilt.py (bao_vectors, cycle_phase, density)
- 조 11.1.3: examples/physics/darksector/ce_residual_forward_model.py (DESI DR2 BAO 13성분 평균·공분산)
- 조 11.1.4: examples/physics/rendering/ce_rendering_vacuum_tilt.py (score)
- 조 11.1.5: examples/physics/rendering/ce_rendering_registry.py (행 독립 가정, BAO_CINV)
- 조 11.2.1: examples/physics/rendering/ce_rendering_ledger.py (ce_rows('IV'), ce_rows('full'))
- 조 11.2.2: examples/physics/rendering/ce_rendering_ledger.py (ce_rows)
- 조 11.2.3: examples/physics/rendering/ce_rendering_ledger.py (ce_rows)
- 조 11.2.4: examples/physics/rendering/ce_rendering_gradient.py (bao_rows)
- 조 11.2.5: examples/physics/rendering/ce_rendering_growth.py
- 조 11.2.5: examples/physics/rendering/ce_rendering_neutrino.py
- 조 11.3.2: examples/physics/rendering/ce_rendering_vacuum_tilt.py (score, ruler 'fixed'·'free')
- 조 11.3.2: examples/physics/rendering/ce_rendering_gradient.py (joint_v, joint_rows)
- 조 11.3.3: examples/physics/rendering/ce_rendering_ledger.py (compare)
- 조 11.3.4: examples/physics/rendering/ce_rendering_thermal_time.py (score('V-dS'))
- 조 11.3.5: examples/physics/rendering/ce_rendering_w3_growth.py
- 조 11.3.5: examples/physics/rendering/ce_rendering_mimetic_vacuum.py
- 조 11.3.6: examples/physics/rendering/ce_rendering_gradient.py
- 조 11.4.1: examples/physics/rendering/ce_rendering_data_sensitivity.py
- 조 11.4.2: examples/physics/rendering/ce_rendering_data_sensitivity.py
- 조 11.4.3: examples/physics/rendering/ce_rendering_registry.py (PMNS_DATA['noSK'])
- 조 11.4.4: examples/physics/rendering/ce_rendering_open_checks.py (rescore_2026)
- 조 11.4.5: examples/physics/rendering/ce_rendering_o1_map.py
- 조 11.4.5: examples/physics/rendering/ce_rendering_ladder.py
- 조 11.4.5: examples/physics/rendering/ce_rendering_jwst_cchp.py
- 조 11.4.5: examples/physics/rendering/ce_rendering_theta_nu.py
- 조 11.4.6: examples/physics/rendering/ce_rendering_ruler.py
- 조 11.5.1: examples/physics/rendering/ce_rendering_sn_holdout.py (자료: benchmarks/cosmology/pantheon_binned_v1, kinetic_dark_sector_gate.load_pantheon_binned)
- 조 11.5.2: examples/physics/rendering/ce_rendering_sn_holdout.py
- 재현(저장소 루트, 한국어 출력은 PYTHONIOENCODING=utf-8): python -B -c "from examples.physics.rendering import ce_rendering_ledger as L; rows,bao=L.ce_rows('IV'); print(bao); [print(o) for o in rows]" ('full'이면 43행)
- 재현: python -B -m examples.physics.rendering.ce_rendering_data_sensitivity / ce_rendering_open_checks / ce_rendering_sn_holdout / ce_rendering_gradient

### 12장 모형 비교와 계약 판정

- 조 12.1.1: examples/physics/rendering/ce_rendering_ledger.py (LEDGER, NATS_PER_BIT_CHI2)
- 조 12.1.2: examples/physics/rendering/ce_rendering_ledger.py (LEDGER, ledger_totals)
- 조 12.1.3: examples/physics/rendering/ce_rendering_ledger.py (v14에 잠김, 20.9/34.6 출력)
- 조 12.1.4: examples/physics/rendering/ce_rendering_ledger.py (UNAUDITED)
- 조 12.1.5: examples/physics/rendering/ce_rendering_open_checks.py (장부 검산)
- 조 12.1.5: examples/physics/rendering/ce_rendering_ledger.py (block_scores, M_ITEMS)
- 조 12.2.1: examples/physics/rendering/ce_rendering_ledger.py (baseline_pulls, planck_bao_chi2)
- 조 12.2.2: examples/physics/rendering/ce_rendering_ledger.py (flexible_baseline_m)
- 조 12.2.3: examples/physics/rendering/ce_rendering_ledger.py (compare)
- 조 12.2.4: examples/physics/rendering/ce_rendering_ledger.py (compare, break_even_bits)
- 조 12.2.5: examples/physics/rendering/ce_rendering_ledger.py (block_scores)
- 조 12.3.1: examples/physics/rendering/ce_rendering_audit.py (val01_decomposition)
- 조 12.3.2: examples/physics/rendering/ce_rendering_audit.py (val01_decomposition)
- 조 12.3.3: examples/physics/rendering/ce_rendering_audit.py (val01_decomposition)
- 조 12.3.4: examples/physics/rendering/ce_rendering_gradient.py (bao_rows)
- 조 12.4.1: examples/physics/rendering/ce_rendering_ledger.py (parameter_bits)
- 조 12.4.2: examples/physics/rendering/ce_rendering_ledger.py (mdl_blocks)
- 조 12.4.3: examples/physics/rendering/ce_rendering_audit.py (grammar_mdl)
- 조 12.4.4: examples/physics/rendering/ce_rendering_audit.py (grammar_mdl)
- 조 12.4.5: examples/physics/rendering/ce_rendering_formula_status.py
- 조 12.4.6: examples/physics/rendering/ce_rendering_core_four.py
- 조 12.5.1: examples/physics/rendering/ce_rendering_audit.py (registration_timing)
- 조 12.5.2: examples/physics/rendering/ce_rendering_audit.py (registration_timing)
- 조 12.5.3: examples/physics/rendering/ce_rendering_nu_cosmo.py
- 조 12.6.3: examples/physics/rendering/ce_rendering_audit.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_e4_gauge.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_thermal_time.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_time_semantics.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_formula_status.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_core_four.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_weight_epoch.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_parabolic_lock.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_horizon_constant.py
- 조 12.7.1: examples/physics/rendering/ce_rendering_tower_temperature.py
- 재현(저장소 루트, PYTHONIOENCODING=utf-8): python -B -m examples.physics.rendering.ce_rendering_ledger
- 재현: python -B -m examples.physics.rendering.ce_rendering_audit / ce_rendering_formula_status / ce_rendering_core_four

### 13장 사전 등록 예측

- 조 13.1.1: experiments/preregistration/rendering_predictions_v1.json … experiments/preregistration/rendering_predictions_v26.json
- 조 13.1.1: experiments/preregistration/README.md
- 조 13.1.3: examples/physics/rendering/ce_rendering_audit.py (registration_timing)
- 조 13.1.6: tests/test_rendering_registry.py (판본별 자기 해시·코드 해시·예측 재현)
- 조 13.1.6: tests/test_holdout_preregistration.py (동결·해시·자료 역할·재피팅 금지)
- 조 13.1.6: experiments/preregistration/validate_holdout_manifest.py
- 재현: python -B -m experiments.preregistration.validate_holdout_manifest --no-verify-artifacts
- 재현: python -B -m pytest -p no:cacheprovider tests/test_holdout_preregistration.py -q
- 조 13.2.4: examples/physics/rendering/ce_rendering_registry.py (P01–P04, v1)
- 조 13.2.3: examples/physics/rendering/ce_rendering_registry.py (P05, P06, v1)
- 조 13.2.8: examples/physics/rendering/ce_rendering_registry.py (P07, v1)
- 조 13.2.7: examples/physics/rendering/ce_rendering_registry.py (P08, v1)
- 조 13.2.10: examples/physics/rendering/ce_rendering_registry.py (P09, P10, v1)
- 조 13.2.8: examples/physics/rendering/ce_rendering_derivations.py (P11, v2)
- 조 13.2.7: examples/physics/rendering/ce_rendering_planck_readout.py (P12, P13, v3)
- 조 13.2.7: examples/physics/rendering/ce_rendering_nu_ledger.py (P13, P14 갱신, v8)
- 조 13.2.7: examples/physics/rendering/ce_rendering_bao_ruler.py (P14, v4)
- 조 13.2.8: examples/physics/rendering/ce_rendering_spiral.py (P15, v5)
- 조 13.2.10: examples/physics/rendering/ce_rendering_growth.py (P16, v6)
- 조 13.2.5: examples/physics/rendering/ce_rendering_neutrino.py (P17–P19, v7)
- 조 13.2.2: examples/physics/rendering/ce_rendering_event_scale.py (P20, v7)
- 조 13.2.10: examples/physics/rendering/ce_rendering_vacuum_tilt.py (P21, P22, v9)
- 조 13.2.11: examples/physics/rendering/ce_rendering_probability_weight.py (P23, v10)
- 조 13.2.10: examples/physics/rendering/ce_rendering_probability_weight.py (P24, v10)
- 조 13.2.9: examples/physics/rendering/ce_rendering_gradient.py (P25, v11)
- 조 13.2.8: examples/physics/rendering/ce_rendering_open_predictions.py (P26, v12)
- 조 13.2.6: examples/physics/rendering/ce_rendering_open_predictions.py (P27, P28, v12)
- 조 13.2.4: examples/physics/rendering/ce_rendering_boundary.py (P29, v13)
- 조 13.2.4: examples/physics/rendering/ce_rendering_bool.py (P30, v13)
- 조 13.2.11: examples/physics/rendering/ce_rendering_ledger.py (P31, v14, 플랑크 해상도 가설 문단)
- 조 13.2.8: examples/physics/rendering/ce_rendering_o1_map.py (P32, v14)
- 조 13.2.9: examples/physics/rendering/ce_rendering_spread.py (P33, v14)
- 조 13.2.2: examples/physics/rendering/ce_rendering_e4_anchor.py (P34, v15)
- 조 13.2.2: examples/physics/rendering/ce_rendering_fp_ladder.py (P34 빠진 보정, v19)
- 조 13.2.6: examples/physics/rendering/ce_rendering_higgs_cosmos.py (P35, v16)
- 조 13.2.2: examples/physics/rendering/ce_rendering_distinction.py (P36, v16–v17)
- 조 13.2.8: examples/physics/rendering/ce_rendering_ladder.py (P37, v18)
- 조 13.2.8: examples/physics/rendering/ce_rendering_one_event.py (P37 LS 정리, v18)
- 조 13.2.2: examples/physics/rendering/ce_rendering_pole.py (P38, v19)
- 조 13.2.6: examples/physics/rendering/ce_rendering_mass_rate.py (P39, v20)
- 조 13.2.11: examples/physics/rendering/ce_rendering_horizon_pixel.py (P40, v21)
- 조 13.2.5: examples/physics/rendering/ce_rendering_nu_cosmo.py (P17 기각 판정, P41, v22)
- 조 13.2.5: examples/physics/rendering/ce_rendering_nu_fit.py (P41 압축 적합, v23)
- 조 13.2.5: examples/physics/rendering/ce_rendering_nu_lens.py (P41 렌즈 적합)
- 조 13.2.11: examples/physics/rendering/ce_rendering_joint_render.py (P42, v25)
- 조 13.2.8: examples/physics/rendering/ce_rendering_time_semantics.py (P43, v25)
- 조 13.2.11: examples/physics/rendering/ce_rendering_record_dynamics.py (P44, v26)

## A.3 동결과 해시 잠금

사전 등록 파일 v1–v26(`experiments/preregistration/rendering_predictions_v*.json`)은 예측 값과 기각 조건을 동결하고, 그때까지의 렌더링 모듈 해시를 잠근다. 잠긴 모듈을 고치려면 다음 판본을 등록하고 모든 변경을 그 판본의 변경 내역에 적어야 한다. 잠긴 모듈의 주석과 동결 파일의 문구에는 이 논문보다 앞선 지위 표기가 남아 있을 수 있다. 그런 차이는 부록 B의 해당 장에 적었다.
