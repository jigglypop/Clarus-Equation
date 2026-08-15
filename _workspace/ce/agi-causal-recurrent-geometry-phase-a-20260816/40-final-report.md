# 인과 재귀 기하 Phase A V1 최종 보고서

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-connectome-geometric-memory-20260816

Formal scope: synthetic known-identity development complete

Release: NOT CLEAN (combined-process validation-harness P1)

## 초록

Connectome·SCC·리만기하학 기반 기억 가설의 첫 검증 단위를 생물학적 기억이나 AGI 전체가 아니라, 관측된 문맥 아래의 개입-조건부 재귀 동역학 식별 문제로 좁혔다. 정규화된 합성 선형계에서 문맥별 전이 $A_z$와 공유 개입 행렬 $B$를 학습하는 R1을 단일 전이행렬을 쓰는 pooled R3와 사전등록된 development graph 24개에서 비교했다. 결합 설계행렬의 rank가 $Kn+m$일 때만 계수가 유일하게 식별된다는 조건부 정리, rank 부족 반례, unknown-mix similarity no-go를 코드와 독립 계산으로 함께 고정했다. 등록된 total Gaussian NLL에서 R1의 개선량은 평균 $30285.54$, graph-seed bootstrap 95% 구간 $[17565.59,49970.74]$였고, input-time shuffle penalty는 평균 $137135.54$, 구간 $[115336.16,159938.57]$였다. 이 결과의 지위는 synthetic development **[산출]**이며 strict 성능 정리, 외부 자료 재현 또는 AGI **[예측]**이 아니다. confirmation은 실행 가능한 holdout이 아니고, 단일 pytest 프로세스 결합 실행에는 test-harness P1이 남아 있다.

## 1. 서론

원 가설은 재귀 네트워크의 인과 구조, SCC 계층, 상태공간 기하와 기억 접근 비용 사이에 검증 가능한 연결이 있을 수 있다는 연구 프로그램이다. 선행 감사에서는 동일 edge semantics로 SCC를 반복하면 첫 condensation 뒤 DAG가 되어 비자명한 계층이 자동으로 생기지 않는다는 반례, unknown latent mixing 아래 exact support가 식별되지 않는다는 반례, 정적 SPD metric만으로 방향성과 부호를 복원할 수 없다는 no-go가 확인됐다. 따라서 Phase A는 이 부모 주장들을 되살리지 않고, 더 좁고 측정 가능한 질문을 택했다.

이번 질문은 다음과 같다. 완전히 관측된 정규화 좌표, 관측된 문맥 label, 알려진 개입 입력을 가진 유한 이산시간 선형계에서 문맥별 전이와 공유 개입 효과를 학습 자료만으로 복원할 수 있는가, 그리고 이 구조가 pooled transition보다 fresh development graph의 held-out 개입을 더 잘 예측하는가. 이 질문은 기억, 의식 또는 AGI 여부를 묻지 않는다. 성공하더라도 얻는 것은 합성 동역학 benchmark의 조건부 식별성과 development 비교뿐이다.

## 2. 정의와 표기

**[정의]** 상태 $x_t\in\mathbb R^n$, 개입 $u_t\in\mathbb R^m$, 관측된 문맥 $z_t\in\{0,\ldots,K-1\}$에 대해 Phase A 생성계를

$$
x_{t+1}=A_{z_t}x_t+Bu_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma^2I_n)
\tag{1}
$$

로 정의한다. $x_t$, $u_t$, $A_z$, $B$, $\epsilon_t$, $\sigma$는 manifest에 등록된 기준척도로 나눈 무차원 합성 좌표와 계수다. 이 무차원성은 식별성, 인과적 타당성 또는 물리적 실재성을 증명하지 않는다.

**[정의]** 관측은 $y_t=Cx_t+\nu_t$로 구분한다. `known_identity`에서는 $C=I$이고 선언된 선형 class 안의 계수 식별을 평가할 수 있다. `known_mask`에서는 알려진 좌표 부분공간 예측만 허용한다. `unknown_mix`에서는 관측 좌표계의 예측은 가능해도 latent exact edge와 coefficient error를 주장하지 않는다. anatomy, latent causal support, observed-coordinate predictive transition은 서로 다른 typed field다.

**[정의]** R1은 문맥별 $\widehat A_z$와 하나의 공유 $\widehat B$를 동시에 적합한다.

$$
\widehat x_{t+1}
=\widehat A_{z_t}x_t+\widehat Bu_t.
\tag{2}
$$

R3는 모든 문맥에 하나의 $\widehat A$를 쓰되 같은 공유 $\widehat B$와 ridge 값을 쓰는 pooled baseline이다. 두 model의 nominal 자유도는 각각 $n(Kn+m)$과 $n(n+m)$이며 차이는 $(K-1)n^2$다.

**[정의]** graph seed $s$의 primary endpoint는 같은 held-out batch와 같은 scorer-only $\sigma$를 사용한 total Gaussian NLL 차이다.

$$
\Delta_s
=\operatorname{NLL}_{\mathrm{R3},s}
-\operatorname{NLL}_{\mathrm{R1},s}.
\tag{3}
$$

$\Delta_s>0$이면 해당 graph에서 R1의 held-out density score가 더 작다. 통계 단위는 transition frame이 아니라 graph seed이며, bootstrap도 graph seed를 재표집한다.

## 3. 공리와 사전등록 선택

**[공리 A1: 모델 선택]** 유한 이산시간, fully observed `known_identity`, 관측된 문맥, context-specific $A_z$와 context-shared $B$, isotropic Gaussian generator noise를 Phase A V1의 합성 class로 택한다. 이 선택은 실제 신경계가 이 class에 속한다는 명제가 아니다.

**[공리 A2: 좌표 선택]** 합성 state와 intervention은 manifest의 양의 유한 reference scale로 정규화된 무차원 좌표다. state scale은 $n$개, input scale은 $m$개이며 Gaussian residual도 무차원이다.

**[공리 A3: 학습 경계]** learner는 training의 $(x_t,u_t,z_t,x_{t+1})$와 사전 고정 ridge만 받는다. truth $A_z,B$, held-out target, scorer $\sigma$, confirmation 정보는 learner API에 전달하지 않는다.

**[공리 A4: 난수와 split 선택]** graph, train trajectory, held-out trajectory, intervention, train noise, evaluation noise, shuffle, bootstrap은 SHA-256 domain-separated namespace를 사용한다. pilot, development와 confirmation 역할은 분리한다. 초기 development 후보가 focused test에 소비된 사실이 드러난 뒤 그 block 전체를 폐기하고, 결과와 무관한 hash-derived rotation-2 규칙으로 새 24개 development graph를 고정했다.

**[공리 A5: 채점 선택]** 두 arm은 같은 training batch, held-out batch, ridge와 manifest truth $\sigma$를 사용한다. $\sigma$는 scorer-only이고, 두 endpoint의 bootstrap은 등록된 동일 graph-resample index matrix를 공유한다. PA-H1과 PA-H2는 각각 mean, median, bootstrap 95% 하한이 모두 양수일 때만 development GO로 기록한다.

**[공리 A6: 실행 선택]** manifest와 네 required artifact hash를 고정한 뒤 development output을 evaluator 진입 전에 배타 예약하고, 등록 block을 한 번만 실행한다. confirmation은 `reservation_only`, `custody_unverified`, `not_executable_holdout`, `execution_authorized=false`로 제한한다.

## 4. 조건부 정리와 증명

### 4.1 공유-$B$ 식별 정리

**[정리]** (T1: 조건부 유한표본 식별) 각 training row에서 현재 문맥 block에만 $x_t^T$를 놓고 마지막 $m$개 열에 $u_t^T$를 놓은 설계행렬을 $\Phi\in\mathbb R^{N\times(Kn+m)}$라 하자. noiseless known-identity 선형 class에서 $(A_0,\ldots,A_{K-1},B)$가 자료로부터 유일하게 정해질 필요충분조건은

$$
\operatorname{rank}(\Phi)=Kn+m
\tag{4}
$$

이다.

**증명.** 계수들을 $W\in\mathbb R^{n\times(Kn+m)}$에 쌓으면 target matrix는 $Y=\Phi W^T$다. $\Phi$가 full column rank이면 각 target coordinate의 선형계가 유일해 $W$가 유일하다. 반대로 rank가 부족하면 $\Phi v=0$인 $v\ne0$이 존재한다. 임의의 $a\in\mathbb R^n$에 대해 $W'=W+av^T$는 $\Phi W'^T=\Phi W^T$를 만족하므로 계수는 유일하지 않다. 따라서 식 (4)는 필요충분하다.

상태 block만 모은 $D\in\mathbb R^{N\times Kn}$와 input matrix $U\in\mathbb R^{N\times m}$를 쓰면 $\Phi=[D\ U]$다. $P_D$를 $D$의 column space에 대한 직교 projector라 할 때

$$
\operatorname{rank}(\Phi)
=\operatorname{rank}(D)
+\operatorname{rank}\!\left((I-P_D)U\right).
\tag{5}
$$

문맥 $z$의 state row matrix를 $X_z$라 하면 $\operatorname{rank}(D)=\sum_z\operatorname{rank}(X_z)$다. 따라서 식 (4)는 모든 $X_z$가 rank $n$이고 residualized input이 rank $m$인 조건과 동치다. 동등하게 $U^T(I-P_D)U$가 양의 정부호여야 한다.

**[정리]** (T2: 문맥별 full rank만으로는 부족함) 모든 $X_z$가 rank $n$이어도 $U_z=X_zR_z$이면 input column은 $D$의 column space에 속해 $(I-P_D)U=0$이다. 이때 식 (5)의 마지막 rank는 0이고 $A_z$와 $B$ 사이에 동일 관측을 만드는 자유도가 남는다. 구현은 ridge가 수치해를 반환하더라도 이 경우 exact-edge certificate를 거부한다.

### 4.2 공통-scale NLL 항등식

**[정리]** (T3: 공통-scale NLL 항등식) held-out vector sample이 $N_*$개이고 residual scalar 수가 $D_*=N_*n$일 때 model $M$의 등록 score는

$$
\operatorname{NLL}_M
=\frac{D_*}{2}\log(2\pi\sigma^2)
+\frac{\operatorname{SSE}_M}{2\sigma^2}.
\tag{6}
$$

두 arm에 같은 $\sigma$와 같은 held-out rows를 쓰므로 정규화항은 상쇄되어

$$
\Delta_s
=\frac{\operatorname{SSE}_{\mathrm{R3},s}
-\operatorname{SSE}_{\mathrm{R1},s}}
{2\sigma^2}.
\tag{7}
$$

이 된다. 이는 직접 대수 전개다. scorer $\sigma$를 learner에 전달하거나 model별 test residual로 재추정하지 않는 이유도 이 비교를 고정하기 위해서다.

### 4.3 unknown-mix no-go

**[정리]** (T4: similarity 비식별) invertible $S$에 대해 $x'_t=Sx_t$, $A'=SAS^{-1}$, $C'=CS^{-1}$로 두면

$$
C'x'_t=Cx_t
\tag{8}
$$

이므로 두 latent system은 같은 관측열을 만든다. $A$와 $A'$의 zero pattern은 일반적으로 다를 수 있다. 따라서 unknown invertible mixing 아래 관측열만으로 latent exact support를 보편적으로 복원할 수 없다. 구현의 `unknown_mix` fixture는 support가 다른 두 system의 관측 trajectory가 byte-level로 같음을 재현하고 latent coefficient scoring을 거부한다.

## 5. 구현 산출

**[산출]** 구현 certificate에서 self-contained NumPy module은 generator-owned truth, learner-visible batch, frozen fit, evaluator-owned scorer를 분리한다. joint singular values, observed/required rank, 문맥별 state rank, residualized-input singular values와 rank tolerance를 결과에 남긴다. exact-edge 평가 허용 조건은 fit에 결박된 `known_identity`, declared linear class, full rank, finite valid inputs의 conjunction이다. 이 conjunction은 coefficient/support를 평가할 정의역이 열렸다는 뜻이며, noisy finite fit의 support가 자동으로 참이라는 뜻은 아니다.

**[산출]** 실행 무결성 감사에서 focused red-team은 total NLL 단위, categorical label 손실 변환, 숨은 bootstrap stream, one-shot race, config type coercion, cross-chart scoring, unclaimable coefficient error, confirmation schema, tautological replay와 equal-context kill-test 누락을 one-shot 전에 재현하고 닫았다. 새 development block은 manifest 밖에서 사전 사용되지 않았고, required artifact hash와 canonical manifest self-hash는 실행 전후 일치했다.

**[정의]** 초기 후보 `2001`--`2024`는 폐기 split `ABANDONED_PRE_REGISTRATION_TEST_CONTAMINATION`이다. 그 일부가 소형 focused fixture에 사용됐으므로 남은 seed까지 재사용하지 않았다. 이 block의 수치는 PA-H1 또는 PA-H2의 증거로 세지 않는다.

## 6. development 비교

**[산출]** 사전등록 synthetic development의 locked runner를 정확히 한 번 실행한 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| graph seed | $24$ |
| residual scalar / graph | $768$ |
| joint required/observed rank | $14/14$ (모든 graph) |
| 문맥별 state rank | $4$ (모든 문맥·graph) |
| residualized-input rank | $2$ (모든 graph) |
| R1 / R3 nominal dof | $56/24$ |
| $\Delta_s$ mean | $30285.541596706124$ |
| $\Delta_s$ median | $17949.904754418854$ |
| $\Delta_s$ graph-bootstrap 95% | $[17565.590274881404,49970.74491150572]$ |
| $\Delta_s\le0$ | $0/24$ |
| shuffle penalty mean | $137135.5433569638$ |
| shuffle penalty median | $125970.40032055078$ |
| shuffle graph-bootstrap 95% | $[115336.15701641495,159938.57170912955]$ |
| shuffle penalty $\le0$ | $0/24$ |

PA-H1과 PA-H2는 사전등록된 세 조건을 모두 만족해 development GO다. PA-I3의 observation/rank 경계도 모든 graph에서 충족됐다. 최대 coefficient error는 transition에서 약 $0.06924$, shared input에서 약 $0.00978$이었다. 이 값들은 고정된 $N_*$, $n$, $\sigma$와 generator heterogeneity에 의존하므로 다른 dataset의 보편 effect size로 읽을 수 없다.

**[산출]** Negative control에서 모든 $A_z$를 같게 만든 noiseless full-rank fixture에서는 R1과 R3의 held-out prediction과 NLL이 일치한다. 따라서 구현은 extra 문맥 자유도 자체를 자동 우위로 판정하지 않는다. input-time shuffle은 state, context, target과 held-out noise를 고정한 채 training input 대응만 깨며, 등록 development에서는 모든 graph에서 intact R1보다 나빴다.

## 7. 관측 비교

이번 run은 외부 관측자료를 사용하지 않았다. MICrONS, engram, H01, Cogitate 또는 행동·신경 개입 자료와 수치 비교하지 않았고, 따라서 생물학적 구조-기능 연결, 기억 회상, 의식 또는 AGI에 대한 관측 결론은 없다. 외부 자료의 범위와 한계는 선행 run의 source audit에 남아 있으며, 그 내용을 synthetic development 결과와 합치지 않는다.

## 8. 미완성 과제와 한계

**[미완성]** Confirmation manifest marker는 `reserved_unopened`지만 검증된 custodian, raw-seed reveal artifact 또는 미래 공공 난수 선택 규칙이 없다. 따라서 현재 지위는 `reservation_only`, `custody_unverified`, `not_executable_holdout`이다. 별도 run에서 custody가 검증된 protocol을 사전등록하기 전에는 blinded confirmation을 실행할 수 없다.

**[미완성]** Release validation에서 focused suite와 legacy causal suite를 별도 pytest process로 실행하면 각각 41개와 30개가 통과한다. 같은 process에서 함께 collect하면 legacy module이 parent package를 먼저 import해 focused test 두 개의 절대 `sys.modules` 부재 assertion이 실패하며 결과는 `69 passed, 2 failed`다. clean subprocess sentinel과 loader 호출 전후 snapshot은 신규 parent import가 없음을 확인했으므로 science result를 바꾸는 failure는 아니지만, combined/full-suite release gate는 깨끗하지 않다. 잠긴 V1 test를 사후 수정하면 one-shot provenance가 깨지므로 V1.1에서 assertion을 before/after snapshot 또는 subprocess sentinel로 교체해야 한다.

**[미완성]** 외삽 범위인 nonlinear dynamics, hidden context, partial observation, unknown mixing, state-dependent intervention, non-Gaussian noise와 distribution shift에서는 T1의 설계와 결과가 그대로 성립하지 않는다. 특히 known-mask에는 관측 subspace의 closure 또는 lumpability 조건이 추가로 필요하다. 실제 connectome에는 anatomy, dynamics, learning, node intervention와 partial-cue recall을 한 dataset에서 공동 측정하는 protocol이 필요하다.

**[미완성]** 기억과 AGI bridge에서 Phase A는 transition identification만 다룬다. SCC hierarchy, controllability Gramian의 cue energy, manifold separation, 기억 접근·재고정, self-model과 consciousness는 별개의 계약과 kill test가 필요하다. 이번 GO를 그 bridge의 증명이나 AGI evidence로 부르지 않는다.

## 9. 재현성

잠긴 implementation, test, runner와 manifest는 다음 경로에 있다.

- `reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py`
- `tests/test_causal_recurrent_geometry_benchmark.py`
- `examples/agi/causal_recurrent_geometry_development_run.py`
- `experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json`
- `_workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816/artifacts/development-results.json`

focused와 legacy 검증은 서로 다른 process에서 실행한다.

```text
.venv\Scripts\python.exe -m pytest tests/test_causal_recurrent_geometry_benchmark.py -q --basetemp .pytest_tmp_phasea_focused_verify
.venv\Scripts\python.exe -m pytest tests/test_sparse_causal_bridge.py tests/test_latent_causal_bridge.py tests/test_nested_scc_memory_benchmark.py -q --basetemp .pytest_tmp_phasea_legacy_verify
```

수학 verifier와 정적 검사는 다음과 같다.

```text
.venv\Scripts\python.exe _workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816/artifacts/verify_phase_a_math.py
.venv\Scripts\ruff.exe check reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py examples/agi/causal_recurrent_geometry_development_run.py
.venv\Scripts\python.exe -m compileall -q reality_stone/python/reality_stone/clarus/causal_recurrent_geometry_benchmark.py tests/test_causal_recurrent_geometry_benchmark.py examples/agi/causal_recurrent_geometry_development_run.py
```

development result SHA-256은 `7c4b9eb9ba08bed4cfc192262cc47c4a1cf56326526b3848cc6e4d5d89780df9`다. development runner는 이미 한 번 실행됐고 output path가 배타 예약돼 있으므로 **다시 실행하지 않는다**.

## 10. 참조

새 외부 관측값이나 문헌을 사용하지 않았으므로 이 light follow-up의 source lane은 `Status: SKIPPED (새 외부 관측·인용 없음)`이다. 내부 근거는 2026-08-16에 고정된 `00-contract.md`, `11-math.md`, `12-routes.md`, `20-audit.md`, `30-implementation.md`, `31-validation.md`, `artifacts/split-contamination-ledger.md`, `artifacts/post-implementation-review.md`와 one-shot result다. 생물학 dataset과 선행 문헌의 1차 출처 경계는 predecessor `_workspace/ce/agi-connectome-geometric-memory-20260816/10-sources.md`에서 관리한다.
