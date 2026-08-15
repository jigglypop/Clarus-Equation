# CE 우주론 전면 닫힘 형식 지위 감사

Status: COMPLETE

Gate: PASS

감사 기준일: 2026-08-16  
역할: `ce-status-auditor`  
제품·정본 수정: 없음

## 0. Gate의 정확한 뜻

이 `PASS`는 다음 한 가지를 뜻한다.

> 이번 run의 활성 결론 집합에서 완전 반례가 맞은 부모 route를 모두 제외했고,
> 남은 주장마다 정의·정리·공리·산출·경험식·미완성 지위가 근거와 일치하여
> **열린 P0가 0개**다.

다음은 뜻하지 않는다.

- CE 우주론의 U1--U8 물리 목표가 모두 닫혔다는 뜻이 아니다.
- 현재 밀도분율, $H_0$, 원시 스펙트럼 또는 진공 절대척도가 제1원리에서
  예측됐다는 뜻이 아니다.
- verifier의 exit 0가 관측 적합, release 또는 blind confirmation을 뜻하지 않는다.
- 아직 제품·정본에 남아 있는 과거 라벨이 과학적으로 승인됐다는 뜻이 아니다.

사용자의 목표는 route와 분리한다. `TARGET-HYPOTHESIS`인 U1--U8은 삭제하거나
내리지 않고 `[미완성]`으로 보존한다. 완전 반례는 직접 맞은 `ROUTE`의 활성
정리·산출·예측 지위만 닫는다. 과거 계산과 값은 이름 붙은 historical 또는
compatibility boundary로 남긴다.

필수 입력은 모두 EOF까지 확인했다.

| 입력 | 상태 | 감사 판정 |
|---|---|---|
| `00-contract.md:1-156` | `Status: COMPLETE` | U1--U8 목표, 비파괴 통합, 성공·금지 조건 유효 |
| `10-sources.md:1-175` | `Status: COMPLETE` | 공식 출처와 CE bridge를 분리했고 두 P0 및 holdout 0건을 보존 |
| `11-math.md:1-867` | `Status: COMPLETE` | 최소 주장, 반례 범위, 수치·차원 경계를 전부 판정 |
| `12-routes.md:1-271` | `Status: COMPLETE` | 각 미완성 목표에 구조적으로 다른 대안과 kill test가 존재 |

아래 표에서 `P0-CLOSED`는 소스 파일을 삭제했다는 뜻이 아니다. 반례가 있는
강한 부모 문장을 이 run의 활성 결론에서 제외하고, 가능한 가장 좁은 역사적
경계 또는 정리만 보존해 **열린 P0가 아니게 했다**는 뜻이다.

경로 표의 짧은 근거 이름은 다음 run-relative 파일을 뜻한다.

- `M`: `11-math.md`
- `R`: `12-routes.md`
- `S`: `10-sources.md`
- `V`: `artifacts/canonical-version-map.md`
- `D`: `artifacts/density-dark-alternative-derivations.md`
- `T`: `artifacts/transient-transition-action.md`
- `B`: `artifacts/background-h0-forward-route.md`
- `P`: `artifacts/primordial-lambda-alternative-routes.md`

## 1. Claim ID별 현재/실제 지위

### 1.1 U1--U2: 원장과 abundance

| Claim ID / 종류 | 파일:줄 | 현재 주장·지위 | 실제 지위 | P / 처분·보존 범위 |
|---|---|---|---|---|
| `T-U1-CANON` TARGET | `00-contract.md:31-38`; `M:63`; `V:11-28` | 판본을 하나의 정본으로 통합 | **[미완성]** | P1. inventory는 닫혔고 실제 registry/consumer migration은 U8 구현 전 |
| `R-U1-Q-DEF` ROUTE | `M:64`; `V:23-28,92-93` | $q$를 survival로도 부르는 혼용 | **[정의]** $q=q_{ext}$는 소멸, $s=1-q$는 생존 | P2. legacy 문구만 교정; 수치·alias 삭제 금지 |
| `R-U1-Q-THM` ROUTE | `M:65,157-171` | 작은 고정점과 안정도 | **[정리]** | PASS. $D>1$에서 $(0,1/D)$의 유일한 작은 근과 $Dq<1$ 보존 |
| `R-U1-CORE` ROUTE | `M:66`; `V:71-75` | core가 $alpha_s\to s_W^2\to\delta\to D\to q$를 고름 | **[경험식]** | P1. scale/scheme과 $s_W^2=4\alpha_s^{4/3}$은 모형·외부 입력 |
| `R-U1-EXACT` ROUTE | `M:67,173-184` | full-precision chain | **[산출]** | PASS. `CE_CORE_EXACT_V1`로만 활성 |
| `R-U1-LEGACY` ROUTE | `M:68,183-197`; `V:76-86` | rounded $D/q$와 runtime triplet이 exact처럼 혼재 | **[공리: 호환 원장]** | P2. `LEGACY_DELTA_5DP_V1`, `LEGACY_ROUNDED_RUNTIME_V1`로 별도 보존 |
| `R-U1-OBS` ROUTE | `M:69`; `S:16-20`; `V:137-147` | theory/runtime/observation을 같은 상수처럼 사용 | **활성 동일시 제외** | P0-CLOSED. 관측은 별도 manifest reference만 허용 |
| `T-U2-ABS` TARGET | `00-contract.md:40-49`; `M:75` | 관측 밀도 없이 absolute baryon abundance | **[미완성]** | P1. mass, total yield, entropy, freeze surface, $H_*$ 필요 |
| `R-U2-DIRECT` ROUTE | `M:76`; `V:95-97`; `D:395-467` | $q$ 자체가 오늘의 $\Omega_b$라는 보편 유도 | **[공리: `LEGACY_DIRECT_READOUT_V1`]**만 보존 | P0-CLOSED. 정리·예측 부모만 제외; 목표와 역사값은 보존 |
| `R-U2-GW` ROUTE | `M:77,203-226`; `D:259-289` | extinction-conditioned offspring law | **[정리]** $K\mid E\sim\mathrm{Poisson}(Dq)$ | PASS. species/density bridge는 포함하지 않음 |
| `R-U2-COMP` ROUTE | `M:78,221-243`; `D:291-367` | aggregate descendant fraction $Dq$ | **[산출: 조건부]** | P1. equal-energy conserved relic와 aggregate detector가 추가 공리 |
| `R-U2-REACT` ROUTE | `M:79,245-287`; `S:34-38` | two-current EFT의 relaxation/entropy | **[산출: effective closure]** | P1. microscopic collision 또는 SK action 없음 |
| `R-U2-FREEZE` ROUTE | `M:80,289-301`; `S:44-45` | freeze 뒤 yield 식 | **[산출: 식]** | P1. $Y_X$, 질량, stoichiometry, dilution을 고정하지 않음 |
| `R-U2-SPIN` ROUTE | `M:81,303-322`; `T:73-124` | $(Dq)(1/D)=q$가 density bridge까지 닫음 | **[산출: 대수]** | P1. $1/D\leftrightarrow\Omega_m$의 action/stress lemma는 별도 |

### 1.2 U3--U4: dark sector, 배경과 성장

| Claim ID / 종류 | 파일:줄 | 현재 주장·지위 | 실제 지위 | P / 처분·보존 범위 |
|---|---|---|---|---|
| `T-U3-SPLIT` TARGET | `00-contract.md:51-58`; `M:87` | 하나의 공변 이론이 dark split과 perturbation을 고정 | **[미완성]** | P1. action coupling, full modes, transition이 필요 |
| `R-U3-RLO` ROUTE | `M:88`; `R:79-82` | $R_D=\alpha_sD$ 또는 과거 분할비가 자연의 dark ratio | **[경험식]** | P1. action에서 coupling을 유도하지 못함; named alternative로 보존 |
| `R-U3-D1` ROUTE | `M:89,414-454`; `S:51-54` | interacting-vacuum background fixed point | **[산출: background]** | P1. $u^\mu$, $\delta Q^\mu$, perturbation 안정성 미고정 |
| `R-U3-D2` ROUTE | `M:90,456-504`; `R:80` | conformal scalar dark-only scaling point/Jacobian | **[산출: 조건부]** | P1. 두 coupling은 target 역산, fifth force/full modes 미검증 |
| `R-U3-ALLFIX` ROUTE | `M:91,506-521`; `D:977-1003` | conserved baryon까지 nonzero인 영구 가속 fixed point | **활성 route 제외** | P0-CLOSED. $d\log\Omega_b/d\log a=3w_{tot}$; transient 목표는 보존 |
| `R-U3-TRANSIENT` ROUTE | `M:92`; `R:137-151`; `T:916-954` | action-defined $\Sigma_*$에서 읽고 전방 적분 | **[미완성]** | P1. 유일 clock, stress, freeze 및 observer epoch 필요 |
| `T-U4-KERNEL` TARGET | `00-contract.md:60-67`; `M:98`; `B:525-539` | radiation 포함 공용 FLRW/거리/성장 kernel | **[미완성: 구현]** | P1. 수학·scratch는 닫혔고 product integration/transfer backend가 남음 |
| `R-U4-FLRW` ROUTE | `M:99,552-582`; `B:98-174` | radiation 포함 $E(a)$, Ricci와 세 극한 | **[정리]** | PASS. $R/H^2=12-9\Omega_m-12\Omega_r$, 극한 $(0,3,12)$ |
| `R-U4-RICCI-OLD` ROUTE | `M:100,593-599`; `B:136-174,291-298` | legacy $12-9\Omega_m$이 radiation/running에도 exact | **기존 산출 제외; historical witness** | P0-CLOSED. radiation, $\epsilon'$ 누락 범위만 제외 |
| `R-U4-QUAD-OLD` ROUTE | `M:101,593-609`; `B:282-301` | 기존 Simpson이 nonuniform/even grid를 처리 | **기존 산출 제외; compatibility snapshot만** | P0-CLOSED. 마지막 구간 누락과 $+33.63\%$ 반례 |
| `R-U4-GROWTH-OLD` ROUTE | `M:102,584-609`; `B:250-280` | 평균 step solver가 arbitrary grid API를 만족 | **기존 산출 제외; uniform-$N$ snapshot만** | P0-CLOSED. warped grid 최대 $278.97\%$ 반례 |
| `R-U4-REPLACE` ROUTE | `M:103,601-609`; `B:458-490` | interval-local quadrature/local-step RK4 | **[산출]** | PASS. analytic/Heath/grid refinement까지; product 적용은 T-U4와 별도 |

### 1.3 U5--U6: $H_0$, 원시 스펙트럼과 절대척도

| Claim ID / 종류 | 파일:줄 | 현재 주장·지위 | 실제 지위 | P / 처분·보존 범위 |
|---|---|---|---|---|
| `T-U5-H0` TARGET | `00-contract.md:69-75`; `M:109`; `B:24-28` | physical early inputs와 likelihood에서 $H_0$ 식별 | **[미완성]** | P1. 현재 physical posterior 없음 |
| `R-U5-TOY` ROUTE | `M:110,627-634`; `B:304-315` | legacy toy가 baryon-aware CMB readout | **[경험식: `HISTORICAL_TOY`]**만 보존 | P0-CLOSED. $\omega_b$ 100배 변화에도 output 동일; 과학 readout 제외 |
| `R-U5-FULL` ROUTE | `M:111`; `R:170` | CLASS/CAMB full spectra+likelihood | **[미완성]** | P1. 최우선 route이나 adapter/likelihood 실행 없음 |
| `R-U5-COMP` ROUTE | `M:112,636-648`; `R:171` | supplied $z_*$ compressed solver | **[산출: synthetic]** | P1. independent-grid 및 full-solver calibration 전 conditional |
| `R-U5-IDL` ROUTE | `M:113,650-658`; `R:172` | uncalibrated BAO+SN의 식별성 | **[정리]** | PASS. 독립 $r_d$ 없이는 $H_0r_d$만 식별 |
| `T-U6-PRIM` TARGET | `00-contract.md:77-86`; `M:119` | 같은 action에서 $A_s,n_s,r$ 공동 산출 | **[미완성]** | P1. exact MS modes, CE scale, reheating 필요 |
| `R-U6-PROJECT` ROUTE | `M:120`; `P:8-25` | 관측에 가까운 projector가 사전 예측 | **[경험식]** | P1. target-aware candidate selection; `[예측]` 아님 |
| `R-U6-R2` ROUTE | `M:121,662-698`; `P:27-85` | Starobinsky slow-roll $n_s,r,A_s$ | **[산출: 조건부]** | P1. $A_s$가 $M$을 calibration하고 $N_*$는 reheating 의존 |
| `T-U6-LAMBDA` TARGET | `00-contract.md:82-86`; `M:122`; `P:249-268` | 관측 scale 없이 late vacuum 절대척도 선택 | **[미완성]** | P1. unique branch/coefficient와 radiative stability 없음 |
| `R-U6-IDENTITY` ROUTE | `M:123,721-756`; `S:123-129` | 동일 horizon convention의 entropy--Friedmann 관계 | **[정리]** | PASS. one-scale identity; 독립 예측 두 개로 계수 금지 |
| `R-U6-PHASE-VAR` ROUTE | `M:124,700-719`; `P:87-158` | quadratic $s$-flow가 phase law를 정지해로 가짐 | **[산출: 변분 존재구성]** | P1. $K,\kappa,s_0$의 미시 기원·유일성 없음 |
| `R-U6-PHASE-H0` ROUTE | `M:125,721-756`; `S:127-129` | true de Sitter entropy의 $H$를 무표시로 현재 $H_0$로 읽음 | **활성 동일시 route 제외** | P0-CLOSED. apparent-horizon boundary와 true-dS route를 분리 보존 |
| `R-U6-RG/4F/STOCH` ROUTE | `M:126,758-770`; `R:195-198` | 각 mechanism이 CE scale을 유일 선택 | **[미완성]** | P1. field content, flux branch, coefficient/sign/measure가 자유 |

### 1.4 U7: provenance와 blind 판정

| Claim ID / 종류 | 파일:줄 | 현재 주장·지위 | 실제 지위 | P / 처분·보존 범위 |
|---|---|---|---|---|
| `T-U7-PROV` TARGET | `00-contract.md:88-94`; `M:132`; `V:223-227` | 공식 release/covariance/hash/snapshot 원장 | **[미완성: 구현]** | P1. inventory 완료, machine manifest migration 전 |
| `R-U7-HYBRID` ROUTE | `M:133,138-150`; `S:16-20` | hybrid tuple/식별 불가 DESI row를 공식 posterior로 사용 | **활성 관측 증거에서 제외** | P0-CLOSED. 원문 값은 historical display로만 보존 |
| `R-U7-DESI` ROUTE | `M:134`; `S:137-143` | frozen vector와 SPD covariance의 이차형식 | **[산출: exploratory]** | P1. asset hash, parser-wide SPD, model provenance 필요 |
| `R-U7-HOLDOUT` ROUTE | `M:135,148-150`; `S:18,143` | 현재 manifest가 independent confirmatory holdout | **[미완성]** | P1. `unassigned/NOT_READY`, qualifying holdout 0건 |
| `R-U7-PRED` ROUTE | `M:136`; `R:204-214` | 현재 수치 근접도가 blind prediction | **활성 blind 결론 제외** | P0-CLOSED. target-aware/exploratory 비교만 보존 |

### 1.5 U2/U3 전이 action의 세부 분해

| Claim ID / 종류 | 파일:줄 | 현재 주장·지위 | 실제 지위 | P / 처분·보존 범위 |
|---|---|---|---|---|
| `R-U23-SPIN-STAT` ROUTE | `T:245-280`; `M:324-337` | $v_D''=0$만으로 physical spinodal | **[정리: 선택한 tilted free energy]** $f_y=f_{yy}=0$ | PASS. $y_*=1/D$, $h_*=D-1-\log D$; 단순 inflection 해석은 제외 |
| `R-U23-SPIN-FRACTION` ROUTE | `T:282-353`; `M:339-363` | spinodal composition이 자동 full $\Omega_m$ | **[산출: 조건부 bridge]** | P1. cold/comoving/no-entrainment/equal partial energy/two-sector saturation 필요 |
| `R-U23-SPIN-FREEZE` ROUTE | `T:355-373`; `M:365-373` | finite cooling이 exact $1/D$에서 자동 freeze | **[미완성]** | P1. critical slowing 때문에 constraint, freeze field 또는 계산된 offset 필요 |
| `R-U23-SPEC-B1` ROUTE | `T:377-520`; `R:84-134` | matter--vacuum subsystem spectator | **[산출: target-engineered 존재구성]** | P1. crossing은 유일하지만 $\Omega_b=q(1-\Omega_r)$; loop/fifth-force/stress 미검증 |
| `R-U23-SPEC-B2` ROUTE | `T:522-650`; `R:115-125` | full-density spectator | **[산출: target-engineered 존재구성]** | P1. $\Omega_b=q$는 exact이나 root가 보통 0/2개; memory/oriented clock 필요 |
| `R-U23-SCALE` ROUTE | `T:485-502`; `M:398-410` | phase/de Sitter, transition, present scales를 하나로 둠 | **[정의]+[산출: 구분]** | PASS. $H_L(=H_\Lambda\text{ for }V_L)$, $H_*$, $H_0$는 일반적으로 서로 다른 양이며 동일시 근거가 없음 |

### 1.6 U8: 비파괴 통합

| Claim ID / 종류 | 파일:줄 | 현재 주장·지위 | 실제 지위 | P / 처분·보존 범위 |
|---|---|---|---|---|
| `T-U8-INTEGRATE` TARGET | `00-contract.md:96-103`; `V:299-314` | 정본·코드·검증 진입점 통합 | **[미완성]** | P1. 이 감사는 통합을 승인할 뿐 실행하지 않음 |
| `R-U8-ORDER` ROUTE | `V:229-277`; `R:216-246` | registry-first staged migration | **[공리: 통합 정책]** | 승인. baseline→registry→alias→consumer의 순서를 강제 |
| `R-U8-ALIASES` ROUTE | `V:236-255`; `R:29-34` | 과거 이름을 즉시 제거하거나 exact로 조용히 교체 | **[공리: compatibility boundary]** | 승인. alias·characterization test·named historical config를 유지 |
| `R-U8-FAILCLOSED` ROUTE | `00-contract.md:98-103`; `V:268-273` | 실패를 exit 0 성공으로 숨기지 않는 단일 gate | **[미완성: 구현]** | P1. 제품 CLI와 manifest/status gate를 실제 연결해야 함 |

## 2. P0 폐쇄 원장

아래 열 건은 강한 부모 형태를 활성 결론에서 제외하여 닫았다. 재개 조건을
충족하지 않은 채 같은 뜻을 다른 이름으로 되살리면 즉시 `Gate: REVISE`다.

| Closure ID | 해당 Claim | 완전 반례·결정적 경계 | 제외 범위 | 보존/재개 조건 | 상태 |
|---|---|---|---|---|---|
| `CL-P0-01` | `R-U1-OBS` | exact/legacy/runtime/observation은 precision·role·covariance가 다름 (`V:11-28,67-86`) | 같은 key/default의 무표시 동일시 | typed role/model/source registry와 observation manifest | CLOSED-EXCLUDED |
| `CL-P0-02` | `R-U2-DIRECT` | 고정점에는 current, stress, yield, critical-density 사상이 없음 (`D:248-367`) | $q\to\Omega_b$의 정리·예측 지위 | named legacy axiom은 보존; 새 cascade/action/yield가 증명되면 별도 route 재개 | CLOSED-NARROWED |
| `CL-P0-03` | `R-U3-ALLFIX` | conserved baryon에 $d\log\Omega_b/d\log a=3w_{tot}$ (`M:506-521`) | 세 nonzero fraction의 영구 가속 fixed point | action-defined transient surface와 이후 전방 적분 | CLOSED-EXCLUDED |
| `CL-P0-04` | `R-U4-RICCI-OLD` | radiation 극한 오차가 거의 12, running $\epsilon'$ 누락 (`M:573-599`) | legacy Ricci의 과학 route | historical toy 보존; exact trace/kinematic 식으로 교체 | CLOSED-EXCLUDED |
| `CL-P0-05` | `R-U4-QUAD-OLD` | nonuniform $+33.63\%$, even-grid 마지막 interval 누락 (`M:593-609`) | 해당 integrator 의존 산출 | old snapshot 보존; interval-local rule 통과 후 새 route | CLOSED-EXCLUDED |
| `CL-P0-06` | `R-U4-GROWTH-OLD` | warped grid 최대 $278.97\%$ (`M:593-609`) | arbitrary-grid 지원 주장 | uniform-$N$ snapshot 보존; local-step solver로 교체 | CLOSED-EXCLUDED |
| `CL-P0-07` | `R-U5-TOY` | $\omega_b$를 100배 바꿔도 $\theta_*$가 bit-identical (`M:627-634`) | baryon-aware physical $H_0$ readout | historical toy 보존; all-input-active full/compressed route | CLOSED-NARROWED |
| `CL-P0-08` | `R-U6-PHASE-H0` | $H_\Lambda^2=\Omega_\Lambda H_0^2$이며 horizon/epoch가 다름 (`M:721-756`; `S:127-129`) | true dS $H$와 current $H_0$ 동일시 및 one identity를 두 예측으로 계수 | current apparent-horizon boundary 또는 true-dS route를 명시적으로 분리 | CLOSED-EXCLUDED |
| `CL-P0-09` | `R-U7-HYBRID` | 공식 DR2는 $\Omega_\Lambda=0.6973\pm0.0036$; hybrid tuple에는 단일 covariance 없음 (`S:16-20`) | 잘못 식별된 관측 row/hybrid posterior의 증거 지위 | 원문 historical display; single official release/model/chain만 활성 | CLOSED-EXCLUDED |
| `CL-P0-10` | `R-U7-PRED` | independent holdout 0건, 이미 본 자료·후보 선택 (`S:18,143`; `M:148-150`) | 현재 근접도를 blind prediction이라 부르는 결론 | freeze 뒤 독립 release/object/covariance와 사전 kill rule | CLOSED-EXCLUDED |

열린 P0 수: **0개**.

## 3. U1 통합 승인 순서

U1은 “모든 숫자를 하나로 덮어쓰기”가 아니라 **한 registry 안에서 서로 다른
typed quantity를 병존시키는 통합**만 승인한다. 순서를 바꾸거나 compatibility
alias를 먼저 제거하는 구현은 승인하지 않는다.

1. 현재 exact, legacy, runtime, 과거 density configuration, observation 출력을
   characterization fixture로 먼저 동결한다.
2. 소비자를 바꾸기 전에 `CE_CORE_EXACT_V1`, `LEGACY_DELTA_5DP_V1`,
   `LEGACY_ROUNDED_RUNTIME_V1`, named density configurations와 `RouteClaim`을
   registry에 추가한다.
3. 관측 수치는 registry literal이 아니라 versioned observation manifest ID로만
   참조한다. hybrid와 provenance 미확정 row는 historical/excluded로 남긴다.
4. `ACTIVE_RATIO`, `STRUCT_RATIO`, `BACKGROUND_RATIO`, legacy contraction과
   epsilon 이름은 compatibility alias로 유지한다. $q$를 survival alias로 만들지
   않는다.
5. exact solver와 rounded-$D$ legacy solver를 병렬로 두고 residual, precision,
   model ID를 직렬화한다.
6. target, fixed-point theorem, legacy direct-readout axiom, conditioned composition,
   current/freeze route를 서로 다른 Claim ID로 유지한다.
7. ratio audit → residual → cosmology CLI → runtime 순으로 소비자를 한 번에 하나씩
   전환하고 각 단계에서 old/new parity를 검사한다.
8. 공용 FLRW kernel은 exact radiation/trace/local-grid 식을 사용한다. legacy
   LO/SFE/phase/H0 toy는 이름 붙은 alternative로 보존한다.
9. 관측 manifest에는 release/model/units/source/hash/covariance/blind role을 필수로
   둔다. 공식 source가 없는 row는 score에서 fail-closed한다.
10. 정본 문서는 새 원장을 참조하되 과거 문서를 광범위 삭제·이동하지 않는다.
11. q/s 반전, exact/legacy 혼용, raw/normalized 혼용, model ID 부재,
    source/covariance 부재, target-aware prediction 라벨을 하나의 fail-closed gate로
    막는다.
12. deprecation/removal은 모든 consumer가 전환되고 최소 두 compatibility release를
    거친 뒤에도 별도 승인 대상으로 남긴다.

이 순서의 형식 상태는 `APPROVED-FOR-IMPLEMENTATION`이다. 실제 migration이 이미
완료됐다는 뜻은 아니다.

## 4. 살아 있는 수학과 물리 경계

### 4.1 바로 보존·구현 가능한 좁은 결과

1. **[정의]/[정리]** $q$는 extinction, $1-q$는 survival이며, 작은 고정점은
   유일하고 $Dq<1$이다.
2. **[정리]** conditioned Poisson law는 평균 $Dq$이고, aggregate equal-node-energy
   detector 아래 descendant fraction은 조건부로 $Dq$다.
3. **[정리: 선택한 free energy]** physical stationary spinodal은
   $f_y=f_{yy}=0$이며 $y_*=1/D$, $h_*=D-1-\log D$다.
4. **[산출: 제한 조건]** cold/comoving/equal-partial-energy/two-sector saturation이면
   spinodal의 current composition은 energy fraction $1/D$가 된다.
5. **[정리]/[산출]** radiation 포함 FLRW Ricci, interval-local quadrature,
   local-step growth와 식별성 null test를 구현할 수 있다.
6. **[정리]** 같은 horizon convention의 entropy--Friedmann 관계는 one-scale
   identity다.

### 4.2 승격하면 안 되는 경계

- conditioned branching theorem은 species/current/yield 정리가 아니다.
- stationary spinodal의 위치는 finite-rate exact freeze 정리가 아니다.
- subsystem spectator의 unique crossing은 $\Omega_b=q$가 아니라
  $q(1-\Omega_r)$를 준다.
- full-density spectator의 exact $\Omega_b=q$는 unique surface를 주지 않는다.
- $H_L$ 또는 $H_\Lambda$, transition scale $H_*$, present $H_0$ 사이에는 action과
  전방 진화를 건너뛸 수 있는 항등식이 없다. 일반적으로 $H_L\ne H_*\ne H_0$이며,
  우연한 수치 일치도 동일 정의를 뜻하지 않는다.
- corrected numerical solver는 full Einstein--Boltzmann likelihood가 아니다.
- source provenance가 확인됐다는 사실은 CE bridge가 증명됐다는 뜻이 아니다.

## 5. P1/P2와 목표별 재개 조건

| 목표 | 현재 최선 활성 경로 | 남은 P1 | 재개 조건 / kill 경계 |
|---|---|---|---|
| U1 원장 | typed registry + compatibility facade | actual migration | role/model/source 없는 scientific output을 모두 거부하고 parity 통과 |
| U2 abundance | reacting current와 conditioned cascade의 경쟁 | species map, microscopic rate, total yield, entropy | 관측 density 없이 current·freeze·$Y_i$·critical normalization을 한 action에서 계산 |
| U3 dark split | conformal scalar 또는 transient surface | UV coupling, perturbation, unique transition | background+full modes 안정, fifth-force/growth kill test, observer epoch 미주입 |
| U4 background | exact kernel + external CLASS/CAMB reference | product integration, early transfer | all-grid analytic/Heath/two-solver error budget 통과 |
| U5 $H_0$ | full likelihood primary, compressed/inverse-ladder crosscheck | recombination, nuisance, covariance, adapter | all physical inputs active, single released likelihood, null calibration test 통과 |
| U6 primordial | Starobinsky exact MS route | $M$ generator, reheating, exact modes | $A_s$ target 없이 scale을 고정하고 $A_s,n_s,r$ 공동 검증 |
| U6 vacuum | RG/four-form/stochastic/separated horizon routes | coefficient, branch, radiative stability, epoch | $H_0,\Omega_\Lambda$ 미주입 selection law와 독립 cross-output |
| U7 blind | immutable manifest + future/independent split | qualifying holdout가 아직 0 | freeze 뒤 최초 공개, 독립 object/likelihood/covariance와 사전 kill rule |
| U8 integration | U1 승인 순서 + fail-closed CLI | code/docs/tests/manifest migration | false status 문자열·hybrid·silent fallback이 scientific PASS를 만들지 못함 |

P2는 네 묶음이다.

1. $q_{ext}$와 $s_{branch}=1-q_{ext}$의 의미 반전.
2. exact $q$, rounded-$D$ $q$, display/runtime `0.0487`의 무표시 혼용.
3. raw triplet 합 `1.0001`과 flat-normalized background의 혼용.
4. current apparent horizon, asymptotic de Sitter horizon, transition 및 observer
   epoch의 $H$ label 혼용.

## 6. 수량 집계

이 감사가 고정한 최소 Claim row는 **54개**다. 행별 실제 지위는 중복 없이
다음과 같다.

| 실제 지위 묶음 | 수 |
|---|---:|
| `[정의]` | 1 |
| 승인된 `[정리]` | 6 |
| 조건부·수치 `[산출]` | 16 |
| `[공리]`·`[경험식]`·호환/통합 정책 | 8 |
| TARGET 또는 ROUTE `[미완성]` | 15 |
| 활성 결론에서 제외된 route | 8 |
| 합계 | 54 |

우선순위 회계는 다음과 같다.

- `P0-CLOSED` 부모 형태: **10개** — 위 8개 제외 route와, historical boundary로만
  좁힌 direct-density/H0-toy 2개.
- 열린 P0: **0개**.
- P1 row: **31개**.
- P2 row: **2개**; 별도로 네 종류의 표기 부채가 여러 consumer에 분포한다.
- 좁은 명제 자체에 우선 결함을 찾지 못한 row: **11개**.
- 활성 `[예측]`: **0개**.
- 독립 confirmatory holdout: **0개**.

## 7. 최종 판정과 종료 체크

형식 gate는 `PASS`다. 이유는 반례가 맞은 route의 강한 부모 결론을 모두
`CLOSED-EXCLUDED` 또는 `CLOSED-NARROWED`로 처리했고, 목표 가설은 대체 route와
구체적 재개 조건을 가진 `[미완성]`으로 보존했기 때문이다.

동시에 다음 상태는 분명히 구분한다.

- **Formal status gate:** PASS
- **U1 implementation authorization:** APPROVED-FOR-IMPLEMENTATION
- **Physical full closure:** INCOMPLETE
- **Observational prediction/confirmation:** NONE
- **Release gate:** NOT READY

- [x] `00/10/11/12` 네 입력이 모두 COMPLETE인지 확인했다.
- [x] 54개 최소 주장에 Claim ID, 현재/실제 지위와 파일:줄 근거를 부여했다.
- [x] TARGET-HYPOTHESIS를 내리거나 삭제하지 않았다.
- [x] 완전 반례 열 건의 route/부모 삭제 범위와 보존 범위를 고정했다.
- [x] conditioned branching theorem과 density/current bridge를 분리했다.
- [x] stationary spinodal과 finite-rate freeze를 분리했다.
- [x] spectator의 exactness--uniqueness tradeoff와 radiation correction을 분리했다.
- [x] $H_L$, $H_*$, $H_0$ 및 horizon/epoch convention을 분리했다.
- [x] corrected FLRW/numerics와 physical $H_0$ inference를 분리했다.
- [x] exact/legacy/runtime/observation 병존 통합 순서를 비파괴적으로 승인했다.
- [x] 공식 provenance와 blind confirmation을 분리하고 holdout 0건을 보존했다.
- [x] 열린 P0가 0이며 `Gate: PASS`와 모순되지 않음을 확인했다.
- [x] 이 파일 외 제품·정본·다른 stage 파일을 수정하지 않았다.

