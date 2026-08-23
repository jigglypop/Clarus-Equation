# 11 수학 검산 — p* 튜플 (0.0487, 0.2623, 0.6891)의 모체 판정

Status: COMPLETE

## 1. 대상·정의역·전제

- 판정 대상: `LEGACY_ROUNDED_RUNTIME_V1` = (active, struct, background) = (0.0487, 0.2623, 0.6891),
  `reality_stone/python/reality_stone/clarus/cosmology_registry.py` L300-319. raw_sum = 1.0001.
- 가설: (H-a) CE 코어 체인 산출의 §3 사상 + 4자리 반올림, (H-b) 원장 §5 관측 기준선의 반올림.
- 판정 기준(계약 고정): 성분별 |차이| ≤ 5×10⁻⁵, 세 성분 모두 일치해야 채택. 관측 대조는
  `docs/검증_원장/상수_우주론_원장.md` §5가 지목하는 `benchmarks/cosmology/observations_v1.json`
  기록값만 사용(외부 웹 대조 없음, sourcer SKIPPED).
- 검산은 레지스트리 코드를 import하지 않고 수식에서 Decimal 60자리로 독립 재구현했다.
  레지스트리 import는 교차 확인 전용으로만 1회 수행.

## 2. §3 사상의 정본 추출 (원장 인용)

원장 §3은 **단일한 3-튜플 사상을 정본으로 제공하지 않는다.** 명시된 것은:

1. §3.1 `C-B-LEGACY-01` [공리]: $q_{\rm ext}\mapsto\Omega_b$ — **active 성분만** 정의. "정리나 사전 관측량이 아니다."
2. §3.2 `C-B-COMP-01` [산출]: $f_b^{(m)}=Dq_{\rm ext}$ — matter 내부 분율. 3-튜플 사상 아님.
3. §3.3 `C-B-TRANSIENT-ALG-01` [산출]: $\Sigma_*$에서 $\Omega_m=1/D$, $\Omega_b=q_{\rm ext}$ (곱셈 항등식, 두 물리 전제 조건부).
   SUBSYS 변형: $\Omega_m=(1-\Omega_r)/D$, $\Omega_b=q_{\rm ext}(1-\Omega_r)$, $\Omega_r$은 원장이 고정하지 않는 외부 입력.
4. §3.3 `C-B-TRANSIENT-PHYS-01` [미완성]: 물리 브리지 미폐쇄 — "오늘의 밀도 예측으로 표기하지 않는다."

3-튜플에 도달하려면 원장에 없는 **보조 공리 2개**가 추가로 필요하다(아래 §5 숨은 공리 참조):
평탄 폐쇄 $\Omega_\Lambda=1-\Omega_m$, 분할 $\Omega_{DM}=\Omega_m-\Omega_b$. H-a에 최대한 유리하게
이 완성을 채택하고 검산했다: active $=q$, struct $=1/D-q$, background $=1-1/D$.

## 3. 독립 재계산 값

체인 재계산(Decimal 60자리, Newton, residual < 1e-58; 원장 §2.1 표와 |차이| < 5×10⁻³¹):

| 양 | CE_CORE_EXACT_V1 | LEGACY_DELTA_5DP_V1 |
|---|---:|---:|
| $\delta$ | 0.177758423409973817923… | 0.17776 (공리) |
| $D$ | 3.177758423409973817923… | 3.17776 |
| $q_{\rm ext}$ | 0.048646719644028206426… | 0.048646633337214076306… |
| $1/D$ | 0.314687231304012583376… | 0.314687075172448517194… |
| $1/D-q$ | 0.266040511654984378815… | 0.266040441835234440889… |
| $1-1/D$ | 0.685312768695987416624… | 0.685312924827551482805… |

레지스트리 import 교차 확인: binary64 registry 값과 독립 계산의 |차이| ≤ 2.1×10⁻¹⁷ (일치).
두 체인의 $q$ 차이 8.63×10⁻⁸은 원장 §2.2 기재값과 일치.

## 4. 성분별 비교표

4자리 반올림(half-even; half-up도 동일, 예외는 §6의 이중반올림 관찰), 허용 오차 5×10⁻⁵.

| 성분 | 튜플 | exact 체인 §3.3 (diff) | 5dp 체인 §3.3 (diff) | Planck2018_base | Planck_ACT_SPT | ACT_DR6_DESI | SPT3G_CMBSPA | DESI_DR2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active | 0.0487 | 0.0486 (−5.33e-5) FAIL | 0.0486 (−5.34e-5) FAIL | 0.0493 FAIL | 0.0476 FAIL | 0.0486 (−1.39e-4) FAIL | 0.0496 FAIL | — |
| struct | 0.2623 | 0.2660 (+3.74e-3) FAIL | 0.2660 FAIL | 0.2642 FAIL | 0.2552 FAIL | 0.2535 FAIL | 0.2664 FAIL | — |
| background | 0.6891 | 0.6853 (−3.79e-3) FAIL | 0.6853 FAIL | 0.6865 FAIL | 0.6972 FAIL | 0.6979 FAIL | 0.6840 FAIL | 0.6973 (+8.2e-3) FAIL |

관측 열은 §5 JSON의 $(\omega_b, \omega_c, H_0)$에서 $\Omega_i=\omega_i/h^2$, 평탄 폐쇄로 계산
(중성미자·radiation 항은 manifest에 없어 0 처리 — 이 보정 ~2×10⁻³은 결론을 바꾸지 못함).
SUBSYS 변형($\Omega_r=9.2\times10^{-5}$ 민감도 시험 포함)도 전 성분 FAIL — artifacts 로그 참조.

**경계 논증 (H-a의 견고한 기각):** §3의 모든 사상에서 active는 $q(1-\Omega_r)\le q=0.0486467$
($\Omega_r\ge0$). 그런데 $0.0487-5\times10^{-5}=0.04865>0.0486467$이므로 물리적 $\Omega_r$
어떤 값에서도 active 성분은 허용 오차 안에 들어갈 수 없다. struct/background의 3.7×10⁻³대
불일치는 허용 오차의 약 75배로, 반올림 관행이나 radiation 보정으로 흡수 불가.

## 5. 숨은 공리와 자유도

- §3에서 3-튜플로 가려면 평탄 폐쇄와 DM 분할 공리 2개가 추가로 필요하다(원장 미기재). 본 검산은
  이를 H-a에 유리한 최소 완성으로 선언하고 사용했다 — 이 자유도를 다 써도 H-a는 기각된다.
- SUBSYS 변형의 $\Omega_r$은 자유 파라미터다. active를 0.0487로 만들려면 $\Omega_r=-1.10\times10^{-3}$
  (음의 radiation)이 필요해 비물리적.
- §3.3 자체가 [미완성] `C-B-TRANSIENT-PHYS-01` 조건부이므로, 설령 수치가 맞았더라도
  H-a는 "산출 소비"이지 예측 지위 승격이 아니었다.

## 6. 판정: **UNRESOLVED** (계약 3항 발동)

- **(H-a) 기각.** §3의 어떤 사상(3.1 직접, 3.3 ALG, 3.3 SUBSYS)과 어떤 체인(exact, legacy 5dp)의
  조합도 세 성분을 재현하지 못한다. active는 경계 논증으로 전 사상에서 기각, struct/background는
  허용 오차의 ~75배 차이.
- **(H-b) 기각 (§5 기록값 한정).** §5 manifest의 5개 행 중 어느 것도 세 성분은커녕 단일 성분도
  허용 오차 안에서 튜플과 일치하지 않는다 (최근접: ACT_DR6_DESI active −1.39×10⁻⁴).
- 따라서 계약 판정 기준 3항: 튜플은 멀티레포 분리에서 **별도 동결 공리(출처 불명 명기)**로 처리한다.
  이는 레지스트리 자체 표기(formal_status=AXIOM, note "operational defaults, not a CE observational
  prediction") 및 선행 run(`_workspace/ce/_archive/cosmology-full-closure-unification-20260815/11-math.md`
  R-U1-LEGACY [공리] 판정)과 정합적이다.

부수 관찰 (target-aware, 판정 근거 아님): $q_{\rm ext}$를 5자리로 먼저 반올림하면 0.04865
(= `bootstrap_solver.py` L45 `OMEGA_B_OBS`)이고 이를 half-up 4자리로 다시 반올림하면 0.0487이다
(half-even이면 0.0486). 즉 active 성분만은 이중 반올림된 체인 $q$와 정합 가능하나, struct/background는
어떤 체인 산출과도 정합하지 않으므로 튜플 전체의 모체 규명은 되지 않는다. 관측 근접은 증명이 아니다.

## 7. P0/P1/P2

- **P0: 없음.** 판정 계약이 세 결과를 모두 사전 등록했고, 레지스트리·원장의 기존 지위 표기([공리],
  COMPATIBILITY_ONLY)는 이 판정과 모순되지 않는다. 무너지는 부모 명제 없음.
- **P1-1:** 원장 §3은 3-튜플 사상을 정본으로 제공하지 않는다. 닫는 데 필요한 최소 보조정리:
  평탄 폐쇄·DM 분할을 §3에 명시적 [공리]로 등재하거나, 3-튜플 소비를 §3 유도 소비로 부르는
  문서 표현을 금지하는 원장 조항.
- **P1-2:** 튜플의 실제 모체가 리포 정본 어디에도 없다(체인도 §5 관측도 아님). 분리 계획에서
  출처 불명 동결 공리로 명기하는 계약 후속 조치가 필요하다.
- **P2-1:** `bootstrap_solver.py` L45-48의 `OMEGA_B_OBS/OMEGA_LAMBDA_OBS/OMEGA_DM_OBS` 이름은
  관측이 아님에도 OBS 접미를 사용 — 주석은 부인하지만 이름이 오도적.
- **P2-2:** active 성분의 이중 반올림 정합(0.04865→0.0487)은 반올림 규약(half-up vs half-even)에
  의존 — 기록만 남김.

## 8. 재현

```
cd C:/Users/dongh/OneDrive/Desktop/Clarus-Equation
PYTHONDONTWRITEBYTECODE=1 ./.claude/hooks/python.cmd python -B \
  _workspace/ce/pstar-br8-adjudication-20260823/artifacts/adjudicate_pstar.py
PYTHONDONTWRITEBYTECODE=1 ./.claude/hooks/python.cmd python -B \
  _workspace/ce/pstar-br8-adjudication-20260823/artifacts/registry_crosscheck.py
```

- 스크립트: `_workspace/ce/pstar-br8-adjudication-20260823/artifacts/adjudicate_pstar.py`,
  `artifacts/registry_crosscheck.py`
- 전체 로그: `_workspace/ce/pstar-br8-adjudication-20260823/artifacts/adjudicate_pstar.log`
