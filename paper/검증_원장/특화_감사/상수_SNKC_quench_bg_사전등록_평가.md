# SNKC quench 배경 사전등록 holdout 평가 보고

Status: COMPLETE (2026-08-30)

## 초록

SNKC 우주론에서 부모 명제 `SNKC-R2-JOINT-COSMOLOGY-03`이 완전 반례로 삭제된 뒤, 남겨진 등록 요건에 따라 새 [예측] 후보를 세우는 첫 시도를 수행했다. 등재된 새 모형 계약 4종을 식별 가능성·no-go 통로·falsifier 정의 가능성·도달 가능성 기준으로 정렬해 `SNKC-R2-THEATER-OPENING-CONTRACT-06`의 배경 한정 축소 진입을 선택하고, 동결 kinetic 기본점에 cold quench 생성 성분을 더한 branch를 해시 동결 매니페스트로 사전등록했다. 이 저장소가 한 번도 채점하지 않은 두 자료 — eBOSS DR16 consensus BAO-only(1차)와 Moresco cosmic chronometer 15점(2차) — 에서 봉인 평가를 정확히 1회 실행한 결과, 사망 문턱 $\Delta\chi^2>+9$는 발동하지 않았다($\Delta\chi^2_P=+0.071$, $\Delta\chi^2_X=+1.300$, $\Delta\chi^2_{\rm CC}=-0.021$). 다만 동결점에서 $\Omega_{\rm prod,0}$이 배경 비식별로 판명되어 실제 시험된 내용은 동결 kinetic 배경의 $E(z)$ 형상뿐이며, holdout의 독립성은 완전 봉인이 아닌 약한 독립성(weak blindness) 등급이다. 이 결과는 배경 형상 정합의 미기각 보고이지 이론의 확증이 아니다.

## 1. 서론

원장 `SNKC-R2-JOINT-COSMOLOGY-03`은 "현재 SNKC background가 암흑물질·암흑에너지·허블 텐션과 CMB/LSS/growth를 동시에 식별·예측한다"는 부모 명제를 완전 반례(배경이 유일한 섭동을 함의하지 않음)로 삭제하면서, 미래의 경험 주장은 완전 지정 off-trajectory EFT, initial state와 finite-renormalization scheme, complete likelihood, data split, preregistered holdout을 갖춘 별도 [예측]으로만 등록하도록 요건을 남겼다. 이 보고는 그 요건을 배경 한정으로 좁혀 이행한 첫 run의 계약, 계산, 봉인 평가와 감사 결과를 하나의 서사로 기록한다. 모든 판정의 정본은 원장 행 `SNKC-R2-THEATER-QUENCH-BG-PREREG-10`(계약), `SNKC-R2-THEATER-QUENCH-BG-EVAL-10`(결과), `SNKC-R2-DESI-CONTROL-NOTE-10`(대조군 표기 정정)이며, 본 문서는 그 행들을 독자가 따라갈 수 있는 순서로 재서술한다.

## 2. 후보 정렬과 진입점 선택

원장에 등재된 새 모형 계약 4종을 (i) 현존 구조만으로 사전고정 forward 계산이 가능한가, (ii) 삭제된 부모·no-go와 충돌하지 않는 통로가 있는가, (iii) 독립 falsifier와 같은-경계 대조군을 정의할 수 있는가, (iv) 봉인 holdout 1회 평가에 도달 가능한가의 순서로 정렬했다.

`SNKC-R2-THEATER-OPENING-CONTRACT-06`이 1위였다. 매끈한 개장의 정확한 Bogoliubov 점유수, quench 장부 replacement, 총 Ward 폐쇄, cold 늦은시간식이 조건부 정리로 이미 존재하고, no-go들(순간 개장 UV 반례, 생성입자 단독 DE no-go)의 우회 통로가 전부 명시적이었기 때문이다. `SNKC-R2-PHYSICAL-STABILITY-CONTRACT-02`는 자유도 0의 유도 과제라 이론적으로 가장 강하지만 holdout이라는 종점이 없어 2위, `SNKC-R2-K4-MULTICOMPONENT-CONTRACT-02`는 연산자 계수가 전부 미유도 자유 입력이라 3위, `QD-RESIDUAL-OBSERVABLE-CONTRACT-01`은 rank 1/nullity 4 비식별성을 제거할 R4 독립 폐쇄 구조가 정본에 부재해 구조적으로 죽은 경로(4위)로 판정했다. 이 정렬과 세 우회 경로(R-A 배경 한정 quench 증강, R-B 다성분 안정성 유도, R-C 성장 경험식의 공리 명시화)는 run 기록에 보존되어 있으며, 선택된 것은 R-A다.

## 3. 정의와 표기

동결 kinetic 배경은 단일 작용

$$
P(T,X)=\rho_\infty\left[\frac{\kappa}{2}\left(\frac{X}{X_*}-1\right)^2-\left(1-e^{-\Gamma T}\right)\right],\qquad X_*=\tfrac12
$$

의 $\Gamma=10$, $\kappa=10^{17}$ 기본점과 현재 경계 tuple $(\Omega_b,\Omega_r,\Omega_K,\Omega_V)=(0.049,\ 9\times10^{-5},\ 0.26391,\ 0.687)$로 정의된다. 이 동결은 본 run 이전에 원장에서 완료된 것이며, 본 run은 어떤 파라미터도 재조정하지 않았다. $E(z)=H(z)/H_0$는 무차원 배경, $s=c/(H_0 r_d)$는 BAO 비교의 profiled 공통 척도, $\Delta\chi^2$는 등록 branch에서 같은-동결-경계 평탄 $\Lambda$CDM 대조군을 뺀 값이다.

## 4. 공리

1. **[공리: quench tuple 채택]** 개장 작용의 무차원 tuple을 $(m_{\rm in}/E_*,\ m_{\rm out}/E_*,\ \tau E_*)=(0.15849,\ 17.783,\ 1.2589)$로 채택한다. 선택 규칙(cold 창 통과 집합의 log10-무게중심 최근접 격자점)은 관측 무접촉 스캔에서 결과 확인 전에 스크립트로 고정되었다.
2. **[공리: finite renormalization]** `SNKC-R2-ADIABATIC-SCHEME-AX-09`의 4차 adiabatic subtraction과 선언 counterterm을 그대로 채택한다.
3. **[공리: off-trajectory EFT의 배경 한정 채택]** 위 $P(T,X)$와 quench 작용을 배경 수준에서 유효한 것으로 채택한다. 섭동 주장은 하지 않으므로 `SNKC-R2-THEATER-PERTURBATION-OPEN-07`과 저촉하지 않는다.

## 5. 정리와 산출 (인용)

배경식과 코드의 등가(u-ODE가 위 작용에서 정확히 나오고 총 연속방정식이 항등식 $(2+3\delta)/(1+1.5\delta)=2$로 닫힘)는 math 레인이 독립 상태변수·독립 적분기로 검산했다 [정리: 조건부]. cold 창의 비공집합은 9261 격자점 중 2616점이 사전 선언 문턱($w_{\rm prod}\le0.01$, $\rho_{\rm prod}>0$)을 통과함으로 확인되었고, 통과 집합 전체에서 $m_{\rm out}\tau\ge3.98$이라는 단일 곱 경계가 성립한다 [산출]. 정준점의 $w_{\rm prod}(a_*)=2.1767\times10^{-4}$는 개장 후 $a^{-2}$로 감소하므로 배경에서 pressureless로 취급했다 [산출].

교정 자유도 $\Omega_{\rm prod,0}$은 이미 소비된 DESI DR2 13점 단독의 argmin으로 0에 동결되었다 [산출]. 이때 41점 $\chi^2$ 곡선의 spread가 $1.71\times10^{-8}$에 불과함이 드러났다: 동결점에서 kinetic 유체의 $w_k\sim10^{-18}$이라 $\Omega_K'\leftrightarrow\Omega_{\rm prod,0}$ 예산 교환이 배경 $E(z)$를 관측창 전체에서 상대 $3\times10^{-9}$ 이하로만 바꾼다. **따라서 quench 층은 이 평가에서 아무것도 시험되지 않은 [공리] 선언층이고, 등록 예측의 실질 내용은 동결 kinetic 배경의 $E(z)$ 형상이다.** 이 비식별 판정은 `SNKC-R2-ZEROD-TO-INITIAL-NOGO-04`를 독립적으로 보강한다.

## 6. 사전등록과 blindness 등급

정본 매니페스트는 `experiments/preregistration/cosmology_snkc_quench_bg_v1.json`(sha256 `b93b9f05384584a4203c3bd403bccc7de4c4c521c54905ee0dfdcd683f748207`, 동결 2026-08-30)이다. 기존 `ce-cosmology-future-holdout-v2`의 미배정 confirmatory holdout은 변경·소비하지 않았다 — sourcer가 2026-08-02~08-30 구간의 적격 신규 BAO release 부재를 다중 채널로 확인했으므로 그 매니페스트는 규정대로 대기한다.

두 holdout은 2020–21년 공개 자료로, 보증되는 것은 이 저장소가 그 수치를 채점한 적 없다는 사실뿐이다(약한 독립성). 또한 매니페스트 동결과 holdout 수치 취득 후, 모형-자료 계산 전에 평가 프로토콜의 명료화 1건 — 가우시안 1차 판정(P)과 비가우시안 확장(X)의 분해, "P 또는 X 어느 쪽이든 문턱 초과 시 기각"의 강화 — 이 선언되었다. 감사는 이를 방향 보수적(기각 범위 확대만 가능)·판정 불변(모든 허용 해석에서 동일 결론)인 문서화된 deviation으로 판정했고, 그 대가로 이 run의 등급은 "clean prereg"가 아니라 **weak blindness + post-access 명료화 명기 [예측] 후보**다.

## 7. 봉인 평가 결과 (중립 서술)

평가는 정확히 1회 실행되었고 이후 어떤 재조정·재실행도 없었다.

| 비교 | 등록 branch | 동결-tuple $\Lambda$CDM | $\Delta\chi^2$ | dof |
|---|---:|---:|---:|---|
| 1차 P: eBOSS DR16 BAO-only 가우시안 8관측 | 5.723637 | 5.652856 | $+0.070781$ | 7 |
| 1차 X: P + ELG·Ly$\alpha$ 공식 비가우시안 테이블 | 12.541139 | 11.241161 | $+1.299978$ | — |
| 2차: Moresco CC 15점 (mod 공분산) | 6.085568 | 6.106113 | $-0.020545$ | 14 |
| 2차 감도: mod_ooo 공분산 | 6.079306 | 6.099014 | $-0.019708$ | 14 |

사망 문턱 $\Delta\chi^2>+9$(P 또는 X)는 발동하지 않았다. 1차에서 등록 branch는 대조군보다 일관되게 근소 열세이고, 2차에서는 근소 우세다. 두 배경의 차이 자체가 $z=1.5$에서 $+0.34\%$ 수준이라 현재 자료의 분해능 안에서 두 모형은 사실상 구분되지 않는다. 이 결과는 미기각·배경 형상 정합의 보고이며, 우월성·확증·닫힘의 근거가 아니다. profiled 값들은 $s\simeq30.0$(P), CC의 $H_0\simeq66.4\pm7.6\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$였다.

부수 정정: 구현 검산 중 `SNKC-R2-DESI-BAO-03`의 "같은 경계 평탄 $\Lambda$CDM 12.60835" 표기가 실제로는 CE_RATIOS tuple 경계 값임이 확인되어(엄밀한 동결-tuple 대조군은 13.44235), 원장 본문이 감사 확정으로 정정되었다.

## 8. 미완성 과제와 한계

- $\Omega_{\rm prod,0}$의 배경 비식별로 quench 기전 자체는 미시험 — quench 층을 실제로 시험하려면 배경이 아닌 관측축(생성 시점의 유효 자유도 변화, 조기 성분의 $r_d$ 영향을 profiled 척도 없이 보는 절대 보정 등)이 필요하다 [미완성].
- 섭동·성장·CMB/LSS는 `SNKC-R2-THEATER-PERTURBATION-OPEN-07`에 의해 이 run의 주장 범위 밖이다. Einstein–Boltzmann 수준 likelihood는 여전히 최대 공백이다 [미완성].
- CC 공분산의 young 성분 수치는 공식 공개 코드에 부재하여 공식 노트북 레시피만 따랐다 [미완성]. Ly$\alpha$ 결합 상관계수는 문헌 추출에 실패해 공식 2D 그리드 직접 사용으로 대체했다.
- confirmatory 등급의 holdout은 v2 매니페스트의 적격 신규 release(2026-08-02 이후 공개, 전체 공분산) 공개를 기다린다. 그 release가 오면 본 run의 동결 모형을 재조정 없이 그대로 채점하는 것이 다음 단계다.
- weak blindness와 post-access 명료화는 이 run의 항구적 등급 한정이다. 소급 승격은 불가하다.

## 9. 재현성

- 매니페스트: `experiments/preregistration/cosmology_snkc_quench_bg_v1.json` (sha256 위 §6).
- forward model·교정: `examples/physics/darksector/kinetic_dark_sector_quench_gate.py`; focused test `tests/test_kinetic_dark_sector_quench_gate.py` (6 passed).
- 봉인 평가: `examples/physics/darksector/kinetic_dark_sector_quench_holdout_eval.py`; focused test `tests/test_kinetic_dark_sector_quench_holdout_eval.py` (8 passed, 합성 자료 전용).
- holdout 자료: `benchmarks/cosmology/snkc_quench_bg_holdout_v1/` — 11개 원본 파일, 파일별 sha256·출처 URL·commit·접근일은 그 README에 고정.
- 실행: 레포 루트에서 `.claude\hooks\python.cmd pytest tests/test_kinetic_dark_sector_quench_gate.py tests/test_kinetic_dark_sector_quench_holdout_eval.py`. 환경 receipt: doctor PASS, Python 3.11.9, numpy 2.4.6 (2026-08-30).

## 10. 참조

- SDSS eBOSS DR16 BAO 데이터: github.com/CobayaSampler/bao_data, commit bb0c1c9 (접근 2026-08-30); Alam et al. 2021, Phys. Rev. D 103, 083533 (arXiv:2007.08991); Bautista et al. 2020 (arXiv:2007.08993); Neveux et al. 2020 (arXiv:2007.08998); Raichoor et al. 2020 (arXiv:2007.09007); du Mas des Bourboux et al. 2020 (arXiv:2007.08995).
- Cosmic chronometer: gitlab.com/mmoresco/CCcovariance, commit 88141333 (접근 2026-08-30); Moresco et al. 2012 (arXiv:1201.3609); Moresco et al. 2015 (arXiv:1503.01116); Moresco et al. 2016 (arXiv:1601.01701); Moresco et al. 2020 (arXiv:2003.07362).
- 원장 정본: `paper/검증_원장/상수_우주론_원장.md` — `SNKC-R2-THEATER-QUENCH-BG-PREREG-10`, `SNKC-R2-THEATER-QUENCH-BG-EVAL-10`, `SNKC-R2-DESI-CONTROL-NOTE-10`.
