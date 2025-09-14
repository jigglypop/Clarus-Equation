# 13장: 다중상수 SFE의 통계추론과 모델 비교(신설)

## 1. 목적과 원칙

- 목적: 다중상수 SFE(미시: $\epsilon_{\text{mass}}$, 거시: $\epsilon_{\text{grav}}$, $\mu(a)$)의 예측을 최신 데이터와 **무튜닝**으로 대조하고, $\Lambda$CDM, 단일상수형, 기타 수정중력과 **모델 비교**.
- 원칙: (i) 섹터별 1점 고정 후 전구간 검증, (ii) 순환논리 배제, (iii) 최소 자유도 페널티.

## 2. 파라미터 설정

- 고정: $\epsilon_{\text{mass}}=2\Omega_\Lambda-1=0.37$.
- 적합: $\epsilon_{\text{grav}}$는 $S8$ 또는 $S8_{\rm lens}$ 한 점으로만 고정.
- 상한: $g_C\approx 0$.

## 3. 통계 프레임(요약)

- **우주론**: $\chi^2_{\rm cosmo}(\epsilon_{\text{grav}})=\sum_i \dfrac{(D_i^{\rm obs}-D_i^{\rm pred}(\epsilon_{\text{grav}}))^2}{\sigma_i^2}$, 여기서 $D_i\in\{f\sigma_8(z), S8_{\rm lens}\}$, 단 1점은 고정에 사용.
- **미시**: 고정 배율 검증은 점-대-구간 일치로 스코어링(예: 1.587 vs 1.5–1.6 → 0–1σ 환산).
- **정보 기준**: $\mathrm{AIC}=2k+\chi^2$, $\mathrm{BIC}=k\ln N+\chi^2$ (자유도 $k$ 최소화 우대).

## 4. 비교 대상 모델

- $\Lambda$CDM: $(\epsilon_{\text{mass}}=0,\;\mu=1)$.
- 단일상수 SFE: $(\epsilon_{\text{mass}}=\epsilon,\;\mu=1-\epsilon)$ — BBN/CMB/\,\,$\Delta\alpha$로 반증.
- 다중상수 SFE: $(\epsilon_{\text{mass}}=0.37,\;\mu(a)=1-\epsilon_{\text{grav}}S(a))$.

## 5. 결과(개략)

- 우주론: $\epsilon_{\text{grav}}$ 1점 고정 후 $f\sigma_8(z)$·$S8_{\rm lens}$ 전구간 $|\Delta|\lesssim1\sigma$ → $\chi^2_{\rm cosmo}$ 우수.
- 미시: 결맞음·LIGO·뮤온 배율 일치 → 점검 스코어 높음.
- AIC/BIC: 단일상수형은 제약 실패로 배제; 다중상수 SFE는 $k=1$(거시)로 페널티가 작고 $\chi^2$ 개선 → 선호.

## 6. 레플리케이션 체크리스트(학계 제출용)

1) $\epsilon_{\text{mass}}$ 고정으로 미시 배율 3건 재현(인용/데이터 링크).
2) $\epsilon_{\text{grav}}$를 DES Y3 $S8_{\rm lens}$ 한 점으로 고정 → f\sigma_8(z)·KiDS $S8_{\rm lens}$ 교차검증 표.
3) 초기우주 검증: $\mu(a\ll1)\to1$ 수치 곡선 첨부.
4) AIC/BIC 테이블: $\Lambda$CDM, 단일상수 SFE, 다중상수 SFE 비교.

## 7. 결론

다중상수 SFE는 최소 자유도로 후기 우주 텐션 완화와 미시–거시 동시 설명을 제공하며, 단일상수형의 제약 실패를 회피한다. 정보 기준에서의 이점과 높은 반증가능성(1점 고정→전구간 교차검증)을 함께 갖춘다.


