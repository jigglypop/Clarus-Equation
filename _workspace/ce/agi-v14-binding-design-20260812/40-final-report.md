# 40-final-report — agi-v14-binding-design-20260812

Status: COMPLETE

## 질문과 답

v13 계열의 두 실패(T=8 시간 신용할당 붕괴, heldout 조합 일반화 천장)를 원리적으로 해결하는 전이 방정식 족을 유도했다. 답은 세 정리와 한 공리로 닫혔다.

## 확정된 수학 (11-math, 감사 판정 일치)

- **C1 [정리]** 무손실 슬롯(닫힘 구간 정확 항등) + 분리 저장 + bilinear readout이면 임의 T·heldout에서 오차 $\le \Phi(-(1-t)/\sigma_N)+3\sigma^4/t^2$, $\sigma_N=\tfrac{\sqrt{51}}{2}\sigma$ (σ=0.08에서 ~2×10⁻⁴). T-불변·heldout-불변. MC 검산 1.815×10⁻⁴로 일치.
- **C2-선형 [정리]** 상태가 latent 연결이고 판독이 선형이면 32셀 실현 불가 (증명: $\sum_b \operatorname{sign}(w_k^\top b)b = 4w_k$와 $\sum_k w_k = 0$의 모순). 수치 천장 21/32.
- **C2-가법 [정리]** 곱 항 없는 임의 가법 판독도 불가 (초과수준집합 전순서 중첩 + 등기수 논증). 천장 26/32. GRU의 heldout 0.55는 이 천장 정합.
- **A-SAL [공리, 구조 장애 논증 부속]** 선형 게이트 $\sigma(u^\top x+\beta)$는 부호 가변 신호를 latch할 수 없음 — salience는 짝함수(에너지) $\sigma(a\|m\odot x\|^2+b)$여야 함. v10/v13/V9 공통 실패 서명의 원인.

## 경로 탐색 (12-routes, toy 6+8 seed — 본 채점 아님)

- **route L [GO-후보, 경험식]**: 에너지 게이트 + HRR 순환 컨볼루션 binding, n=8, ~351 params. toy: id/noise/horizon/combined 0.994/0.974/0.984/0.919, heldout 0.972 (gru20 0.889/0.551 대비). $\langle w, u\circledast v\rangle = u^\top C(w)v$ — circulant 제약이 암기 자유도를 제거.
- 대조 G/I가 두 병목(게이트 축, 유지 축)의 독립성을 분리 확인. H(latch+선형 readout) heldout 0.424는 C2 교차 예측 정합.
- route M(key-value) 2순위 GO-후보.

## 구현·본 채점: 미실행

감사(Gate: PASS)가 승인한 16-seed 본 채점 + killing test 3종(T∈{16,32} 무열화, circulant ablation, H-형 음성 대조)은 작업 중단(사용자)으로 미실행. **route L의 지위는 "GO-후보(toy), 미채점"에 고정** — toy 수치를 본 채점처럼 인용 금지(감사 P2-3). 본 채점은 후속 run(클라루스장 구현 단계 또는 V9 재개 계약)으로 이월하며, 감사 §4의 승격 조건(G1+G3+G4 + killing test (a) + ablation (b))이 그대로 유효하다.

## 파급

- 클라루스장 run(`agi-clarus-field-20260812`)이 본 run을 PREDECESSOR로 승계 — CF-2(게이트-스케줄 안정성 정리)·CF-5(A-SAL 장 일반화) 증명 완료, R-A1/R-A3로 p* 자기수렴 메커니즘 후보 확보.
- V9 재개 청사진: 에너지 게이트 + latch + 곱 binding + 레벨별 시상수 + CF-2 안정성 — 본 run의 정리들이 수학적 기초.

## 미해결·경계

- route L 본 채점 미실행 (이월). circulant 스팬 주장 무증명(killing test 지정됨). A-SAL 고정 스케일의 과제 정보 의존(P2-4)은 학습형 (a,b)로 구현 시 확인 필요.
- P2-1~P2-4 (표현·인용 경계) — 20-audit §2 목록 유지.
