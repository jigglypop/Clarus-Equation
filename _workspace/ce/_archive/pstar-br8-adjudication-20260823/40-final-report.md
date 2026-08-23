# 40 최종 보고 — p* 런타임 타깃 튜플의 계보 판정

Status: COMPLETE

## 초록

뇌/AGI 런타임이 학습 타깃으로 소비하는 비율 튜플 $(0.6891,\,0.2623,\,0.0487)$이 CE 코어 고정점 체인의 산출인지(H-a), 우주론 관측 기준선의 이식인지(H-b)를 판정했다. 방법은 60자리 십진 정밀도의 독립 재계산과, 원장 §3 사상을 H-a에 최대한 유리한 보조 공리 2개(평탄 폐쇄·DM 분할)로 완성한 뒤의 성분별 대조다. 결과: 체인 산출과는 struct/background 성분에서 허용 오차($5\times10^{-5}$)의 약 75배 차이로 불일치하고, 리포 원장 §5에 기록된 관측 기준선 5종과도 전 성분 불일치한다. 판정은 계약이 사전 등록한 세 번째 분기 **UNRESOLVED**이며, 튜플의 형식 지위는 **출처 불명의 동결 [공리]**다. 한계: 관측 대조는 원장 §5 기록값에 한정했고, 튜플의 실제 역사적 모체는 이 run에서 특정하지 못했다.

## 판정과 근거

1. **(H-a) 기각.** exact 체인($\alpha_s=0.11789$)의 유도 튜플은 $(0.6853,\,0.2660,\,0.0486)$ (4자리 반올림)으로, active는 $-5.33\times10^{-5}$로 근소 초과, struct/background는 $\pm3.7\times10^{-3}$대로 대폭 불일치. 경계 논증: §3 전 사상에서 active $= q(1-\Omega_r) \le 0.0486467 < 0.04865$이므로 물리적 $\Omega_r \ge 0$ 어디서도 통과 불가. legacy 5dp 체인도 동일 패턴으로 기각.
2. **(H-b) 기각 (원장 §5 기록값 한정).** `benchmarks/cosmology/observations_v1.json`의 5개 기준선(Planck2018_base, Planck_ACT_SPT_combined, ACT_DR6_DESI, SPT3G, DESI_DR2) 어느 것도 단일 성분조차 $5\times10^{-5}$ 안에서 일치하지 않는다.
3. **UNRESOLVED — 계약 3항 발동.** 튜플은 별도 동결 공리로 처리하고 출처 불명을 명기한다.

감사(20-audit.md)는 Gate: PASS로 판정했다. 주장 6건 전부 지위-근거 정합, 숨은 공리 0 (보조 공리 2개는 명시 등재).

## 멀티레포 이행에 대한 귀결 (MULTIREPO_PLAN.md P0-1 확정)

- 튜플 `LEGACY_ROUNDED_RUNTIME_V1`은 **ce-agi-runtime 소유의 동결 운영 공리**로 이관한다. ce-core 산출로 승격하지 않고(H-a 기각), 우주론 이식으로도 취급하지 않는다(H-b 기각 — BR-8 위반이 아님).
- 이관 시 $\Omega_\Lambda/\Omega_{DM}/\Omega_b$ 명명·주석을 제거하고 무우주론적 명명(ACTIVE/STRUCT/BACKGROUND)을 고정하며, provenance에 본 run 경로와 "출처 불명 동결 공리"를 명기한다.
- 이 이관으로 `constants.py` → `cosmology_registry` import(P2-1의 절단 대상)가 불필요해진다 — 런타임 상수는 런타임 레포가 자급한다.
- 값 자체는 bit-for-bit 보존한다. 값 변경은 별도 계약(식 개정 규율)이 필요한 행위이며 이 run의 범위 밖이다.

## 미완성 과제 (이관)

- P1-1: 원장 §3에 3-튜플 사상이 없다 — 평탄 폐쇄·DM 분할을 [공리]로 등재하거나 "§3 유도 소비" 표현을 금지하는 조항을 ce-ledger-write 작업으로 추가.
- P1-2: 튜플의 역사적 모체 부재 — 이관 시 출처 불명 명기로 처리 (본 보고서 귀결 반영).
- P2 3건: `bootstrap_solver.py` `*_OBS` 명명 오도, 이중 반올림 규약 의존 관찰, 11-math §3 표의 $5\times10^{-12}$ 전사 오류(정본은 artifacts 로그).

## 재현성

- 재계산: `artifacts/adjudicate_pstar.py`, 교차확인: `artifacts/registry_crosscheck.py`, 로그: `artifacts/adjudicate_pstar.log`.
- 실행: `.claude/hooks/python.cmd python -B _workspace/ce/pstar-br8-adjudication-20260823/artifacts/adjudicate_pstar.py`

## 참조

- `reality_stone/python/reality_stone/clarus/cosmology_registry.py` L200-228, L300-319 (2026-08-23 접근)
- `docs/검증_원장/상수_우주론_원장.md` §2, §3, §5 (2026-08-23 접근)
- `benchmarks/cosmology/observations_v1.json` (2026-08-23 접근)
