# 00 계약 — p* 런타임 타깃과 BR-8 이식 금지의 충돌 판정

Status: COMPLETE

## 질문

뇌/AGI 런타임이 학습 타깃 p*로 쓰는 비율 튜플 $(\Omega_\Lambda,\Omega_{DM},\Omega_b)\mapsto(0.6891,\,0.2623,\,0.0487)$ (`LEGACY_ROUNDED_RUNTIME_V1`)은

- (H-a) CE 코어 고정점 체인($\alpha_s \to s_W^2 \to \delta \to D_{\rm eff} \to q_{\rm ext}$)의 유도값을 반올림한 **코어 산출 소비**인가,
- (H-b) 우주론 관측 기준선(Planck류)을 반올림해 이식한 **BR-8 위반**인가.

판정 결과는 멀티레포 분리(MULTIREPO_PLAN.md Phase 0 P0-1)의 상수 소유권을 결정한다: (H-a)면 튜플을 ce-core 소유 산출로 승격하고 무우주론적 명명(ACTIVE/STRUCT/BACKGROUND)을 유지, (H-b)면 런타임 타깃을 교체하고 이식을 폐기한다.

## 정의역·기호

- 코어 체인: `reality_stone/python/reality_stone/clarus/cosmology_registry.py`의 `CE_CORE_EXACT_V1` (`_build_core_exact_v1`, L200-228). $\alpha_s=0.11789$, $s_W^2=4\alpha_s^{4/3}$, $\delta=s_W^2(1-s_W^2)$, $D_{\rm eff}=3+\delta$, $q_{\rm ext}=e^{-D_{\rm eff}(1-q_{\rm ext})}$ 저근.
- 밀도 사상: `docs/검증_원장/상수_우주론_원장.md` §3 「확률에서 밀도로 가는 주장」이 정본으로 규정하는, 체인 산출에서 세 비율로 가는 사상.
- 판정 대상 튜플: 같은 파일 `LEGACY_ROUNDED_RUNTIME_V1` (L300-319), provenance note "operational defaults, not a CE observational prediction".
- 소비 지점: `clarus/runtime.py`(13회)·`agent.py`(3회)·`stdp.py`(5회)의 `ACTIVE_RATIO/STRUCT_RATIO/BACKGROUND_RATIO` 사용.

## 판정 기준 (결과 확인 전 고정)

- 체인 유도값을 §3 사상으로 밀어 얻은 세 비율을 소수 4자리로 반올림한 값이 판정 대상 튜플과 **세 성분 모두 일치**하면 (H-a) 채택.
- 하나라도 불일치하고, 대신 리포 원장(상수_우주론_원장 §5)에 기록된 관측 기준선의 반올림과 일치하면 (H-b) 채택.
- 둘 다 아니면 UNRESOLVED로 두고 분리 계획에서 튜플을 별도 동결 공리(출처 불명 명기)로 처리한다.
- 관측 기준선은 리포 원장 §5에 이미 기록된 값만 쓴다(최신성 재검증은 이 판정의 범위 밖 — 수치 동일성 질문이므로). 외부 웹 대조 불요, sourcer 레인 SKIPPED.

## 허용 오차

반올림 일치 판정: $|x_{\rm chain} - x_{\rm tuple}| \le 5\times10^{-5}$ (4자리 반올림 반폭).

## PREDECESSOR

없음 (신규 판정). 관련 선행 문서: `_workspace/ce/brain-bio-constrained-frame.md` §6 (BR-8 선언), MULTIREPO_PLAN.md §0 F1-F2.
