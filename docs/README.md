# SFE 문서 인덱스

SFE 프로젝트의 문서는 **핵심 이론**과 **최신 유도**를 중심으로 재편되었습니다.
과거의 연구 노트들은 `archive` 폴더에 보관되어 있습니다.

## 문서 사용 규칙 (주장 레벨)

이 프로젝트 문서에는 같은 기호(예: $\epsilon$, $\Omega_\Lambda$)가 서로 다른 목적의 문맥에서 등장한다. 혼동을 피하기 위해, 본 문서 집합은 아래 규칙을 따른다.

- **입력(캘리브레이션)**: 관측값을 사용해 SFE 내부 파라미터를 고정하는 단계
- **출력(예측)**: 고정된 파라미터로 새로운 관측량을 계산하는 단계
- **비교(검증)**: 출력이 관측과 어느 정도 일치하는지 평가하는 단계
- **가정(모형 선택)**: 평탄성, 성분 분해, 무차원화 스케일 등 물리 모형을 닫기 위해 추가로 채택하는 조건

특히 우주론에서는, “요약값(예: $\Omega_\Lambda$) 맞춤”과 “함수형 예측(예: $E(z)$, $D_L(z)$, $f\sigma_8(z)$)”을 구분한다. 요약값 정합은 출발점이 될 수 있으나, 이론의 강도는 함수형 예측에서 결정된다.

## 핵심 이론 (Core Theory)
**경로**: `docs/Core_Theory/` (구 Part 8)

SFE의 핵심 아이디어와 통합 프레임워크(후보)를 정리한 부분입니다.
- [통합 프레임워크](Core_Theory/8.1_SFE_리만기하학적_통합_프레임워크.md)
- [SCQE (자가보정 양자소자)](Core_Theory/8.4_양자컴퓨팅의_한계와_SCQE.md)
- [0.37 법칙과 우주론](Core_Theory/8.5_우주론적_함의와_0.37_법칙.md)
- [정합성 최종 검증](Core_Theory/8.9_정합성_최종_검증.md)

## 유도 및 응용 (Derivations & Applications)
**경로**: `docs/Derivations_Applications/` (구 Part 10)

핵심 난제들에 대해 SFE 공리·마스터 작용을 적용한 **수학적 모형, 유도 스케치, 수치 검증**을 모아 둔 부분입니다.
- [암흑에너지 유도](Derivations_Applications/04_Dark_Energy_Derivation.md)
- [나비에-스토크스 해법](Derivations_Applications/01_Navier_Stokes.md)
- [단백질 접힘 경로](Derivations_Applications/03_Protein_Folding_Derivation.md)
- [리만 제타 영점](Derivations_Applications/02_Riemann_Zeta_Derivation.md)
- [마스터 액션 유도](Derivations_Applications/06_Master_Action_Universal_Derivation.md)

## 아카이브 (Archive)
**경로**: `docs/archive/`

초기 연구, 방법론, 확장 이론 등 **구버전/실험적 노트**가 포함된 참고용 문서들입니다.  
현재 본문과 톤·표현이 다를 수 있으며, 향후 정리·삭제 대상(대기 상태)로 취급합니다.
- Part 1: 이론 기초
- Part 2: 핵심 검증 (초기)
- Part 3: 확장 이론 (입자물리)
- Part 4: 방법론
- Part 5: 고급 주제
- Part 6: 공학 응용
- Part 7: 양자 컴퓨팅 (초기 모델)
- Part 9: 시스템 공학
