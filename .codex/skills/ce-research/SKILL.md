---
name: ce-research
description: Clarus-Equation 연구를 출처·수학·대안 경로의 독립 레인, 형식 지위 감사, 구현·수치 검증, 최종 집필로 수행한다. CE/Clarus-EQ 식·가설·논문·코드의 유도, 반례, 재현, 승격 판단이나 병렬 연구를 요청할 때 사용한다. 단순 계산과 guard 제품 벤치만 필요한 요청에는 사용하지 않는다.
---

# CE Research

먼저 ../../agents/ce-status-auditor.md와 필요한 역할 카드만 읽는다. 모든 카드를 선로딩하지 않는다.

## 6단계

1. 질문·정의역·주장·기호·허용 오차를 00-contract.md에 고정한다.
2. physics-sourcer, math-verifier, route-explorer를 독립 문맥에서 병렬 실행한다.
3. status-auditor가 형식 출처와 반례를 감사한다.
4. 승인된 범위만 impl-engineer가 구현하고 $ce-validate로 검증한다.
5. P0/P1은 지목된 역할만 최대 2회 수정한다.
6. paper-writer가 판정을 강화하지 않고 최종 보고서를 작성한다.

닫힘 판단에는 $ce-closure-gate, 무차원 식에는 $ce-dimensionless, 문서 반영에는 $ce-doc-write를 적용한다.

## Rust 코어

    cargo run --quiet --locked --release --target-dir <codex-home>/target/ce-research-core --manifest-path <codex-home>/skills/ce-research/core/Cargo.toml -- init <run-dir>
    cargo run --quiet --locked --release --target-dir <codex-home>/target/ce-research-core --manifest-path <codex-home>/skills/ce-research/core/Cargo.toml -- check <run-dir> <lanes|gate|build|final>

모든 산출물은 _workspace/ce/<run-id>/에 쓴다. 원본은 역할 권한 밖에서 수정하지 않는다. 마지막 상태 메시지에 다음을 한 줄로 남긴다.

    CE_RUN=_workspace/ce/<run-id>

