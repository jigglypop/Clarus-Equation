# 리만·MRA 보존 문서 안내

이 폴더는 리만 제타 함수에서 동기를 얻은 attention·positional encoding의 사양과
내부 실험 원고를 보존한다. 구현 결과나 수치 실험은 리만 가설의 증명 또는 물리 이론의
검증이 아니다. 현재 CE 통합 논문에 사용할 때는 입력·baseline·ablation·OOD 범위를
별도로 확인한다.

- [수학 정리·반례 감사](math_claims_audit.md): 로그 대칭·sheet·Mellin·Gram 양성·자기수반성의 조건과 과도한 사양의 수정

- [Mellin–Riemann Attention 원고](mra_paper.md): 내부 ablation과 length extrapolation 기록
- [MRA block 사양](mra_block_spec.md): tensor·score·제약·backend 계약
- [Riemann surface positional encoding 사양](riemann_pe_spec.md): 좌표 lift와 attention 구현 계약
