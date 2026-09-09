# CE 관측 예측 연결 묶음

먼저 goal_contract_ko.md, observable_report_ko.md를 읽는다.
기존 G01–G08 및 R01–R12는 모두 유지하며 이번에 완료 처리하지 않았다.

실행:
```bash
OPENBLAS_NUM_THREADS=1 python -W error observable_bridge.py
OPENBLAS_NUM_THREADS=1 python -W error postcomparison_diagnostics.py
```

관측 자료는 data/와 data_manifest.json, 실제 결과는 results/에 있다.
requirements-reference.txt는 참조 실행 버전이다. 새로운 모형 개발에 이미 사용한 자료를 독립 holdout으로 부르지 않는다. 이 묶음은 초기상태와 절대척도가 입력인 기존 후기 벤치마크의 관측 진단이며 CE의 최종 관측 예측 완성이 아니다.

source_context/의 첨부 자료는 수정하지 않은 출처 사본이다. 원격 Git을 변경하지 않았다.
