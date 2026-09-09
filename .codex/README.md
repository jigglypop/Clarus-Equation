# 간결한 Codex 하네스

이 프로젝트는 Codex 설정 표면을 세 곳만 둔다. `../AGENTS.md`는 안정적인 작업 규약이고, `config.toml`은 오케스트레이션 기본값을 정하며, `agents/worker.toml`은 선택적으로 쓸 수 있는 단일 작업자 역할을 정의한다.

루트 스레드는 `gpt-6-astra`와 `medium` 추론을 사용한다. 생성된 에이전트는 기본적으로 `gpt-5.6-sol`과 `low` 추론을 사용한다. 루트가 범위, 충돌, 통합, 최종 판단을 소유한다. 입력과 완료 조건이 분명한 독립 작업에만 작업자를 쓰며, 동시 실행 한도는 목표 작업자 수가 아니다.

프로젝트 전용 프롬프트, 스킬, 생명주기 훅, 연구 실행 규약, Python 실행기는 두지 않는다. Codex 기본 공급자가 Responses 프로토콜을 사용하므로 별도 API나 도구 호출 어댑터도 두지 않는다.

저장소 루트에서 다음 명령으로 하네스를 검증한다.

```powershell
python -B -m pytest -p no:cacheprovider tests/test_repository_harness.py -q
```
