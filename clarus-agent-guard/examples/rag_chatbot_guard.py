"""RAG chatbot guard — drive the runtime headless (no server needed).

Shows the core claim: small cells route each turn *before* any LLM call,
and the choice (answer / memory / search / draft) is recorded as a DAGlet
with causal reasons on every edge.

    python -m examples.rag_chatbot_guard
"""

from server.scheduler import run_event

TURNS = [
    "2 더하기 2는?",                       # -> answer (no tools)
    "지난 회의에서 우리가 결정한 거 알려줘",   # -> memory
    "오늘 환율 최신으로 찾아줘",             # -> search
    "지난 회의 기준으로 그 메일 발송해줘",     # -> draft (blocked external action)
]


def main() -> None:
    for turn in TURNS:
        d = run_event(turn)
        path = d.motif()
        print(f"\nUSER: {turn}")
        print(f"  route : {d.result}   score={d.score:.2f}")
        print(f"  daglet: {path}")
        for e in d.edges:
            print(f"        - {e.src} -> {e.dst}  [{e.kind.value}] {e.reason}")


if __name__ == "__main__":
    main()
