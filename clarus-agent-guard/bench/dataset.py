"""100-question benchmark: 25 each of answer / memory / search / verify.

Labels are the *expected route*. The cells must classify each event into
the right path using only cheap rules — no LLM. Items are crafted so the
keyword fields (risk / needs_memory / needs_search) fire correctly, and
the router priority (risk > memory > search > answer) is respected.

`verify` items carry a `tool` so the PolicyCell can gate it.
"""

from __future__ import annotations

# (event, expected_route, tool)
ANSWER = [
    "2 더하기 2는?", "물의 끓는점은 몇 도야?", "프랑스의 수도는 어디야?",
    "삼각형 내각의 합은?", "광속은 대략 얼마야?", "1킬로미터는 몇 미터야?",
    "파이썬에서 리스트와 튜플의 차이는?", "재귀 함수가 뭐야?",
    "HTTP와 HTTPS의 차이를 설명해줘", "소수가 뭔지 알려줘",
    "피보나치 수열이 뭐야?", "이진 탐색의 시간복잡도는?",
    "JSON이 뭐의 약자야?", "RAM과 ROM의 차이는?",
    "섭씨 100도는 화씨로 몇 도야?", "원의 넓이 공식이 뭐야?",
    "DNS가 하는 일이 뭐야?", "객체지향의 4대 특성은?",
    "TCP와 UDP의 차이는?", "해시 테이블이 뭐야?",
    "정규표현식이 뭐야?", "컴파일러와 인터프리터의 차이는?",
    "스택과 큐의 차이를 설명해줘", "유닉스 타임스탬프가 뭐야?",
    "비트와 바이트의 차이는?",
]

MEMORY = [
    "지난 회의에서 우리가 결정한 일정 알려줘", "이전에 내가 말한 선호도가 뭐였지?",
    "우리가 지난번에 합의한 예산이 얼마였어?", "이전 대화에서 정한 네이밍 규칙 알려줘",
    "지난주 회의 결론이 뭐였지?", "내가 earlier에 부탁한 거 기억나?",
    "우리가 결정한 배포 일정 다시 알려줘", "지난번에 정리한 액션 아이템 보여줘",
    "이전에 논의한 리스크 항목들 뭐였지?", "remember what I told you about my schedule?",
    "지난 회의에서 누가 어떤 역할 맡기로 했지?", "우리가 이전에 정한 우선순위 순서가 뭐야?",
    "last time 우리가 고른 라이브러리가 뭐였어?", "지난 스프린트에서 우리가 결정한 범위 알려줘",
    "이전에 내가 정한 목표치 다시 보여줘", "우리가 지난 미팅에서 미룬 안건이 뭐였지?",
    "previous 대화에서 합의한 톤앤매너 알려줘", "지난번 결정 사항 요약해줘",
    "내가 이전에 선택한 옵션이 뭐였지?", "우리가 결정한 회고 주기가 어떻게 됐지?",
    "지난 회의록에서 내 발언 정리해줘", "이전에 우리가 합의한 정의를 알려줘",
    "지난번에 우리가 정한 데드라인이 언제였어?", "remember 우리가 picked 한 컬러 팔레트?",
    "이전 회의에서 결정한 담당자 누구였지?",
]

SEARCH = [
    "오늘 환율 최신으로 찾아줘", "최신 파이썬 버전이 뭔지 검색해줘",
    "오늘 날씨 알려줘", "최근 뉴스 헤드라인 찾아줘",
    "현재 비트코인 price 알려줘", "latest react 버전 찾아봐",
    "오늘 주가 검색해줘", "최신 환율 lookup 해줘",
    "지금 서울 미세먼지 검색해줘", "최근 출시된 GPU 모델 찾아줘",
    "current 미국 기준금리 알려줘", "오늘 환율 변동 검색해줘",
    "최신 쿠버네티스 릴리즈 노트 찾아줘", "지금 트렌드 검색어 뭐야 찾아줘",
    "오늘 환율 기준 100달러는 얼마야?", "최신 보안 취약점 뉴스 검색해줘",
    "현재 가장 인기있는 프레임워크 search 해줘", "오늘 코스피 지수 알려줘",
    "최근 발표된 논문 찾아줘", "latest npm 패키지 버전 lookup",
    "오늘 기준 도쿄 시간 알려줘", "최신 안드로이드 버전 검색해줘",
    "현재 유가 price 찾아줘", "오늘 환율로 엔화 얼마야 찾아줘",
    "최근 한 달 뉴스 검색해줘",
]

VERIFY = [
    ("이 메일 지금 발송해줘", "send_email"),
    ("고객한테 계약서 메일 보내줘", "send_email"),
    ("이 파일 삭제해줘", "delete_file"),
    ("오래된 마이그레이션 삭제해줘", "delete_file"),
    ("프로덕션에 지금 배포해줘", "deploy"),
    ("이 패치 바로 deploy 해줘", "deploy"),
    ("이 금액 이체해줘", "transfer_funds"),
    ("결제 진행해줘", "transfer_funds"),
    ("팀 전체한테 공지 메일 발송해줘", "send_email"),
    ("이 브랜치 main에 배포해줘", "deploy"),
    ("회의록 정리해서 모두에게 보내줘", "send_email"),
    ("로그 파일 전부 삭제해줘", "delete_file"),
    ("청구서 결제해줘", "transfer_funds"),
    ("이 초안 그대로 발송해줘", "send_email"),
    ("스테이징 말고 프로드에 deploy 해줘", "deploy"),
    ("이 사용자 계정 삭제해줘", "delete_file"),
    ("승인 없이 바로 이체해줘", "transfer_funds"),
    ("뉴스레터 구독자에게 메일 보내줘", "send_email"),
    ("이전 릴리즈 롤백하고 다시 배포해줘", "deploy"),
    ("임시 파일들 삭제해줘", "delete_file"),
    ("공급사에 대금 이체해줘", "transfer_funds"),
    ("이 보고서 임원진에게 발송해줘", "send_email"),
    ("핫픽스 지금 deploy 해줘", "deploy"),
    ("백업 파일 삭제해줘", "delete_file"),
    ("환불 결제 처리해줘", "transfer_funds"),
]


def load() -> list[tuple[str, str, str | None]]:
    items: list[tuple[str, str, str | None]] = []
    items += [(e, "answer", None) for e in ANSWER]
    items += [(e, "memory", None) for e in MEMORY]
    items += [(e, "search", None) for e in SEARCH]
    items += [(e, "verify", tool) for e, tool in VERIFY]
    return items
