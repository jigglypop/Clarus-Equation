"""Entry point.  `uvicorn server.main:app --reload`

The app is a thin shell over the runtime in scheduler.py. If FastAPI is
not installed you can still drive everything from Python:

    from server.scheduler import run_event
    print(run_event("지난 회의 기준으로 메일 보내줘").to_dict())
"""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Clarus Agent Guard", version="0.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
