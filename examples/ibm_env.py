import os
from pathlib import Path


def load_ibm_api_key() -> str:
    key = os.environ.get("IBM_API_KEY")
    if key:
        return key
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "=" not in text:
                continue
            name, value = text.split("=", 1)
            if name.strip() == "IBM_API_KEY":
                value = value.strip().strip("'\"")
                if value:
                    os.environ["IBM_API_KEY"] = value
                    return value
    raise RuntimeError("IBM_API_KEY is not set. Set it in environment or .env")


