from fastapi import HTTPException

from backend.core.config import OPENAI_API_KEY


def require_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")
