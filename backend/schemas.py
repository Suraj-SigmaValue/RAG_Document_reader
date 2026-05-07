from typing import List, Optional, TypedDict

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    chunks: List[dict]
    token_usage: dict
    retrieval_timing: Optional[dict] = None


class GraphState(TypedDict):
    question: str
    context: List[dict]
    answer: str
    retrieval_timing: Optional[dict]
    token_usage: Optional[dict]
