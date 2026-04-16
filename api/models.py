from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    question: str
    source: Optional[str] = None
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list
