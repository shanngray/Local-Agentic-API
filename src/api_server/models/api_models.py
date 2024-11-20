from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    question: str
    max_turns: int | None = 5
    debug: bool | None = False

class QueryResponse(BaseModel):
    response: str
    status: str = "success"
    error: str | None = None

class APIMessage(BaseModel):
    sender: str
    receiver: str
    content: str
    timestamp: str
    message_type: str = "text"
    conversation_id: str

    @classmethod
    def create(cls, sender: str, receiver: str, content: str, conversation_id: str, message_type: str = "text") -> "AgentMessage":
        """Factory method to create a message with current timestamp."""
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
            message_type=message_type
        )

class AgentResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class FeedbackMessage(BaseModel):
    """Model for feedback messages between agents."""
    sender: str
    receiver: str
    conversation_id: str
    score: int
    feedback: str
    timestamp: str
