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
    sender: str = Field(..., description="Name of the sending agent")
    receiver: str = Field(..., description="Name of the receiving agent")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO format timestamp")
    message_type: str = Field(default="text", description="Type of message being sent")
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    
    @classmethod
    def create(cls, sender: str, receiver: str, content: str, conversation_id: str, message_type: str = "text") -> "APIMessage":
        """Factory method to create a message with current timestamp."""
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
            message_type=message_type
        )

    class Config:
        schema_extra = {
            "example": {
                "sender": "agent_1",
                "receiver": "agent_2",
                "content": "Hello, how are you?",
                "timestamp": "2024-03-21T10:00:00Z",
                "message_type": "text",
                "conversation_id": "conv_123456"
            }
        }

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
