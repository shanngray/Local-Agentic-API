from typing import Dict, Optional, List, Any
from pydantic import BaseModel
import httpx
from ..models.api_models import APIMessage, AgentResponse, FeedbackMessage
import os
import sys
from pathlib import Path
import uuid
from datetime import datetime
import asyncio

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from log_config import setup_logger, log_event

logger = setup_logger(
    name="AgentDirectory",
    log_path=Path("logs"),
    level=os.getenv("SERVER_LOG_LEVEL", "INFO"),
    console_logging=os.getenv("CONSOLE_LOGGING", "True").lower() == "true"
)

class AgentDirectory(BaseModel):
    name: str
    address: str
    port: int
    agent_type: str
    status: str = "active"
    description: str
    tools: List[str]

class AgentDirectoryService:
    def __init__(self):
        self.agents: Dict[str, AgentDirectory] = {}
        log_event(logger, "directory.startup", "Agent Directory Service initialized")
        
    def register_agent(self, agent: AgentDirectory) -> None:
        self.agents[agent.name] = agent
        log_event(
            logger, 
            "directory.agent_registered", 
            f"Agent registered: {agent.name} ({agent.agent_type}) on port {agent.port}"
        )
        
    def get_agent(self, name: str) -> Optional[AgentDirectory]:
        return self.agents.get(name)
        
    def lookup_agents(self, agent_name: str = None) -> Dict:
        """Lookup registered agents.
        
        Args:
            agent_name: Optional specific agent to look up
            
        Returns:
            Dict of agent information
        """
        if agent_name:
            agent = self.get_agent(agent_name)
            return {agent_name: agent.dict()} if agent else {}
        return {name: agent.dict() for name, agent in self.agents.items()}

    async def route_message(self, message: APIMessage) -> None:
        log_event(
            logger,
            "directory.message_route",
            f"Routing message: {message.sender} → {message.receiver} ({message.conversation_id})"
        )
        
        if message.receiver not in self.agents:
            error_msg = f"Agent {message.receiver} not registered"
            log_event(logger, "directory.route_error", error_msg, level="ERROR")
            raise ValueError(error_msg)
            
        receiver = self.agents[message.receiver]
        target_url = f"http://{receiver.address}:{receiver.port}/receive"
        
        # Add retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    log_event(
                        logger, 
                        "server.request", 
                        f"Attempt {attempt + 1}: Sending to {target_url}",
                        level="DEBUG"
                    )
                    response = await client.post(
                        target_url,
                        json=message.dict(),
                        timeout=300.0
                    )
                    log_event(
                        logger,
                        "server.response",
                        f"Response status: {response.status_code}",
                        level="DEBUG"
                    )
                    
                    response.raise_for_status()
                    return AgentResponse(**response.json())
                    
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                log_event(
                    logger,
                    "server.error",
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}",
                    level="ERROR"
                )
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    log_event(
                        logger,
                        "server.request",
                        f"Waiting {wait_time}s before retry...",
                        level="WARNING"
                    )
                    await asyncio.sleep(wait_time)
        
        error_msg = f"All {max_retries} attempts failed"
        log_event(logger, "directory.route_error", error_msg, level="ERROR")
        raise last_exception

    async def route_feedback(self, feedback: FeedbackMessage) -> None:
        """Route feedback message from one agent to another."""
        log_event(
            logger,
            "directory.feedback_route",
            f"Routing feedback: {feedback.sender} → {feedback.receiver}"
        )
        
        if feedback.receiver not in self.agents:
            error_msg = f"Agent {feedback.receiver} not found"
            log_event(logger, "directory.feedback_error", error_msg, level="ERROR")
            raise ValueError(error_msg)
            
        receiver = self.agents[feedback.receiver]
        target_url = f"http://{receiver.address}:{receiver.port}/feedback"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    target_url,
                    json=feedback.dict(),
                    timeout=30.0
                )
                response.raise_for_status()
                return {"success": True}
                
        except Exception as e:
            error_msg = f"Failed to deliver feedback: {str(e)}"
            log_event(logger, "directory.feedback_error", error_msg, level="ERROR")
            raise
