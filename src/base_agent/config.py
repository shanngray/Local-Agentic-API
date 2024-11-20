from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from chromadb import PersistentClient

class AgentConfig(BaseModel):
    """Configuration settings for the BaseAgent.
    
    Attributes:
        model: The OpenAI model to use
        system_prompt: Default system prompt
        max_history: Maximum number of messages to keep in history
        debug: Enable debug logging
        tools_path: Path to tools directory
        log_path: Path to log directory
        log_level: Log level for the agent
        agent_name: Name of the agent
        enabled_tools: List of tool names to load
        api_port: Port for the API server
        api_host: Host for the API server
        console_logging: Enable console logging
        log_file: Enable log file
        memory_store_path: Path to memory store directory
        temperature: Controls randomness in model outputs (0.0-1.0)
    """
    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    system_prompt: str = "You are a helpful AI assistant."
    max_history: int = 100
    debug: bool = False
    console_logging: bool = False
    log_file: bool = True
    tools_path: Path = Path(__file__).parent / "tools"
    log_path: Path = Path(__file__).parent.parent.parent / "logs"
    log_level: str = "INFO"
    agent_name: str = "AI Assistant"
    description: str = agent_name
    enabled_tools: List[str] = []  # List of tool names to load
    api_port: int  # Added api_port field
    api_host: str = "localhost"  # Add this parameter
    memory_store_path: Path = Path(__file__).parent.parent.parent / "agent_memory"

    class Config:
        env_prefix = "AGENT_"  # Allow env vars like AGENT_MODEL to override
