import os
from src.base_agent.base_agent import BaseAgent, AgentConfig    
from pathlib import Path
from chromadb import PersistentClient

def create_name_agent(chroma_client: PersistentClient) -> BaseAgent:
    """Create an agent instance with shared ChromaDB client."""
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=False,
        model="gpt-4o-mini",
        system_prompt="""enter system prompt here""",
        agent_name="NameAgent",
        description="description of agent",
        enabled_tools=["agent_search", "list_agents"],
        api_port=0000,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent