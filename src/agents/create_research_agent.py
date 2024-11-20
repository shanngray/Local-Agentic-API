import os
from src.base_agent.base_agent import BaseAgent, AgentConfig    
from pathlib import Path
from chromadb import PersistentClient

def create_research_agent(chroma_client: PersistentClient) -> BaseAgent:
    """Create an agent instance."""
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=False,
        model="gpt-4o-mini",
        system_prompt=(
            "You are a research agent. You can find information on the internet using the web_search tool. "
            "You can also communicate with other agents using the send_message tool when needed."
        ),
        agent_name="ResearchAgent",
        description="Research agent that can find information on the internet",
        enabled_tools=["agent_search", "list_agents", "web_search"],
        api_port=8013,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent