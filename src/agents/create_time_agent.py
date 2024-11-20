import os
from src.base_agent.base_agent import BaseAgent, AgentConfig    
from pathlib import Path
from chromadb import PersistentClient

def create_time_agent(chroma_client: PersistentClient) -> BaseAgent:
    """Create an agent instance."""
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=False,
        model="gpt-4o-mini",
            system_prompt="""You are a time and timezone expert. Help users with time-related queries using 
            the get_current_time tool. You can also communicate with other agents using the send_message tool 
            when needed. Explain time differences and conversions clearly.""",
        agent_name="TimeAgent",
        description="Time and timezone expert that can provide current time in different timezones",
        enabled_tools=["agent_search", "list_agents", "get_current_time"],
        api_port=8011,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent