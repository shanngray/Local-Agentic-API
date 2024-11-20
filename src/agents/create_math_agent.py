import os
from pathlib import Path
import sys
from chromadb import PersistentClient

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from src.base_agent.base_agent import BaseAgent, AgentConfig

def create_math_agent(chroma_client: PersistentClient) -> BaseAgent:
    """Create an agent instance."""
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=False,
        model="gpt-4o-mini",
        system_prompt="""You are a mathematical expert that can perform calculations and solve math problems.
            When you need help with non-mathematical queries, you can use the agent_search and list_agents tools to find other agents 
            that specialize in those areas. After finding a suitable agent, you can use the send_message tool to 
            communicate with them. Always explain your reasoning and process.""",
        agent_name="MathAgent",
        description="Mathematical expert that can perform calculations and solve math problems",
        enabled_tools=["agent_search", "list_agents", "calc"],
        api_port=8010,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent