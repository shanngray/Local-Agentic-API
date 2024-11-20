import os
from src.base_agent.base_agent import BaseAgent, AgentConfig    
from pathlib import Path
from chromadb import PersistentClient

def create_editor_agent(chroma_client: PersistentClient) -> BaseAgent:
    """Create an agent instance."""
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=False,
        model="gpt-4o-mini",
        system_prompt=(
            "You are an expert editor. You can proof read and, then edit blog posts on any topic. You can also "
            "communicate with other agents using the send_message tool when needed. You understand "
            "meaning of teamwork and always look for other agents that can help you with your tasks."
        ),
        agent_name="EditorAgent",
        description="Editor agent that can proof read and edit blog posts",
        enabled_tools=["agent_search", "list_agents","edit_blog", "read_blog"],
        api_port=8014,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent