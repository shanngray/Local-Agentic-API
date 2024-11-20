import httpx
from typing import Dict
import logging
from openai import AsyncOpenAI
import json
import os
from typing import TypedDict

logger = logging.getLogger(__name__)

class AgentMatch(TypedDict):
    agent_name: str
    confidence: float
    reasoning: str

async def agent_search(task_description: str) -> Dict:
    """Search for the most suitable agent based on task description"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:8000/agent/lookup",
                timeout=300
            )
            response.raise_for_status()
            agent_list = response.json()
    except Exception as e:
        logger.error(f"Error looking up agents: {str(e)}")
        return {"error": str(e)}
        
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return {"error": "OpenAI API key not configured"}

    try:
        subagent = AsyncOpenAI(api_key=api_key)
        openai_response = await subagent.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system", 
                    "content": """You are helping find the most suitable agent for a task. 
                    Return a JSON object with the following structure:
                    {
                        "agent_name": "name of the selected agent",
                        "confidence": 0.95,  # float between 0 and 1
                        "reasoning": "brief explanation of why this agent was chosen"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"Task description: {task_description}\n\nAvailable agents and their details: {json.dumps(agent_list, indent=2)}\n\nWhich agent would be most suitable for this task?"
                }
            ]
        )
            
        recommended_agent: AgentMatch = json.loads(openai_response.choices[0].message.content)
        
        if recommended_agent["agent_name"] in agent_list:
            logger.info(f"Selected agent {recommended_agent['agent_name']} with confidence {recommended_agent['confidence']}")
            return {recommended_agent["agent_name"]: agent_list[recommended_agent["agent_name"]]}
        
        logger.warning("No matching agent found, returning all agents")
        return agent_list

    except Exception as e:
        logger.error(f"Error in agent selection: {str(e)}")
        return {"error": str(e)}

