import httpx
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

async def list_agents() -> Dict:
    """Retrieve information about registered agents from the directory service. 
    Returns a Dict containing agent information including name, description, 
    port, and tools for all agents in directory service
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:8000/agent/lookup"
            )
            return response.json()
    except Exception as e:
        logger.error(f"Error looking up agents: {str(e)}")
        return {"error": str(e)}
