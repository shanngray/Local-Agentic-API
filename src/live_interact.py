import asyncio
import httpx
import re
from datetime import datetime
from api_server.models.api_models import APIMessage
from base_agent.models import AgentStatus
from icecream import ic
import os

async def handle_lookup():
    """Query the directory service for all registered agents."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/agent/lookup")
            agents_dict = response.json()
            print("\nRegistered Agents:")
            for name, agent_info in agents_dict.items():
                print(f"\n- {name} (Port: {agent_info['port']})")
                print(f"  Description: {agent_info['description']}")
                print(f"  Tools: {', '.join(agent_info['tools'])}")
        except Exception as e:
            print(f"\nError looking up agents: {str(e)}")

async def handle_direct_message(agent_name: str, message: str, conversation_id: str):
    """Send a message directly to a specific agent."""
    api_message = APIMessage(
        sender="User",
        receiver=agent_name,
        content=message,
        conversation_id=conversation_id,
        timestamp=datetime.now().isoformat()
    )
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://localhost:8000/agent/message",
                json=api_message.dict(),
                timeout=300.0  # Match server timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Handle AgentResponse structure
            if isinstance(response_data, dict):
                if response_data.get('success'):
                    message_content = response_data.get('message', '')
                    if isinstance(message_content, dict):
                        message_content = message_content.get('message', '')
                    print(f"\n{agent_name}: {message_content}")
                else:
                    print(f"\nError: {response_data.get('message', 'Unknown error')}")
            else:
                print(f"\n{agent_name}: {response_data}")
                    
        except httpx.TimeoutError:
            print(f"\nError: Request to {agent_name} timed out after 300 seconds")
        except Exception as e:
            print(f"\nError: {str(e)}")

async def handle_status_update(agent_name: str, new_status: str, conversation_id: str):
    """Update an agent's status."""
    try:
        status = AgentStatus(new_status.upper())
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://localhost:8000/agent/{agent_name}/status",
                json={"status": status.value}
            )
            result = response.json()
            print(f"\nStatus updated: {result['previous_status']} → {result['current_status']}")
    except ValueError:
        print(f"\nError: Invalid status '{new_status}'. Must be one of: {', '.join(s.value for s in AgentStatus)}")
    except Exception as e:
        print(f"\nError updating status: {str(e)}")

async def handle_show_collection(collection_name: str):
    """Show documents in the specified collection."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"http://localhost:8000/collections/{collection_name}")
            documents = response.json()
            print(f"\nDocuments in {collection_name}:")
            for doc in documents:
                # Handle both dictionary and string document formats
                if isinstance(doc, dict):
                    print(f"\n- ID: {doc.get('_id', 'N/A')}")
                    for key, value in doc.items():
                        if key != '_id':
                            print(f"  {key}: {value}")
                else:
                    print(f"\n- {doc}")
        except Exception as e:
            print(f"\nError showing collection: {str(e)}")

async def handle_list_collections():
    """List all available collections."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/collections")
            collections = response.json()
            print("\nAvailable Collections:")
            for collection in collections:
                print(f"- {collection}")
        except Exception as e:
            print(f"\nError listing collections: {str(e)}")

async def handle_list_status():
    """List status of all agents."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/agent/status/all")
            statuses = response.json()
            print("\nAgent Statuses:")
            # Handle both object and string status formats
            for agent, status in statuses.items():
                if isinstance(status, dict):
                    status_value = status.get('status', 'UNKNOWN')
                else:
                    status_value = status
                print(f"- {agent}: {status_value}")
        except Exception as e:
            print(f"\nError listing statuses: {str(e)}")

def handle_clear():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

async def live_interact():
    EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "/exit", "/quit"}
    
    print("Starting conversation with Agentic Community (type 'exit' to quit)\n")
    print("Available commands:")
    print("  /lookup - List all available agents")
    print("  @agent_name message - Send direct message to agent")
    print("  @agent_name/status/new_status - Update agent status\n")
    print("  /show <collection_name> - Show documents in collection\n")
    print("  /list_col - List all collections\n")
    print("  /list_status - List all agent statuses\n")
    print("  /clear - Clear the screen\n")
    print("  /help - Show this help message\n")

    conversation_id = input("Enter a conversation ID (or 'New'): ")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        # Parse input for different command patterns
        status_pattern = re.match(r"@(\w+)/status/(\w+)", user_input)
        direct_msg_pattern = re.match(r"@(\w+)\s+(.+)", user_input)
        
        match user_input:
            case cmd if cmd.lower() in EXIT_COMMANDS:
                print("Ending conversation. Goodbye!")
                break
                
            case "/lookup":
                await handle_lookup()
                
            case cmd if status_pattern:
                agent_name, new_status = status_pattern.groups()
                await handle_status_update(agent_name, new_status, conversation_id)
                
            case cmd if direct_msg_pattern:
                agent_name, message = direct_msg_pattern.groups()
                await handle_direct_message(agent_name, message, conversation_id)
                
            case cmd if cmd.startswith("/show "):
                collection_name = cmd.split()[1]
                await handle_show_collection(collection_name)
                
            case "/list_col":
                await handle_list_collections()
                
            case "/list_status":
                await handle_list_status()
                
            case "/clear":
                handle_clear()
                
            case "/help":
                print("\nAvailable commands:")
                print("  /lookup - List all available agents")
                print("  @agent_name message - Send direct message to agent")
                print("  @agent_name/status/new_status - Update agent status")
                print("  /show <collection_name> - Show documents in collection")
                print("  /list_col - List all collections")
                print("  /list_status - List all agent statuses")
                print("  /clear - Clear the screen")
                print("  /help - Show this help message")
                
            case _:
                # Default case - send to default agent
                await handle_direct_message("Aithor", user_input, conversation_id)

if __name__ == "__main__":
    asyncio.run(live_interact())
