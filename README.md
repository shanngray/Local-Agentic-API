# AI Agent Interaction System

A multi-agent system that enables collaborative problem-solving through specialized AI agents. The system features a directory service for agent discovery and communication, with persistent memory storage using ChromaDB.

## Features

- **Multi-Agent Architecture**: Multiple specialized agents working together
- **Directory Service**: Central service for agent discovery and message routing
- **Persistent Memory**: ChromaDB-based memory system for agents
- **Interactive Interface**: Command-line interface for direct agent interaction
- **Tool Integration**: Extensible tool system for agent capabilities
- **Health Monitoring**: Built-in health checks and status monitoring
- **Logging System**: Comprehensive logging with configurable levels

## Prerequisites

- Python 3.x
- Poetry (Python dependency management)
- Required API keys:
  - OpenAI API key
  - Tavily API key (for web search capabilities)

## Setup

1. Clone this repository

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create a `.env` file based on `.env.sample`:
   ```
   OPENAI_API_KEY=your_openai_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

## Running the System

The system requires two terminals to run:

1. Start the main system (directory service and agents):
   ```bash
   poetry run python src/main.py
   ```

2. Launch the interaction interface:
   ```bash
   poetry run python src/live_interact.py
   ```

## Available Agents

- **MathAgent**: Mathematical expert for calculations and problem-solving
- **TimeAgent**: Time and timezone specialist
- **EditorAgent**: Expert editor for proofreading and editing blog posts

## Interactive Commands

- `/lookup` - List all available agents
- `@agent_name message` - Send direct message to agent
- `@agent_name/status/new_status` - Update agent status
- `/show <collection_name>` - Show documents in collection
- `/list_col` - List all collections
- `/list_status` - List all agent statuses
- `/clear` - Clear the screen
- `/help` - Show help message

## API Documentation

- API specifications and OpenAPI documentation available at:
  - Directory Service: `http://127.0.0.1:8000/docs`
  - Individual Agent APIs: Available at their respective ports

## Memory Management

The system uses ChromaDB for persistent storage with two types of memory:
- Short-term memory: Recent interactions and temporary data
- Long-term memory: Consolidated and processed information

## Development

### Environment Variables

- `AGENT_DEBUG`: Enable debug mode (default: false)
- `AGENT_LOG_LEVEL`: Set logging level (default: INFO)
- `SERVER_LOG_LEVEL`: Set server logging level (default: INFO)
- `CONSOLE_LOGGING`: Enable console logging (default: True)

### Adding New Agents

Create new agent files in `src/agents/` following the existing patterns:

'''python
def create_new_agent(chroma_client: PersistentClient) -> BaseAgent:
config = AgentConfig(
agent_name="NewAgent",
description="Description of the agent's capabilities",
enabled_tools=["list_of", "enabled", "tools"],
api_port=unique_port_number
)
return BaseAgent(api_key=os.getenv("OPENAI_API_KEY"),
chroma_client=chroma_client,
config=config)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
