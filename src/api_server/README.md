# Agent API Server

## Overview
The Agent API system provides a distributed architecture for AI agents to communicate and interact with each other. The system consists of two main components:
1. Agent API - Individual agent endpoints for receiving messages and managing status
2. Directory Service API - Central service for agent registration, message routing, and system-wide operations

## Usage

1. Start the directory service:
```Bash
python -m src.api_server.server
```

2. Access the Directory Service APIs:
- Agent Registration: `http://localhost:8000/agent/register`
- Agent Lookup: `http://localhost:8000/agent/lookup`
- Message Routing: `http://localhost:8000/agent/message`
- Feedback Routing: `http://localhost:8000/agent/feedback`
- Agent Status: `http://localhost:8000/agent/status/all`
- Health Check: `http://localhost:8000/health`
- Collections: `http://localhost:8000/collections`

3. OpenAPI documentation is available at:
```
http://localhost:8000/docs
http://localhost:8010/docs
```

## Key Features

### Directory Service
- Centralized agent registration and discovery
- Message routing between agents with retry logic
- Agent status monitoring and health checks
- Feedback routing system
- Collection management for vector storage

### Agent Endpoints
- Message receiving and processing
- Status management with state transitions
- Health monitoring
- Feedback processing

## Configuration
- Server configuration is centralized in `config.py`
- Logging levels can be set via environment variables:
  - `SERVER_LOG_LEVEL`: Controls server logging (default: "INFO")
  - `AGENT_LOG_LEVEL`: Controls agent logging (default: "DEBUG")
  - `CONSOLE_LOGGING`: Enable/disable console output (default: "True")

## Technical Details
- Built with FastAPI and Uvicorn
- Async/await for concurrent operations
- Input/output models defined using Pydantic
- Comprehensive error handling with retries
- Integrated logging system
- ChromaDB integration for vector storage
- Configurable timeouts and retry mechanisms

## Error Handling
- Automatic retry with exponential backoff for failed requests
- Timeout handling for long-running operations
- Detailed error logging and tracking
- Standard HTTP error responses

## Security Notes
- Authentication should be implemented for production use
- Rate limiting recommended for public deployments
- Secure communication channels should be configured
- Activity monitoring and logging enabled by default

## Models
All API models are defined using Pydantic and include:
- AgentDirectory: Agent registration information
- APIMessage: Inter-agent communication structure
- FeedbackMessage: Performance feedback between agents
- AgentStatus: Agent state management

For detailed API specifications and models, refer to the OpenAPI documentation at `/docs` endpoint.