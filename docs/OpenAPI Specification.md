# OpenAPI Specification Guide

## Overview
The Agent API system provides a distributed architecture for AI agents to communicate and interact with each other. The system consists of two main components:
1. Agent API - Individual agent endpoints for receiving messages and managing status
2. Directory Service API - Central service for agent registration, message routing, and system-wide operations

The system is designed to facilitate seamless communication between multiple AI agents while maintaining a centralized registry for discovery and routing.

## API Information
- **Title**: Agent APIs
- **Version**: 0.1.0

## Directory Service Endpoints
The Directory Service acts as the central hub for agent management and communication routing.

### Agent Management
1. **POST** `/agent/register`
   - Register a new agent in the directory
   - Requires AgentDirectory object
   - Used when agents start up to announce their presence and capabilities
   - Stores agent metadata including location, type, and available tools

2. **GET** `/agent/lookup`
   - Lookup registered agents
   - Optional query parameter: `agent_name`
   - Returns full agent directory when no name specified
   - Returns specific agent details when name provided

3. **GET** `/agent/status/all`
   - Get current status of all registered agents
   - Performs health checks on each registered agent
   - Returns detailed status including any error conditions

### Message Routing
1. **POST** `/agent/message`
   - Route messages between agents
   - Implements retry logic with exponential backoff
   - Handles timeouts and connection errors
   - Requires APIMessage object

2. **POST** `/agent/feedback`
   - Route feedback between agents
   - Enables agents to provide performance feedback to each other
   - Supports learning and improvement mechanisms
   - Requires FeedbackMessage object

### System Operations
1. **GET** `/health`
   - System health check
   - Verifies Directory Service is operational

2. **GET** `/collections`
   - List all collections and their document counts
   - Provides overview of stored knowledge bases
   - Interfaces with ChromaDB for vector storage

3. **GET** `/collections/{collection_name}`
   - Get documents and metadata from a specific collection
   - Retrieves detailed collection contents
   - Returns documents, metadata, and IDs

## Agent Endpoints
Each individual agent exposes these endpoints for communication and management.

### Message Handling
1. **POST** `/receive`
   - Handle incoming messages
   - Processes messages with configurable timeout
   - Supports various message types
   - Requires APIMessage object

### Status Management
1. **POST** `/status`
   - Update agent's status
   - Validates status transitions
   - Maintains agent state consistency
   - Query parameter: `status` (AgentStatus enum)

### Health
1. **GET** `/health`
   - Agent health check
   - Returns current status and state
   - Used for monitoring and diagnostics

## Request/Response Models

### AgentDirectory
Represents an agent's registration information:
```json
{
  "properties": {
    "name": {"type": "string"},
    "address": {"type": "string"},
    "port": {"type": "integer"},
    "agent_type": {"type": "string"},
    "status": {"type": "string", "default": "active"},
    "description": {"type": "string"},
    "tools": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["name", "address", "port", "agent_type", "description", "tools"]
}
```

### APIMessage
Defines the structure for inter-agent communication:
```json
{
  "properties": {
    "sender": {"type": "string"},
    "receiver": {"type": "string"},
    "content": {"type": "string"},
    "timestamp": {"type": "string"},
    "message_type": {"type": "string", "default": "text"},
    "conversation_id": {"type": "string"}
  },
  "required": ["sender", "receiver", "content", "timestamp", "conversation_id"]
}
```

### FeedbackMessage
Enables performance feedback between agents:
```json
{
  "properties": {
    "sender": {"type": "string"},
    "receiver": {"type": "string"},
    "conversation_id": {"type": "string"},
    "score": {"type": "integer"},
    "feedback": {"type": "string"},
    "timestamp": {"type": "string"}
  },
  "required": ["sender", "receiver", "conversation_id", "score", "feedback", "timestamp"]
}
```

### AgentStatus
Defines possible agent states with valid transitions:
- idle: Default state when not processing
- processing: Actively handling a request
- waiting: Waiting for external response
- learning: Updating internal knowledge
- shutting_down: Graceful shutdown in progress
- memorising: Storing new information
- socialising: Engaging with other agents

## Error Handling
The system implements comprehensive error handling:
- Retry logic with exponential backoff for failed requests
- Timeout handling for long-running operations
- Standard HTTP error responses with validation
- Detailed error logging and tracking

## Security Considerations
1. Consider implementing authentication for agent registration
2. Validate message routing permissions
3. Implement rate limiting for production use
4. Secure inter-agent communication channels
5. Monitor and log all system activities

## Implementation Notes
1. The system uses a distributed architecture with individual agent instances
2. The Directory Service acts as a central router and registry
3. Agents can communicate directly or through the directory service
4. Support for different message types and feedback mechanisms
5. Flexible status management for complex agent behaviors
6. Integrated logging system with configurable levels
7. ChromaDB integration for vector storage and retrieval
8. Configurable timeouts and retry mechanisms
