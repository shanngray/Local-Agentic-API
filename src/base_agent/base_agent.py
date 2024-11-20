import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from openai import AsyncOpenAI
import os
import sys
import importlib.util
from pathlib import Path
import httpx
import atexit
import signal
import traceback
import uuid
from fastapi import HTTPException
import inspect
from chromadb import PersistentClient
from filelock import FileLock

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from .models import Message, ToolCall, AgentStatus, Request
from .config import AgentConfig
from api_server.models.api_models import APIMessage
from src.log_config import setup_logger, log_tool_call, log_agent_message, log_error, log_event
from src.memory.memory_manager import MemoryManager

class ToolError(Exception):
    """Custom exception for tool-related errors."""
    pass

class BaseAgent:
    def __init__(
        self,
        api_key: str,
        chroma_client: PersistentClient,
        config: Optional[AgentConfig] = None
    ):
        """Initialize the base agent with OpenAI client and configuration."""
        super().__init__()
        
        self.config = config or AgentConfig()
        
        # Setup folder for agent files
        self.files_path = Path("agent_files") / self.config.agent_name
        self.files_path.mkdir(parents=True, exist_ok=True)

        self.client = AsyncOpenAI(api_key=api_key)
        self.status = AgentStatus.IDLE
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.waiting_for: Optional[str] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        
        # Initialize internal tools dictionary ONCE at the start
        self.internal_tools: Dict[str, Callable] = {}
        
        # Initialize logger before any other operations
        self.logger = setup_logger(
            name=self.config.agent_name,
            log_path=self.config.log_path,
            level=self.config.log_level,
            console_logging=self.config.console_logging
        )
        
        self._setup_logging()
        
        # Add send_message to internal tools
        self.internal_tools["send_message"] = self.send_message
        self.internal_tools["search_memory"] = self.search_memory
        
        # Register all internal tools AFTER adding them to internal_tools
        for tool_name, tool_func in self.internal_tools.items():
            self.register_tool(tool_func)
            
        # Load enabled external tools
        if self.config.enabled_tools:
            self._load_tool_definitions()

        self.conversation_histories: Dict[str, List[Message]] = {}
        self.agent_directory = None  # Will be set during registration

        # Initialize memory manager with default collections
        self.memory = MemoryManager(
            agent_name=self.config.agent_name,
            logger=self.logger,
            chroma_client=chroma_client
        )
        
        # Initialize collections including feedback
        asyncio.create_task(self.memory.initialize(
            collection_names=["short_term", "long_term", "feedback"]
        ))
        
        # Replace single messages list with conversations dict
        self.conversations: Dict[str, List[Message]] = {}
        self.default_conversation_id = "default"
        
        # Initialize default conversation with system prompt
        self.conversations[self.default_conversation_id] = [
            Message(role="system", content=self.config.system_prompt)
        ]
        
        # Initialize conversation state
        self.friends: Dict[str, Dict[str, Any]] = {}

        # TODO: Turn back on after testing
        # Load existing memory if available
        #self._load_memory()
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._previous_status = None
        self._status_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    def _setup_logging(self) -> None:
        """Configure logging if debug is enabled."""
        if self.config.debug:
            self.config.log_path.mkdir(exist_ok=True)
            
            # Only add handlers if none exist
            if not self.logger.handlers:
                # Always add file handler when debug is enabled
                file_handler = logging.FileHandler(
                    self.config.log_path / f'{self.config.agent_name.lower().replace(" ", "_")}_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                )
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
                # Add console handler only if enabled
                if self.config.console_logging:
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                    
                self.logger.setLevel(logging.DEBUG)
                
            self.logger.info("BaseAgent initialized")

        log_event(self.logger, "agent.registered", f"Initialized {self.config.agent_name}")

    def _load_tool_definitions(self) -> None:
        """Load tool definitions from JSON files."""
        for tool_name in self.config.enabled_tools:
            try:
                tool_path = self.config.tools_path / f"{tool_name}.json"
                if not tool_path.exists():
                    raise ToolError(f"Tool definition not found: {tool_path}")

                log_event(self.logger, "tool.loading", f"Loading tool definition: {tool_path}")
                with open(tool_path) as f:
                    tool_def = json.load(f)
                    self.register_tool(tool_def)
                    
            except Exception as e:
                log_event(self.logger, "tool.error", f"Failed to load tool {tool_name}: {str(e)}")

    async def _load_tool(self, tool_name: str) -> Callable:
        """Load a tool by name, preferring internal tools over external ones.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            Callable: The loaded tool function
            
        Raises:
            ToolError: If tool cannot be loaded
        """
        try:
            # First check for internal tool
            if tool_name in self.internal_tools:
                log_event(self.logger, "tool.loading", f"Loading internal tool: {tool_name}", level="DEBUG")
                return self.internal_tools[tool_name]
            
            # Fall back to external tool
            log_event(self.logger, "tool.loading", f"Loading external tool: {tool_name}", level="DEBUG")
            if str(self.config.tools_path) not in sys.path:
                sys.path.append(str(self.config.tools_path))
            
            spec = importlib.util.spec_from_file_location(
                tool_name,
                str(self.config.tools_path / f"{tool_name}.py")
            )
            if not spec or not spec.loader:
                raise ToolError(f"Could not find tool module: {tool_name}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            tool_func = getattr(module, tool_name)
            log_event(self.logger, "tool.success", f"Successfully loaded external tool: {tool_name}")
            return tool_func
                
        except Exception as e:
            log_event(self.logger, "tool.error", f"Failed to load tool {tool_name}: {str(e)}")
            raise ToolError(f"Failed to load tool {tool_name}: {str(e)}")

    async def _execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        await self.set_status(AgentStatus.WAITING)
        tool_name = tool_call.function["name"]
        tool_args = tool_call.function["arguments"]
        
        log_event(self.logger, "tool.executing", 
                  f"Starting execution of tool: {tool_name}", level="DEBUG")
        
        if tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}"
            log_event(self.logger, "tool.error", error_msg, level="ERROR")
            raise ToolError(error_msg)
        
        try:
            # Load the tool
            log_event(self.logger, "tool.loading", 
                     f"Loading tool implementation: {tool_name}", level="DEBUG")
            tool_func = await self._load_tool(tool_name)
            
            # Parse arguments
            try:
                args = json.loads(tool_args)
                log_event(self.logger, "tool.executing", 
                         f"Parsed arguments for {tool_name}: {args}", level="DEBUG")
            except json.JSONDecodeError as e:
                error_msg = f"Invalid tool arguments: {tool_args}"
                log_event(self.logger, "tool.error", error_msg, level="ERROR")
                raise ToolError(error_msg) from e
            
            # Execute the tool
            source = "internal" if tool_name in self.internal_tools else "external"
            log_event(self.logger, "tool.executing", 
                     f"Executing {source} tool {tool_name} with args: {args}")
            
            result = await tool_func(**args)
            
            # Log the successful execution and result
            log_tool_call(self.logger, tool_name, args, result)
            log_event(self.logger, "tool.success", 
                     f"Tool {tool_name} executed successfully")
            
            return result
                
        except ToolError:
            raise
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            log_error(self.logger, error_msg, exc_info=e)
            raise ToolError(error_msg)
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(AgentStatus.PROCESSING)

    def register_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register multiple tools/functions that the agent can use.
        
        Args:
            tools: List of tool definitions to register
            
        Raises:
            ValueError: If any tool definition is invalid
        """
        for tool in tools:
            self.register_tool(tool)
            
    def register_tool(self, tool: Union[Dict[str, Any], Callable]) -> None:
        """Register a new tool/function that the agent can use.
        
        Args:
            tool: Either a tool definition dictionary or a callable method
        """
        if callable(tool):
            # Handle internal tool (method)
            tool_name = tool.__name__
            if tool_name in self.tools:
                self.logger.warning(f"External tool {tool_name} will be overridden by internal method")
            
            # Get function signature and docstring for parameters
            sig = inspect.signature(tool)
            doc = inspect.getdoc(tool) or ""
            
            # Create tool definition from method
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": doc.split("\n")[0],  # First line of docstring
                    "parameters": {
                        "type": "object",
                        "properties": {
                            param.name: {"type": "string"}  # Simplified - could be enhanced
                            for param in sig.parameters.values()
                        },
                        "required": [
                            param.name
                            for param in sig.parameters.values()
                            if param.default == inspect.Parameter.empty
                        ]
                    }
                }
            }
            
            self.internal_tools[tool_name] = tool
            self.tools[tool_name] = tool_def
            log_event(self.logger, "tool.registered", f"Registered internal tool: {tool_name}")
            
        else:
            # Handle external tool definition (existing logic)
            if "function" not in tool:
                tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                
            # Validate tool definition
            required_fields = {"name", "description", "parameters"}
            if not all(field in tool["function"] for field in required_fields):
                raise ValueError(f"Tool definition missing required fields: {required_fields}")
                
            tool_name = tool["function"]["name"]
            if tool_name in self.internal_tools:
                log_event(self.logger, "tool.warning", 
                         f"External tool {tool_name} will not override existing internal method")
                return
                
            if tool_name in self.tools:
                raise ValueError(f"Tool {tool_name} already registered")
                
            self.tools[tool_name] = tool
            log_event(self.logger, "tool.registered", f"Registered external tool: {tool_name}")

    def _trim_history(self, conversation_id: str) -> None:
        """Trim conversation history to max_history length."""
        if conversation_id not in self.conversations:
            return
            
        messages = self.conversations[conversation_id]
        if len(messages) > self.config.max_history:
            self.logger.debug(f"Trimming history from {len(messages)} messages")
            self.conversations[conversation_id] = [messages[0]] + messages[-(self.config.max_history-1):]

    async def process_message(self, content: str, sender: str, conversation_id: str) -> str:
        if self.status == AgentStatus.SHUTTING_DOWN:
            raise ValueError("Agent is shutting down")

        await self.set_status(AgentStatus.PROCESSING)
        
        try:
            log_event(self.logger, "session.started", 
                      f"Processing message from {sender} in conversation {conversation_id}")
            log_event(self.logger, "message.content", f"Content: {content}", level="DEBUG")

            if conversation_id == "New":
                conversation_id = str(uuid.uuid4())
                log_event(self.logger, "session.created", f"Created new conversation: {conversation_id}")
            
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = [
                    Message(role="system", content=self.config.system_prompt)
                ]
            
            messages = self.conversations[conversation_id]
            
            # Add user message if not duplicate
            last_message = messages[-1] if messages else None
            if not last_message or last_message.content != content:
                user_message = Message(
                    role="user",
                    content=content,
                    sender=sender,
                    name=None,
                    tool_calls=None,
                    tool_call_id=None,
                    receiver=self.config.agent_name,
                    timestamp=datetime.now().isoformat()
                )
                messages.append(user_message)
                log_event(self.logger, "message.added", 
                         f"Added user message to conversation {conversation_id}")

            final_response = None
            iteration_count = 0
            max_iterations = 5
            while iteration_count < max_iterations:
                try:
                    log_event(self.logger, "openai.request", 
                             f"Sending request to OpenAI with {len(messages)} messages")
                    
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.config.model,
                            temperature=self.config.temperature,
                            messages=[m.dict() for m in messages],
                            tools=list(self.tools.values()) if self.tools else None,
                            timeout=300
                        ),
                        timeout=350
                    )
                    
                    raw_message = response.choices[0].message
                    message_content = raw_message.content or ""
                    
                    # Create and add assistant message
                    assistant_message = Message(
                        role="assistant",
                        content=message_content,
                        tool_calls=[ToolCall(
                            id=tc.id,
                            type=tc.type,
                            function={
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        ) for tc in raw_message.tool_calls] if raw_message.tool_calls else None,
                        sender=self.config.agent_name,
                        receiver=sender,
                        timestamp=datetime.now().isoformat(),
                        name=None,
                        tool_call_id=None
                    )
                    messages.append(assistant_message)
                    log_event(self.logger, "message.added", 
                             f"Added assistant message to conversation {conversation_id}")
                    
                    # Process tool calls if present
                    if raw_message.tool_calls:
                        log_event(self.logger, "tool.processing", 
                                 f"Processing {len(raw_message.tool_calls)} tool calls")
                        
                        tool_calls = [ToolCall(
                            id=tc.id,
                            type=tc.type,
                            function={
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        ) for tc in raw_message.tool_calls]
                        
                        # Execute each tool and add results
                        for tool_call in tool_calls:
                            try:
                                tool_result = await self._execute_tool(tool_call)
                                tool_message = Message(
                                    role="tool",
                                    content=json.dumps(tool_result),
                                    tool_call_id=tool_call.id,
                                    timestamp=datetime.now().isoformat(),
                                    name=None,
                                    tool_calls=None,
                                    sender=self.config.agent_name,
                                    receiver=sender
                                )
                                messages.append(tool_message)
                                log_event(self.logger, "tool.result", 
                                         f"Added tool result for {tool_call.function['name']}")
                                
                            except ToolError as e:
                                error_message = Message(
                                    role="tool",
                                    content=json.dumps({"error": str(e)}),
                                    tool_call_id=tool_call.id,
                                    timestamp=datetime.now().isoformat(),
                                    name=None,
                                    tool_calls=None,
                                    sender=self.config.agent_name,
                                    receiver=sender
                                )
                                messages.append(error_message)
                                log_error(self.logger, 
                                         f"Tool error in {tool_call.function['name']}: {str(e)}")
                        
                        iteration_count += 1
                        continue  # Continue loop to process tool results
                    
                    # If we have a message without tool calls, evaluate to see if we should continue or stop
                    if message_content:
                        continue_or_stop = await self._continue_or_stop(messages)
                        if continue_or_stop == "STOP":
                            final_response = message_content
                            log_event(self.logger, "message.final", 
                                     f"Final response generated for conversation {conversation_id}")
                            break
                        else:
                            # Wrap and append non-STOP message
                            messages.append(Message(
                                role="user",
                                content=continue_or_stop,
                                timestamp=datetime.now().isoformat(),
                                name=self.config.agent_name,
                                tool_calls=None,
                                tool_call_id=None,
                                sender=self.config.agent_name,
                                receiver=self.config.agent_name
                            ))
                            log_event(self.logger, "message.continue", 
                                     f"Added intermediate response for conversation {conversation_id}")
                    
                except asyncio.TimeoutError as e:
                    log_error(self.logger, "OpenAI request timed out", exc_info=e)
                    raise
                except Exception as e:
                    log_error(self.logger, "Error during message processing", exc_info=e)
                    raise
                
                iteration_count += 1
            
            if iteration_count >= max_iterations:
                log_event(self.logger, "message.max_iterations", 
                         f"Reached maximum iterations ({max_iterations}) for conversation {conversation_id}")
                final_response = "I apologize, but I've reached the maximum number of iterations for this conversation. Is there anything specific I can clarify or help with?"
            # Trim history if needed
            self._trim_history(conversation_id)
            
            if final_response is None:
                error_msg = "Failed to generate a proper response"
                log_error(self.logger, error_msg)
                return "I apologize, but I wasn't able to generate a proper response."
            
            return final_response
                
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(AgentStatus.IDLE)

    async def _continue_or_stop(self, messages: List[Message]) -> str:
        """Evaluate message content to determine if we should continue or stop."""
        """Evaluate message content to determine if we should continue or stop.
        
        Args:
            messages: List of conversation messages to evaluate
            
        Returns:
            "STOP" if conversation should end, or a rephrased continuation message
        """
        try:
            # Create prompt to evaluate conversation state
            prompt = Message(
                role="user", 
                content=(
                    "Based on the conversation so far, determine if a complete response has been provided "
                    "or if more follow-up is needed. If complete, respond with exactly 'STOP'. "
                    "If more follow-up is needed, rephrase where the conversation should go next "
                    "as a clear question or instructions."
                )
            )
            
            # Make LLM call to evaluate
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[msg.dict() for msg in messages] + [prompt.dict()],
                temperature=0.7
            )
            
            response = response.choices[0].message.content.strip()
            
            # Log the continuation decision
            if response == "STOP":
                log_event(self.logger, "conversation.complete", "Conversation marked as complete")
            else:
                log_event(self.logger, "conversation.continue", f"Continuing conversation: {response}")
                
            return response
            
        except Exception as e:
            log_error(self.logger, "Error evaluating conversation continuation", exc_info=e)
            return "STOP"  # Default to stopping on error

        return response

    async def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        try:
            while True:
                user_input = input(f"\n{self.config.agent_name}: How can I help you? (type 'exit' to quit): ")
                
                if user_input.lower().strip() in {"no", "end", "quit", "bye", "exit", "goodbye", "done"}:
                    print(f"\n{self.config.agent_name}: Goodbye! Have a great day!")
                    break
                    
                response = await self.process_message(user_input, sender="user")
                print(f"\n{self.config.agent_name}: {response}")
                
        except KeyboardInterrupt:
            print("\n\nExiting due to user interrupt...")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

    # TODO: We might want to get rid of this. Would need to update process_message and would need to make sure that all messages have the sender and receiver fields
    def get_conversation_history(self, agent_name: str) -> List[Message]:
        """Get messages related to a specific agent across all conversations."""
        related_messages = []
        for messages in self.conversations.values():
            related_messages.extend([
                msg for msg in messages 
                if (msg.role == "system") or  # Include system prompt
                   (hasattr(msg, 'sender') and msg.sender == agent_name) or
                   (hasattr(msg, 'receiver') and msg.receiver == agent_name)
            ])
        return related_messages

    async def receive_message(self, sender: str, content: str, conversation_id: str) -> str:
        """Process received message from another agent."""
        log_event(self.logger, "agent.message_received", 
                  f"Message from {sender} in conversation {conversation_id}")
        log_event(self.logger, "message.content", f"{content}", level="DEBUG")
        
        # Get current queue size before adding new message
        queue_size = self.request_queue.qsize()
        queue_position = queue_size + 1
        
        # Initialize conversation if needed
        if conversation_id not in self.conversations:
            log_event(self.logger, "session.created", 
                     f"Creating new conversation: {conversation_id}")
            self.conversations[conversation_id] = [
                Message(role="system", content=self.config.system_prompt)
            ]

        # Update friends list
        current_time = datetime.now().isoformat()
        if sender not in self.friends or self.friends[sender]["last_contact"] != current_time:
            log_event(self.logger, "directory.register", 
                     f"Updating friend record for {sender}")
            self.friends[sender] = {
                "name": sender,
                "last_contact": current_time
            }

        # Create request ID and future
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Queue the request
        request = {
            "id": request_id,
            "sender": sender,
            "content": content,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "queue_position": queue_position
        }
        log_event(self.logger, "queue.added", 
                  f"Queuing request {request_id} from {sender} (position {queue_position} in queue)")
        await self.request_queue.put(request)
        
        try:
            # Wait for response with timeout
            log_event(self.logger, "queue.status", 
                     f"Request {request_id} waiting at position {queue_position}", level="DEBUG")
            response = await asyncio.wait_for(future, timeout=300.0)
            log_event(self.logger, "queue.completed", 
                     f"Request {request_id} completed successfully (was position {queue_position})")
            return response
        except asyncio.TimeoutError:
            log_event(self.logger, "session.timeout", 
                     f"Request {request_id} timed out after 30 seconds (was position {queue_position})", 
                     level="WARNING")
            raise
        finally:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                log_event(self.logger, "queue.status", 
                         f"Cleaned up request {request_id} (was position {queue_position})", 
                         level="DEBUG")

    async def send_message(self, receiver: str, content: str) -> Dict[str, Any]:
        """Send a message via API to another agent registered in the directory service."""
        await self.set_status(AgentStatus.WAITING)
        sender = self.config.agent_name

        if sender == receiver:
            log_event(self.logger, "agent.error", 
                      f"Cannot send message to self: {sender} -> {receiver}", 
                      level="ERROR")
            raise ValueError(f"Cannot send message to self: {sender} -> {receiver}")
        
        # Create a new conversation ID for each message 
        conversation_id = str(uuid.uuid4())
        log_agent_message(self.logger, "out", sender, receiver, content)
        
        async with httpx.AsyncClient(timeout=500.0) as client:
            message = APIMessage.create(
                sender=sender,
                receiver=receiver,
                content=content,
                conversation_id=conversation_id
            )
            
            try:
                log_event(self.logger, "directory.route", 
                         f"Sending message to directory service", level="DEBUG")
                log_event(self.logger, "directory.route", 
                         f"Request payload: {message.dict()}", level="DEBUG")
                
                response = await client.post(
                    "http://localhost:8000/agent/message",
                    json=message.dict(),
                    timeout=500.0
                )
                
                log_event(self.logger, "directory.route", 
                         f"Received response: Status {response.status_code}", level="DEBUG")
                log_event(self.logger, "directory.route", 
                         f"Response content: {response.text}", level="DEBUG")
                
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    log_error(self.logger, f"HTTP error occurred: {str(e)}")
                    log_error(self.logger, f"Response content: {response.text}")
                    raise
                
                result = response.json()
                message_content = result.get("message", "")
                if isinstance(message_content, dict):
                    message_content = message_content.get("message", "")
                
                log_agent_message(self.logger, "in", receiver, sender, message_content)
                
                # Evaluate response quality in background
                asyncio.create_task(self._evaluate_and_send_feedback(
                    receiver=receiver,
                    conversation_id=conversation_id,
                    response_content=message_content
                ))
                
                return {
                    "role": "assistant",
                    "content": message_content,
                    "sender": receiver,
                    "receiver": sender,
                    "timestamp": result.get("timestamp") or datetime.now().isoformat()
                }
                
            except httpx.TimeoutException as e:
                error_msg = f"Request timed out while sending message to {receiver}"
                log_error(self.logger, error_msg, exc_info=e)
                return {
                    "role": "error",
                    "content": f"{error_msg}: {str(e)}",
                    "sender": "system",
                    "receiver": sender,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                log_error(self.logger, f"Unexpected error in send_message", exc_info=e)
                raise
            finally:
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.PROCESSING)

    async def _evaluate_and_send_feedback(
        self,
        receiver: str,
        conversation_id: str,
        response_content: str
    ) -> None:
        """Evaluate response quality and send feedback to the agent."""
        try:
            # Get evaluation from LLM
            score, feedback = await self._evaluate_response(response_content)
            
            # Send feedback via API
            async with httpx.AsyncClient() as client:
                feedback_message = {
                    "sender": self.config.agent_name,
                    "receiver": receiver,
                    "conversation_id": conversation_id,
                    "score": score,
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat()
                }
                
                await client.post(
                    "http://localhost:8000/agent/feedback",
                    json=feedback_message,
                    timeout=30.0
                )
                
            log_event(self.logger, "feedback.sent",
                     f"Sent feedback to {receiver} for conversation {conversation_id}")
                     
        except Exception as e:
            log_error(self.logger, f"Failed to process/send feedback: {str(e)}")

    async def _evaluate_response(self, response_content: str) -> Tuple[int, str]:
        """Evaluate response quality using LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": """Evaluate the quality of the following response. 
                    Provide:
                    1. A score from 0-10 (where 10 is excellent)
                    2. Brief, constructive feedback (Include WHAT is good/bad, WHY it matters, and HOW to improve)
                    
                    Format your response as a JSON object with 'score' and 'feedback' fields."""},
                    {"role": "user", "content": f"Response to evaluate:\n{response_content}"}
                ],
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["score"], result["feedback"]
            
        except Exception as e:
            log_error(self.logger, f"Failed to evaluate response: {str(e)}")
            return 5, "Error evaluating response"  # Default neutral score

    async def receive_feedback(
        self,
        sender: str,
        conversation_id: str,
        score: int,
        feedback: str
    ) -> None:
        """Handle received feedback from another agent."""
        try:
            # Store feedback in memory
            metadata = {
                "sender": sender,
                "conversation_id": conversation_id,
                "score": score,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.memory.store(
                content=feedback,
                collection_name="feedback",
                metadata=metadata
            )
            
            log_event(self.logger, "feedback.received",
                     f"Received feedback from {sender} for conversation {conversation_id}")
                     
        except Exception as e:
            log_error(self.logger, f"Failed to process received feedback: {str(e)}")

    # TODO: We need to update this to load from the memory store
    def _load_memory(self) -> None:
        """Load conversation histories and friends from disk."""
        try:
            if self.messages_file.exists():
                with open(self.messages_file) as f:
                    conversations_data = json.load(f)
                    self.conversations = {
                        conv_id: [Message(**msg) for msg in messages]
                        for conv_id, messages in conversations_data.items()
                    }
                self.logger.info(f"Loaded {len(self.conversations)} conversations from memory")
            
            if self.friends_file.exists():
                with open(self.friends_file) as f:
                    self.friends = json.load(f)
                self.logger.info(f"Loaded {len(self.friends)} friends from memory")
                    
        except Exception as e:
            self.logger.error(f"Error loading memory: {str(e)}")
            # Continue with empty state if load fails

    async def _save_memory(self) -> None:
        """Save all conversations to memory store."""
        try:
            for conversation_id, conversation in self.conversations.items():
                content = "\n".join([
                    f"{msg.role}: {msg.content}" 
                    for msg in conversation 
                    if msg.role != "system"
                ])
                
                # Convert participants list to comma-separated string
                participants = set(
                    getattr(msg, 'sender', None) or msg.role 
                    for msg in conversation
                )
                participants_str = ",".join(sorted(participants))
                
                metadata = {
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "participants": participants_str  # Now a string instead of a list
                }
                
                try:
                    await self.memory.store(
                        content=content,
                        collection_name="short_term",
                        metadata=metadata
                    )
                    log_event(self.logger, "memory.saved", 
                             f"Saved conversation {conversation_id} to memory")
                    # Also save memory dump to disk
                    memory_dump_path = self.files_path / f"{self.config.agent_name}_short_term_memory_dump.md"
                    try:
                        with FileLock(f"{memory_dump_path}.lock"):
                            with open(memory_dump_path, "a", encoding="utf-8") as f:
                                # Add timestamp header
                                f.write(f"\n\n## Memory Entry {datetime.now().isoformat()}\n")
                                for key, value in metadata.items():
                                    f.write(f"{key}: {value}\n")
                                f.write("\n")
                                f.write(content)
                                f.write("\n---\n")
                        log_event(self.logger, "memory.dumped", 
                                f"Saved conversation {conversation_id} to disk")
                    except Exception as e:
                        log_error(self.logger,
                                f"Failed to save conversation {conversation_id} to disk: {str(e)}")
                except Exception as e:
                    log_error(self.logger, 
                             f"Failed to save conversation {conversation_id}: {str(e)}")
                    
        except Exception as e:
            log_error(self.logger, "Failed to save memory", exc_info=e)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully by creating async task."""
        self.logger.info("Shutdown signal received")
        # Create async task for shutdown which includes memory saving
        asyncio.create_task(self.shutdown())

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        await self.set_status(AgentStatus.SHUTTING_DOWN)
        self._shutdown_event.set()
        
        self.logger.info(f"Shutting down {self.config.agent_name}")
        # Save memory as part of shutdown
        await self._save_memory()
        
        # Cancel any pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        
        # Clear queues
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def start(self):
        """Start the agent's main processing loop and initialise memories"""
        self.logger.info(f"Starting {self.config.agent_name} processing loop")
        
        # Initialize memory collections
        await self.memory.initialize()

        while not self._shutdown_event.is_set():
            try:
                await self.process_queue()
                await asyncio.sleep(0.1)  # Prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.IDLE)

    async def process_queue(self):
        """Process any pending requests in the queue"""
        if self.status == AgentStatus.SHUTTING_DOWN:
            return

        if not self.request_queue.empty():
            current_queue_size = self.request_queue.qsize()
            request = await self.request_queue.get()
            queue_position = request.get('queue_position', 'unknown')
            
            await self.set_status(AgentStatus.PROCESSING)
            try:
                log_event(self.logger, "queue.dequeued", 
                         f"Processing request {request['id']} from {request['sender']} "
                         f"(position {queue_position}/{current_queue_size} in queue)")
                
                log_event(self.logger, "queue.processing", 
                         f"Processing message content: {request['content']} "
                         f"(was position {queue_position})", level="DEBUG")
                response = await self.process_message(
                    content=request["content"],
                    sender=request["sender"],
                    conversation_id=request["conversation_id"]
                )
                
                # Complete the future with the response
                if request["id"] in self.pending_requests:
                    self.pending_requests[request["id"]].set_result(response)
                    log_event(self.logger, "queue.completed", 
                             f"Request {request['id']} processed successfully "
                             f"(was position {queue_position}/{current_queue_size})")
                    del self.pending_requests[request["id"]]
                    
            except Exception as e:
                log_error(self.logger, 
                         f"Error processing request {request['id']} "
                         f"(was position {queue_position}/{current_queue_size}): {str(e)}", 
                         exc_info=e)
                if request["id"] in self.pending_requests:
                    self.pending_requests[request["id"]].set_exception(e)
                    del self.pending_requests[request["id"]]
            finally:
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.IDLE)

    async def set_status(self, new_status: AgentStatus) -> None:
        """Update agent status with validation and logging."""
        async with self._status_lock:
            if new_status == self.status:
                return

            valid_transitions = AgentStatus.get_valid_transitions(self.status)
            if new_status not in valid_transitions:
                raise ValueError(
                    f"Invalid status transition from {self.status} to {new_status}"
                )

            self._previous_status = self.status
            self.status = new_status
            log_event(
                self.logger, 
                f"agent.{self.status}", 
                f"Status changed from {self._previous_status} to {self.status}"
            )
            
            # Trigger memory consolidation when entering MEMORISING state
            if new_status == AgentStatus.MEMORISING:
                asyncio.create_task(self._transfer_to_long_term())

    async def _transfer_to_long_term(self) -> None:
        """Transfer short-term memories into long-term storage by extracting structured facts and metadata."""
        try:
            log_event(self.logger, "agent.memorising", "Beginning memory consolidation process")
            
            # Retrieve recent memories from short-term storage
            short_term_memories = await self.memory.retrieve(
                query="",  # Empty query to get all memories
                collection_names=["short_term"],
                n_results=100,
                metadata_filter={
                    "timestamp": {
                        "$gt": (datetime.now() - timedelta(days=1)).isoformat()
                    }
                }
            )
            
            if not short_term_memories:
                log_event(self.logger, "memory.error", "No recent memories found for consolidation")
                await self.set_status(self._previous_status)
                return

            # Process each memory individually
            for memory in short_term_memories:
                try:
                    # Extract structured information using LLM
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": """Analyze this memory and extract structured information in the following format:
                                {
                                    "facts": [list of factual statements],
                                    "decisions": [list of decisions made],
                                    "preferences": [list of expressed preferences],
                                    "patterns": [list of observed patterns],
                                    "metadata": {
                                        "topics": [relevant topics],
                                        "entities": [mentioned entities],
                                        "relationships": [identified relationships],
                                        "sentiment": overall_sentiment,
                                        "importance": importance_score (1-10)
                                    }
                                }
                            """},
                            {"role": "user", "content": f"Memory content:\n{memory['content']}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    structured_info = json.loads(response.choices[0].message.content)
                    
                    # Create metadata for storage
                    metadata = {
                        "memory_id": memory["metadata"].get("id", str(uuid.uuid4())),
                        "original_timestamp": memory["metadata"].get("timestamp"),
                        "source_type": "structured_memory",
                        "topics": structured_info["metadata"]["topics"],
                        "entities": structured_info["metadata"]["entities"],
                        "importance": structured_info["metadata"]["importance"],
                        "timestamp": datetime.now().isoformat()
                    }

                    # Store each type of information separately
                    for fact in structured_info["facts"]:
                        await self.memory.store(
                            content=fact,
                            collection_name="long_term",
                            metadata={**metadata, "type": "fact"}
                        )
                    
                    for decision in structured_info["decisions"]:
                        await self.memory.store(
                            content=decision,
                            collection_name="long_term", 
                            metadata={**metadata, "type": "decision"}
                        )

                    # ... store preferences and patterns similarly ...
                    
                    log_event(self.logger, "agent.memorising",
                             f"Processed memory {metadata['memory_id']} into structured long-term storage")

                    # Save to disk for debugging/backup
                    await self._save_memory_to_disk(structured_info, metadata)

                except Exception as e:
                    log_error(self.logger, f"Failed to process memory: {str(e)}", exc_info=e)
                    continue
                
            await self._cleanup_short_term_memories()
            await self._save_memory_to_disk(structured_info, metadata)
            
            log_event(self.logger, "memory.memorising.complete",
                     f"Completed memory consolidation for {len(short_term_memories)} memories")
                 
        except Exception as e:
            log_error(self.logger, "Failed to consolidate memories", exc_info=e)
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(self._previous_status)

    async def _save_memory_to_disk(self, structured_info: Dict, metadata: Dict) -> None:
        """Save structured memory information to disk for debugging/backup."""
        try:
            memory_dump_path = self.files_path / f"{self.config.agent_name}_long_term_memory_dump.md"
            with FileLock(f"{memory_dump_path}.lock"):
                with open(memory_dump_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n## Structured Memory Entry {metadata['timestamp']}\n")
                    f.write("### Metadata\n")
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            value = ", ".join(value)
                        f.write(f"{key}: {value}\n")
                    
                    f.write("\n### Extracted Information\n")
                    for category in ["facts", "decisions", "preferences", "patterns"]:
                        if structured_info.get(category):
                            f.write(f"\n#### {category.title()}\n")
                            for item in structured_info[category]:
                                f.write(f"- {item}\n")
                    f.write("\n---\n")
                
            log_event(self.logger, "memory.dumped", 
                     f"Saved structured memory {metadata['memory_id']} to disk")
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save memory {metadata['memory_id']} to disk: {str(e)}")

    async def _cleanup_short_term_memories(self) -> None:
        """Clean up old short-term memories after consolidation."""
        try:
            # Currently deleting all memories once they are processed into long_term memory
            old_memories = await self.memory.retrieve(
                query="",
                collection_names=["short_term"],
                metadata_filter={
                    "timestamp": {
                        "$lt": (datetime.now() - timedelta(days=0)).isoformat()
                    }
                }
            )
            
            # Delete old memories
            for memory in old_memories:
                memory_id = memory["metadata"].get("id")
                if memory_id:
                    await self.memory.delete(memory_id, "short_term")
                
            log_event(self.logger, "memory.cleanup", 
                     f"Cleaned up {len(old_memories)} old short-term memories")
                 
        except Exception as e:
            log_error(self.logger, "Failed to cleanup short-term memories", exc_info=e)

    #TODO: Need to fix internal tools so that the descriptions and tool definitions work properly.
    async def search_memory(self, query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use to search your memory using a 'query' and (optional) 'keywords'.
        
        Args:
            query: The search query string
            keywords: Optional list of keywords to help filter results
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Create metadata filter if keywords provided
            metadata_filter = None
            if keywords:
                # Use $in operator instead of $contains for keyword matching
                # This will match if any of the keywords exactly match the content
                metadata_filter = {
                    "content": {"$in": keywords}
                }

            # Search both short and long term memory
            memories = await self.memory.retrieve(
                query=query,
                collection_names=["short_term", "long_term"],
                n_results=5,
                metadata_filter=metadata_filter
            )

            log_event(self.logger, "memory.searched", 
                     f"Memory search for '{query}' returned {len(memories)} results")

            return {
                "query": query,
                "results": memories,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Failed to search memory: {str(e)}"
            log_error(self.logger, error_msg, exc_info=e)
            return {
                "query": query,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        