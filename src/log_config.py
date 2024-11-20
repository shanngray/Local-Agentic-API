import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text

def create_logger_config(
    name: str,
    log_path: Path,
    level: str = "INFO",
    console_logging: bool = True
) -> dict:
    """Create standardized logging configuration."""
    log_path.mkdir(exist_ok=True)
    
    handlers = {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(log_path / f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            "formatter": "standard",
            "level": level
        }
    }
    
    if console_logging:
        handlers["console"] = {
            "class": "rich.logging.RichHandler",
            "formatter": "rich",
            "level": level
        }
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "rich": {
                "format": "%(message)s"
            }
        },
        "handlers": handlers,
        "loggers": {
            name: {
                "handlers": list(handlers.keys()),
                "level": level,
                "propagate": False
            }
        }
    }

def setup_logger(
    name: str,
    log_path: Optional[Path] = None,
    level: str = "INFO",
    console_logging: bool = True
) -> logging.Logger:
    """Setup and return a configured logger."""
    if log_path is None:
        log_path = Path("logs")
    
    # Get the logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    config = create_logger_config(name, log_path, level, console_logging)
    logging.config.dictConfig(config)
    
    # Get the configured logger
    logger = logging.getLogger(name)
    
    # Ensure propagate is False to prevent duplicate logging
    logger.propagate = False
    
    return logger

# Event emojis for WebSocket and agent events
EVENT_EMOJIS = {
    # WebSocket events
    "session.update": "🛠️",
    "session.created": "🔌",
    "session.updated": "🔄",
    # Agent events
    "agent.registered": "🤖📝",
    "agent.message_sent": "🤖💬",
    "agent.message_received": "🤖🗨️",
    "agent.tool_called": "🤖🔧",
    "agent.error": "🤖❌",
    # agent status events
    "agent.idle": "🤖💤",
    "agent.processing": "🤖💫",
    "agent.waiting": "🤖🕒",
    "agent.learning": "🤖📚",
    "agent.shutting_down": "🤖🛑",
    "agent.memorising": "🤖💾",
    "agent.memorising.complete": "🧠🧠🧠",
    "agent.socialising": "🤖👥",
    # Tool events
    "tool.success": "✅",
    "tool.error": "🔧❌",
    # Directory service events
    "directory.lookup": "🔍",
    "directory.register": "📋",
    "directory.route": "🔄",
    # Additional WebSocket events
    "session.started": "🎬",
    "session.completed": "🏁",
    "session.timeout": "⏰",
    "session.iteration": "🔄",
    # Additional Tool events
    "tool.loading": "📥",
    "tool.registered": "📝",
    "tool.executing": "⚙️",
    "tool.warning": "⚠️",
    # Message events
    "message.content": "💬",
    # Memory events
    "memory.loaded": "💾",
    "memory.saved": "💾",
    "memory.error": "💾❌",
    "memory.cleanup": "🧠🧹",
    "tool.called": "🔧",
    # Queue events
    "queue.added": "📊➕",
    "queue.dequeued": "📊⬇️",
    "queue.processing": "📊🔄",
    "queue.completed": "📊✅",
    "queue.error": "📊❌",
    "queue.status": "📊📋",
    # Server events
    "server.startup": "🚀",
    "server.shutdown": "🛑",
    "server.request": "📨",
    "server.response": "📩",
    "server.error": "💥",
    # Directory events
    "directory.startup": "📖",
    "directory.shutdown": "📕",
    "directory.agent_registered": "📝",
    "directory.agent_lookup": "🔍",
    "directory.message_route": "📫💬",
    "directory.route_error": "📮💬❌",
    "directory.feedback_route": "📫📝",
    "directory.feedback_error": "📮📝❌",
    # Health check events
    "health.check": "💓",
    "health.error": "💔",
}

def log_event(logger: logging.Logger, event_type: str, details: str, level: str = "INFO") -> None:
    """Log an event with appropriate emoji and styling.
    
    Args:
        logger: Logger instance
        event_type: Type of event (used to determine emoji)
        details: Event details
        level: Log level to use
    """
    emoji = EVENT_EMOJIS.get(event_type, "ℹ️")
    style = {
        "DEBUG": "dim",
        "INFO": "bold blue",
        "WARNING": "bold yellow",
        "ERROR": "bold red"
    }.get(level.upper(), "default")
    
    log_func = getattr(logger, level.lower())
    log_func(Text(f"{emoji} {details}", style=style))

# Convenience functions for common log types
def log_tool_call(logger: logging.Logger, tool_name: str, args: dict, result: Optional[dict] = None) -> None:
    """Log a tool call with arguments and optional result."""
    log_event(logger, "tool.called", f"Tool: {tool_name}, Args: {args}")
    if result:
        log_event(logger, "tool.success", f"Result: {result}")

def log_agent_message(logger: logging.Logger, direction: str, sender: str, receiver: str, content: str) -> None:
    """Log agent message communication."""
    event_type = "agent.message_sent" if direction == "out" else "agent.message_received"
    log_event(logger, event_type, f"{sender} → {receiver}: {content}")

def log_error(logger: logging.Logger, message: str, exc_info: Optional[Exception] = None) -> None:
    """Log an error with optional exception info."""
    log_event(logger, "agent.error", message, level="ERROR")
    if exc_info:
        logger.exception(exc_info)


'''
Here’s a breakdown of when to use each level and what types of messages they display:

	1.	DEBUG (logging.DEBUG)
	•	Purpose: Use DEBUG for low-level information meant for developers to diagnose issues or understand the internal state of the application.
	•	Messages Visible: DEBUG, INFO, WARNING, ERROR, CRITICAL
	•	Example Use: Tracking variable values, flow through functions, or confirming specific conditions met in code.
	2.	INFO (logging.INFO)
	•	Purpose: Use INFO to record general events in the application’s flow, typically milestones or key points in processing that are helpful for understanding normal operation.
	•	Messages Visible: INFO, WARNING, ERROR, CRITICAL
	•	Example Use: Start and end of main operations, successful connections, or user actions that indicate progression through the application.
	3.	WARNING (logging.WARNING)
	•	Purpose: Use WARNING when something unexpected occurs or when a situation might lead to an error if not addressed, but the application can still proceed.
	•	Messages Visible: WARNING, ERROR, CRITICAL
	•	Example Use: Low disk space, retry attempts, or deprecated function usage.
	4.	ERROR (logging.ERROR)
	•	Purpose: Use ERROR to record problems that prevent part of the program from functioning correctly. The application may still continue, but something significant has gone wrong.
	•	Messages Visible: ERROR, CRITICAL
	•	Example Use: Failure to connect to a database, file not found, or invalid data causing a function to fail.
	5.	CRITICAL (logging.CRITICAL)
	•	Purpose: Use CRITICAL for severe errors that likely indicate a fatal issue, leading the application to stop or malfunction significantly.
	•	Messages Visible: CRITICAL
	•	Example Use: System crashes, critical resource unavailability, or security breaches.
'''